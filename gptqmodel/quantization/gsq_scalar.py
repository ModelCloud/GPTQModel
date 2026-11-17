# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in refinement on an existing GPTQ scalar grid.

The checkpoint still uses GPTQ scales, zero-points, group indices and packers.
This is a local reconstruction objective, not the paper's full block-training
schedule. The baseline is retained unless a hard, export-aware candidate wins.

Integer codes are searched by Gauss-Seidel coordinate descent scored on the
exported hard weight, and every output row is accepted separately. The
Gumbel-Softmax relaxation is not used to move codes: sampling a soft mixture of
codes descends an expectation that the packer never stores, so from a GPTQ
initializer it only ever lost to the initializer and the refiner returned it
unchanged (#4592). Scales are continuous and are refit in closed form instead.
"""

from dataclasses import dataclass

import pcre
import torch

from ..utils.gemv import awq_gemv_codes
from ..utils.packing_math import awq_packed_codes, awq_storage_dtype
from .config import normalize_gsq_config


# Sweep cap for the hard code search. Real layers converge within a handful of
# sweeps, so this bounds the added cost to a small multiple of one quantization
# pass instead of running the configured step budget with no effect.
_HARD_SWEEP_CAP = 8
# Columns advanced together per sweep. Wider blocks need more sweeps for the
# same objective; narrower blocks cost more scalar launches.
_HARD_SWEEP_BLOCK = 8
# Code moves per column, counted as the half width of the search window. Only
# immediate neighbours transfer off the calibration Gram: wider windows lower
# the calibration score further but lose accuracy on held-out activations.
_HARD_SWEEP_WINDOW = 1
# Fractional hard-score gain a row must clear before its codes are replaced.
# The exported objective is measured on the calibration Gram, so a marginal
# gain there is inside the sampling noise of that Gram; only large row gains
# reproduced on held-out activations in the measurements behind this constant.
_HARD_ACCEPT_MARGIN = 0.05
# Scale refits move the grid under the codes, so they alternate with the code
# sweeps. Each round is one code sweep plus one closed-form scale refit; the
# measured gain is reached in two and later rounds only re-optimize.
_HARD_SCALE_ROUNDS = 2


@dataclass
class ScalarGSQResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor
    before: float | None
    after: float | None
    history: list[float]


def asymmetric_error_term(weight_error, teacher, cross_moment, alpha=1.0):
    """Unnormalized linear correction for native versus propagated inputs."""
    if weight_error.ndim != 2 or teacher.shape != weight_error.shape:
        raise ValueError("Asymmetric GSQ requires matching weight errors and teachers")
    width = teacher.shape[1]
    if cross_moment.shape != (width, width):
        raise ValueError("Asymmetric GSQ cross moment must be [in,in]")
    if any(t.device != teacher.device for t in (weight_error, cross_moment)):
        raise ValueError("Asymmetric GSQ tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all()
               for t in (weight_error, teacher, cross_moment)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("Asymmetric GSQ requires finite floating inputs")
    return -2 * alpha * (weight_error * (teacher @ cross_moment)).sum()


def gsq_enabled_for(config, module_name: str) -> bool:
    config = normalize_gsq_config(config)
    return config is not None and config.enabled and (
        config.modules is None or any(pcre.search(pattern, module_name) for pattern in config.modules)
    )


def affine_codes(weight, scales, zeros, g_idx, bits, *, packing, scale_dtype=torch.float16):
    """Recover integer codes using the arithmetic used by the target packer."""
    if packing == "gptq":
        scale = scales[:, g_idx].float()
        values = (weight.float() + zeros[:, g_idx].float() * scale) / scale
    elif packing == "awq_gemm":
        return awq_packed_codes(
            weight, scales, zeros, g_idx, bits=bits,
            scale_dtype=awq_storage_dtype(scale_dtype, gemm=True),
        )
    elif packing in ("awq_gemv", "awq_gemv_fast"):
        if bits != 4 or scale_dtype != torch.float16:
            raise ValueError("AWQ GEMV GSQ requires 4 bits and FP16 scales")
        return awq_gemv_codes(weight, scales, zeros, g_idx)
    else:
        raise ValueError("Unknown affine GSQ packing format")
    # GPTQ's pack_original clamps its codes.
    return values.round().clamp(0, 2**bits - 1)


def _metric_factor(hessian: torch.Tensor) -> torch.Tensor:
    """Factor a positive-semidefinite calibration Gram without extra damping."""
    h = (hessian.float() + hessian.float().T) * 0.5
    factor, info = torch.linalg.cholesky_ex(h)
    if int(info) == 0:
        return factor
    values, vectors = torch.linalg.eigh(h.double())
    tolerance = torch.finfo(torch.float32).eps * h.shape[0] * values.abs().max()
    if values.min() < -tolerance:
        raise ValueError("GSQ requires a positive-semidefinite calibration Hessian")
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)).float()


def refine_gptq_scalar(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    target: torch.Tensor,
    bits: int,
    config,
    hessian: torch.Tensor,
    scale_dtype: torch.dtype = torch.float16,
) -> ScalarGSQResult:
    """Fit codes and optional scales, then accept only a better hard export.

    ``weight`` is the ordinary GPTQ initializer. ``target`` and ``hessian``
    use the original input-column order, including when GPTQ used act order.
    No score is interpreted as model quality; it is a local reconstruction
    objective weighted by the calibration Gram.
    """
    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return ScalarGSQResult(weight.clone(), scales.clone(), zeros.clone(), g_idx.clone(), None, None, [])
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("GSQ bits must be an integer in [2, 8]")
    if weight.ndim != 2 or target.shape != weight.shape or not weight.is_floating_point():
        raise ValueError("GSQ requires floating [out,in] weights and a matching teacher")
    rows, columns = weight.shape
    if scales.ndim != 2 or scales.shape[0] != rows or zeros.shape != scales.shape:
        raise ValueError("GSQ scales and zeros must have shape [out,groups]")
    if g_idx.shape != (columns,) or g_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("GSQ requires an integer group index for each input column")
    if bool((g_idx < 0).any()) or bool((g_idx >= scales.shape[1]).any()):
        raise ValueError("GSQ group index is outside the scale table")
    if hessian.shape != (columns, columns):
        raise ValueError("GSQ Hessian must have shape [in,in]")
    tensors = (weight, target, scales, zeros, g_idx, hessian)
    if any(t.device != weight.device for t in tensors):
        raise ValueError("GSQ tensors must share a device")
    if not all(t.is_floating_point() and bool(torch.isfinite(t).all()) for t in (weight, target, scales, zeros, hessian)):
        raise ValueError("GSQ weights, target, scales, zeros and Hessian must be finite")
    if bool((scales <= 0).any()) or bool((zeros != zeros.round()).any()):
        raise ValueError("GSQ requires positive scales and integer zero-points")
    if bool((zeros < 0).any()) or bool((zeros > 2**bits - 1).any()):
        raise ValueError("GSQ zero-points must be within the target grid")

    choices = min(2**bits, config.candidates)
    bytes_per_row = choices * columns * 4
    if bytes_per_row > config.max_candidate_bytes:
        raise ValueError("GSQ candidate bank for one output row exceeds max_candidate_bytes")
    rows_per_chunk = max(1, config.max_candidate_bytes // bytes_per_row)

    with torch.inference_mode(False), torch.enable_grad():
        group = g_idx.detach().long().clone()
        factor = _metric_factor(hessian.detach().float())
        refined_weights = []
        refined_scales = []
        before_sum = 0.0
        after_sum = 0.0
        energy_sum = 0.0
        weighted_history = [0.0] * (config.steps + 1)
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            result, energy = _refine_rows(
                weight[row_start:row_end], scales[row_start:row_end], zeros[row_start:row_end],
                target[row_start:row_end], group, factor, bits, choices, config, scale_dtype,
                row_start,
            )
            refined_weights.append(result.weight)
            refined_scales.append(result.scales)
            normalizer = energy if energy > torch.finfo(torch.float32).eps else 1.0
            before_sum += result.before * normalizer
            after_sum += result.after * normalizer
            energy_sum += normalizer
            for step, score in enumerate(result.history):
                weighted_history[step] += score * normalizer
        return ScalarGSQResult(
            torch.cat(refined_weights), torch.cat(refined_scales), zeros.detach().clone(),
            g_idx.detach().clone(), before_sum / energy_sum, after_sum / energy_sum,
            [score / energy_sum for score in weighted_history],
        )


def hard_code_sweep(
    codes: torch.Tensor,
    column_scale: torch.Tensor,
    *,
    metric: torch.Tensor,
    teacher: torch.Tensor,
    decode,
    row_loss,
    bits: int,
    offsets: torch.Tensor,
    block: int = _HARD_SWEEP_BLOCK,
) -> torch.Tensor:
    """One Gauss-Seidel sweep over exported integer codes, accepted per row.

    ``decode(codes)`` returns the weight the packer stores for those codes and
    ``row_loss(weight)`` returns one hard reconstruction score per output row.
    ``column_scale`` holds the stored scale of every ``[out, in]`` column, so a
    code step of ``d`` moves the reconstruction by ``column_scale * d``;
    ``metric`` is the factor ``row_loss`` projects through.

    Blocks advance one after another inside the sweep, so the residual
    projection carries the off-diagonal Gram coupling that earlier blocks
    already moved. A row keeps its new codes only when its exact hard score
    improves, so one bad row cannot veto the rest of the chunk.
    """
    limit = 2**bits - 1
    # ``row_loss`` projects through ``metric``, so the Gram is ``metric @
    # metric.T`` and its diagonal is the column sum of ``metric`` squared.
    curvature = metric.square().sum(dim=1)
    current_weight = decode(codes)
    best_codes = codes.clone()
    best_rows = row_loss(current_weight)
    if not bool(torch.isfinite(best_rows).all()):
        raise ValueError("non-finite GSQ hard objective")
    current = codes.clone()
    projected = (current_weight.float() - teacher) @ metric
    for start in range(0, codes.shape[1], block):
        stop = min(codes.shape[1], start + block)
        width = stop - start
        gradient = projected @ metric[start:stop].T
        shifted = current[:, start:stop].unsqueeze(-1) + offsets.view(1, 1, -1)
        valid = (shifted >= 0) & (shifted <= limit)
        shifted = shifted.clamp(0, limit)
        delta = column_scale[:, start:stop].unsqueeze(-1) * (
            shifted - current[:, start:stop].unsqueeze(-1)
        )
        step = delta.square() * curvature[start:stop].view(1, width, 1)
        step = step + 2 * delta * gradient.unsqueeze(-1)
        step = step.masked_fill(~valid, torch.inf)
        choice = step.argmin(-1)
        proposed = shifted.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        move = column_scale[:, start:stop] * (proposed - current[:, start:stop])
        current[:, start:stop] = proposed
        projected = projected + move @ metric[start:stop]
    candidate_rows = row_loss(decode(current))
    if not bool(torch.isfinite(candidate_rows).all()):
        raise ValueError("non-finite GSQ hard objective")
    accepts = candidate_rows < best_rows
    return torch.where(accepts.unsqueeze(-1), current, best_codes)


def _refine_rows(weight, scales, zeros, target, group, factor, bits, choices, config, scale_dtype, row_start):
    rows, columns = weight.shape
    teacher = target.detach().float().clone()
    baseline = weight.detach().clone()
    base_scales = scales.detach().float().clone()
    zero = zeros.detach().float().clone()
    teacher_projected = teacher @ factor
    teacher_energy = teacher_projected.square().sum()
    if not bool(torch.isfinite(teacher_energy)) or float(teacher_energy) <= torch.finfo(torch.float32).eps:
        # A near-zero teacher is not a useful normalization. Score the
        # unnormalized quadratic, which keeps the ordering and the sign.
        teacher_energy = torch.ones_like(teacher_energy)

    def loss(matrix):
        return ((matrix.float() - teacher) @ factor).square().sum() / teacher_energy

    def row_loss(matrix):
        # One raw projection score per output row. Summing these and dividing
        # by the shared teacher energy reproduces ``loss`` exactly, so a
        # per-row acceptance decision is the same objective the checkpoint is
        # reported against.
        error = matrix.float() - teacher
        return (error @ factor).square().sum(dim=1)

    def recover_codes(candidate_weight, candidate_scales):
        # Match pack_original: the scale/zero product, addition and division
        # use the producer tensor dtypes before the integer rounding step.
        native_scale = candidate_scales.to(scales.dtype)[:, group]
        native_zero = zeros[:, group]
        offset = native_zero * native_scale
        return ((candidate_weight + offset) / native_scale).round().clamp(0, 2**bits - 1).float()

    def decode(codes, candidate_scales):
        stored = candidate_scales.to(scale_dtype).float()
        if not bool(torch.isfinite(stored).all()) or bool((stored <= 0).any()):
            raise ValueError("GSQ scales must be finite and positive in the checkpoint dtype")
        return stored[:, group] * (codes - zero[:, group])

    def exported(candidate_weight, candidate_scales):
        return decode(recover_codes(candidate_weight, candidate_scales), candidate_scales)

    base_codes = recover_codes(baseline, scales)
    before = float(loss(exported(baseline, scales)))
    if not torch.isfinite(torch.tensor(before)):
        raise ValueError("non-finite GSQ baseline objective")

    current_codes = base_codes.clone()
    current_scales = base_scales.clone()
    best_rows = row_loss(decode(current_codes, current_scales))
    history = []
    offsets = torch.tensor(
        [0] + [((-1) ** i) * ((i + 1) // 2) for i in range(1, 2 * _HARD_SWEEP_WINDOW + 1)],
        device=weight.device,
    )

    # Hard code search on the exported weight. A GPTQ initializer is already a
    # local optimum of the same Gram, so an accepted move needs the exported
    # score of a complete candidate; the relaxed expectation of a code
    # distribution cannot describe the export and is never used to accept one.
    #
    # Two candidate families are scored every round and merged per row: codes
    # alone, and codes with a refit scale table. Refitting scales moves the
    # grid under the codes, so the accepted state always has its codes
    # re-searched at those scales before the next round.
    def score_rows(candidate_codes, candidate_scales):
        return row_loss(decode(candidate_codes, candidate_scales))

    best_codes = current_codes
    best_scales = current_scales
    rounds = min(config.steps, _HARD_SCALE_ROUNDS) if config.learn_scales else min(config.steps, _HARD_SWEEP_CAP)
    floor = 1.0 - _HARD_ACCEPT_MARGIN
    for _ in range(rounds):
        # Candidate 1: move codes on the current scale table.
        searched = hard_code_sweep(
            current_codes,
            current_scales[:, group],
            metric=factor,
            teacher=teacher,
            decode=lambda codes: decode(codes, current_scales),
            row_loss=lambda codes: score_rows(codes, current_scales),
            bits=bits,
            offsets=offsets,
        )
        current_codes = searched
        code_rows = score_rows(current_codes, current_scales)
        accepts = code_rows < best_rows * floor
        best_codes = torch.where(accepts.unsqueeze(-1), current_codes, best_codes)
        best_scales = torch.where(accepts.unsqueeze(-1), current_scales, best_scales)
        best_rows = torch.where(accepts, code_rows, best_rows)
        history.append(float(best_rows.sum() / teacher_energy))
        if not config.learn_scales or not bool(accepts.any()):
            break
        # Candidate 2: refit the scale table under those codes. Scales move the
        # grid, so the next round re-searches the codes on the new table.
        scaled = _fit_group_scales(
            current_codes, zero, group, teacher, factor, decode,
            scale_dtype, current_scales,
        )
        scale_rows = score_rows(current_codes, scaled)
        accepts = scale_rows < best_rows * floor
        if not bool(accepts.any()):
            break
        best_codes = torch.where(accepts.unsqueeze(-1), current_codes, best_codes)
        best_scales = torch.where(accepts.unsqueeze(-1), scaled, best_scales)
        best_rows = torch.where(accepts, scale_rows, best_rows)
        current_scales = torch.where(accepts.unsqueeze(-1), scaled, current_scales)
        history.append(float(best_rows.sum() / teacher_energy))

    # Final per-row acceptance against the initializer, so a row that got worse
    # somewhere in the loop keeps its GPTQ export.
    final_rows = score_rows(best_codes, best_scales)
    keep = final_rows <= best_rows
    best_codes = torch.where(keep.unsqueeze(-1), best_codes, base_codes)
    best_scales = torch.where(keep.unsqueeze(-1), best_scales, base_scales)
    best_weight = decode(best_codes, best_scales)
    # The aggregate history has one entry per configured step plus the
    # initializer. A converged sweep stops early and scale learning may add its
    # own entries, so normalize the length here instead of per chunk.
    history = history[:config.steps]
    while len(history) < config.steps:
        history.append(history[-1] if history else before)

    # Score the checkpoint that is actually written: the search runs in FP32,
    # so casting to the model dtype can round a code across a half step.
    exported_weight = best_weight.detach().to(weight.dtype)
    after = float(loss(exported(exported_weight, best_scales)))
    if not torch.isfinite(torch.tensor(after)):
        raise ValueError("non-finite GSQ hard objective")
    if after > before:
        raise ValueError("GSQ hard export regressed from its initializer")
    return ScalarGSQResult(
        exported_weight,
        best_scales.detach().to(scales.dtype),
        zeros.detach().clone(),
        group.detach().clone(),
        before,
        after,
        [before] + history,
    ), float(teacher_energy)


def _fit_group_scales(codes, zero, group, teacher, metric, decode, scale_dtype, fallback):
    """Closed-form per-group scale refit with the integer codes held fixed.

    The objective is quadratic in one group scale, but a group's columns still
    enter every output coordinate through the Gram, so the minimizer is not a
    block-diagonal ratio. Hold the group's own reconstruction at zero, project
    the remaining residual, and solve the resulting scalar least squares:

    ``s* = -<d metric, residual> / ||d metric||^2`` over that group's columns.

    Codes are frozen, which is what makes this exact; it also keeps the cost to
    one projection per group instead of a learning-rate schedule.
    """
    # Start from the incoming table so groups the search never visits
    # (padding rows, unused ids) keep their stored value.
    stored = fallback.to(codes.dtype).clone()
    current = decode(codes, fallback)
    for group_id in sorted({int(value) for value in group.tolist()}):
        columns = (group == group_id).nonzero().squeeze(-1)
        delta_codes = codes[:, columns] - zero[:, group_id].unsqueeze(1)
        # Residual with this group's own contribution removed, in metric space.
        without = current.clone()
        without[:, columns] = 0
        residual = (without.float() - teacher) @ metric
        basis = delta_codes @ metric[columns]
        numerator = -(basis * residual).sum(dim=1)
        denominator = basis.square().sum(dim=1)
        scale = torch.where(
            denominator > torch.finfo(torch.float32).eps,
            numerator / denominator.clamp_min(torch.finfo(torch.float32).eps),
            fallback[:, group_id],
        )
        stored[:, group_id] = scale
    refined = stored.to(scale_dtype).float()
    usable = torch.isfinite(refined).all(dim=1) & (refined > 0).all(dim=1)
    refined = torch.where(usable.unsqueeze(1), refined, fallback.to(scale_dtype).float())
    if not bool(torch.isfinite(decode(codes, refined)).all()):
        raise ValueError("GSQ scale refit produced a non-finite export")
    return refined


def refine_affine_scalar(
    weight, scales, zeros, g_idx, *, target, bits, config=None,
    hessian=None, inputs=None, packing="gptq", scale_dtype=torch.float16,
    cross_moment=None, cross_alpha=1.0, _metric=None, _row_start=0,
):
    """Refine [out,in] weights and [out,groups] scales on their actual grid.

    ``g_idx`` maps original columns to groups, including GPTQ activation-order
    permutations. Supply either real activations or their Hessian; neither means
    weight reconstruction (the explicit calibration-free RTN objective).
    Hard checkpoints are scored after weight casting, code reconstruction by the
    selected packer convention, and scale casting to the checkpoint dtype.
    The original baseline tensors are returned unchanged unless that score improves.
    An optional native-minus-current cross moment adds the asymmetric linear
    term. Scores then omit a candidate-independent constant and can be negative;
    they are not absolute normalized reconstruction errors.
    """
    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return ScalarGSQResult(weight.clone(), scales.clone(), zeros.clone(), g_idx.clone(), None, None, [])
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("scalar GSQ bits must be an integer in [1,8]")
    if packing in ("awq_gemm", "awq_gemv", "awq_gemv_fast"):
        if packing != "awq_gemm" and scale_dtype != torch.float16:
            raise ValueError("AWQ GEMV GSQ requires FP16 scales")
        # Resolve the checkpoint scale dtype once so code recovery and the
        # export score use the same stored values the packer will write.
        scale_dtype = awq_storage_dtype(scale_dtype, gemm=(packing == "awq_gemm"))
    if weight.ndim != 2 or target.shape != weight.shape or not weight.is_floating_point():
        raise ValueError("scalar GSQ requires floating [out,in] weights and a matching teacher")
    n, k = weight.shape
    if scales.ndim != 2 or scales.shape[0] != n or zeros.shape != scales.shape:
        raise ValueError("scalar GSQ scales/zeros must have shape [out,groups]")
    if g_idx.shape != (k,) or g_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("scalar GSQ g_idx must be an integer vector of input width")
    if g_idx.min() < 0 or g_idx.max() >= scales.shape[1]:
        raise ValueError("scalar GSQ group index is outside the scale table")
    tensors = (weight, target, scales, zeros, g_idx)
    if any(t.device != weight.device for t in tensors):
        raise ValueError("scalar GSQ tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in tensors[:-1]):
        raise ValueError("scalar GSQ weights, target, scales and zeros must be finite floating point")
    if (scales <= 0).any() or (zeros != zeros.round()).any() or (zeros < 0).any() or (zeros > 2**bits-1).any():
        raise ValueError("scalar GSQ requires positive scales and in-range integer zero-points")
    if hessian is not None and inputs is not None:
        raise ValueError("supply inputs or hessian, not both")
    if hessian is not None and (hessian.shape not in ((k, k), (k,)) or not torch.isfinite(hessian).all()):
        raise ValueError("scalar GSQ Hessian must be finite [in,in] or diagonal [in]")
    if hessian is not None and hessian.ndim == 1 and (hessian < 0).any():
        raise ValueError("scalar GSQ diagonal Hessian must be nonnegative")
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[1] != k or inputs.shape[0] == 0
                               or not torch.isfinite(inputs).all()):
        raise ValueError("scalar GSQ inputs must be finite nonempty [tokens,in]")
    count = min(2**bits, config.candidates)
    bytes_per_row = count * k * 4
    if bytes_per_row > config.max_candidate_bytes:
        raise ValueError("scalar GSQ candidate bank for one output row exceeds max_candidate_bytes")
    rows_per_chunk = max(1, config.max_candidate_bytes // bytes_per_row)
    if n > rows_per_chunk:
        with torch.inference_mode(False), torch.enable_grad():
            if _metric is None:
                if hessian is not None:
                    h = hessian.detach().to(device=weight.device, dtype=torch.float32).clone()
                    _metric = (None, h.sqrt()) if h.ndim == 1 else (_metric_factor(h), None)
                elif inputs is not None:
                    _metric = (inputs.detach().to(device=weight.device, dtype=torch.float32).T.contiguous().clone(), None)
                else:
                    _metric = (None, None)
            results = []
            normalizers = []
            factor, diagonal = _metric
            for start in range(0, n, rows_per_chunk):
                end = min(start + rows_per_chunk, n)
                part = refine_affine_scalar(
                    weight[start:end], scales[start:end], zeros[start:end], g_idx,
                    target=target[start:end], bits=bits, config=config, hessian=hessian,
                    inputs=inputs, packing=packing, scale_dtype=scale_dtype,
                    cross_moment=cross_moment, cross_alpha=cross_alpha,
                    _metric=_metric, _row_start=_row_start + start,
                )
                teacher = target[start:end].float()
                projected = teacher * diagonal if diagonal is not None else (
                    teacher if factor is None else teacher @ factor)
                energy = float(projected.square().sum())
                normalizers.append(energy if energy > torch.finfo(torch.float32).eps else 1.0)
                results.append(part)
            total = sum(normalizers)
            return ScalarGSQResult(
                torch.cat([part.weight for part in results]),
                torch.cat([part.scales for part in results]), zeros.detach().clone(), g_idx.detach().clone(),
                sum(part.before * norm for part, norm in zip(results, normalizers)) / total,
                sum(part.after * norm for part, norm in zip(results, normalizers)) / total,
                [sum(part.history[step] * norm for part, norm in zip(results, normalizers)) / total
                 for step in range(config.steps + 1)],
            )

    with torch.inference_mode(False), torch.enable_grad():
        teacher = target.detach().float().clone()
        groups = g_idx.detach().long().clone()
        base_scales = scales.detach().float().clone()
        zero = zeros.detach().float().clone()
        baseline = weight.detach().clone()
        factor = None
        diagonal = None
        if _metric is not None:
            factor, diagonal = _metric
        elif hessian is not None:
            h = hessian.detach().to(device=weight.device, dtype=torch.float32).clone()
            if h.ndim == 1:
                diagonal = h.sqrt()
            else:
                factor = _metric_factor(h)
        elif inputs is not None:
            factor = inputs.detach().to(device=weight.device, dtype=torch.float32).T.contiguous().clone()
        def project(matrix):
            if diagonal is not None:
                return matrix * diagonal
            return matrix if factor is None else matrix @ factor

        teacher_output = project(teacher)
        energy = teacher_output.square().sum()
        # Near-zero teacher energy is not a useful normalization scale. Use
        # the unnormalized quadratic, retaining any asymmetric linear term.
        denominator = torch.where(energy > torch.finfo(torch.float32).eps, energy, torch.ones_like(energy))
        correction = None
        if cross_moment is not None:
            if hessian is None and inputs is None:
                raise ValueError("asymmetric GSQ requires matching calibration moments or inputs")
            cross = cross_moment.detach().to(device=weight.device, dtype=torch.float32).clone()
            asymmetric_error_term(torch.zeros_like(teacher), teacher, cross, cross_alpha)
            correction = cross_alpha * (teacher @ cross)

        def loss(matrix):
            error = project(matrix - teacher)
            value = error.square().sum()
            if correction is not None:
                value = value - 2 * ((matrix-teacher) * correction).sum()
            return value / denominator

        def recover_codes(candidate_weight, candidate_scales):
            values = affine_codes(candidate_weight, candidate_scales, zero, groups, bits,
                                  packing=packing, scale_dtype=scale_dtype)
            # Packing arithmetic keeps its native dtype above. Optimization
            # needs FP32: Adam's epsilon underflows in FP16 for local grids.
            return values.float()

        def decode(codes, candidate_scales):
            stored_scales = candidate_scales.to(scale_dtype).float()
            if not torch.isfinite(stored_scales).all() or (stored_scales <= 0).any():
                raise ValueError("scalar GSQ scales are not finite positive in the checkpoint dtype")
            if packing == "awq_gemv_fast":
                offset = -(stored_scales * zero).half().float()
                if not torch.isfinite(offset).all():
                    raise ValueError("scalar GSQ GEMV_FAST offsets must be finite in FP16 storage")
                return stored_scales[:, groups] * codes + offset[:, groups]
            return stored_scales[:, groups] * (codes - zero[:, groups])

        def exported(candidate_weight, candidate_scales):
            return decode(recover_codes(candidate_weight, candidate_scales), candidate_scales)

        base_codes = recover_codes(baseline, base_scales)
        if count == 2**bits:
            codes = torch.arange(count, device=weight.device).float().expand(n, k, count)
            valid = torch.ones((n, k, count), device=weight.device, dtype=torch.bool)
        else:
            offsets = torch.tensor([0] + [((-1)**i) * ((i+1)//2) for i in range(1, count)], device=weight.device)
            codes = base_codes.unsqueeze(-1) + offsets
            valid = (codes >= 0) & (codes < 2**bits)
            codes = codes.clamp(0, 2**bits-1)
        # A scalar grid has a distance structure: distant codes must not start
        # as equally plausible alternatives to adjacent rounding choices. Use
        # the author's quadratic-distance prior, without its random initial
        # perturbation (sampling below already uses the private seeded RNG).
        logits = (-0.5 * (codes - base_codes.unsqueeze(-1)).square()).requires_grad_()
        scale_delta = torch.zeros_like(base_scales, requires_grad=config.learn_scales)
        parameters = [logits] + ([scale_delta] if config.learn_scales else [])
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        rng = torch.Generator(device=weight.device).manual_seed(config.seed + _row_start)
        before = float(loss(exported(baseline, base_scales)))
        if not torch.isfinite(torch.tensor(before)):
            raise ValueError("non-finite scalar GSQ baseline objective")
        best, best_weight, best_scales = before, baseline, scales.detach().clone()
        history = [before]
        for step in range(config.steps):
            tau = config.temperature_start * (config.temperature_end / config.temperature_start) ** (
                step / max(config.steps-1, 1))
            uniform = torch.rand(logits.shape, device=weight.device, generator=rng).clamp_(1e-6, 1-1e-6)
            noise = -(-uniform.log()).log()
            probabilities = ((logits.masked_fill(~valid, -torch.inf) + noise) / tau).softmax(-1)
            current_scales = base_scales * scale_delta.exp()
            soft_codes = (probabilities * codes).sum(-1)
            objective = loss(current_scales[:, groups] * (soft_codes - zero[:, groups]))
            if not torch.isfinite(objective):
                raise ValueError("non-finite scalar GSQ relaxed objective")
            optimizer.zero_grad()
            objective.backward()
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("non-finite scalar GSQ gradient")
            optimizer.step()
            with torch.no_grad():
                selected = logits.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
                hard_codes = codes.gather(-1, selected).squeeze(-1)
                hard_scales = (base_scales * scale_delta.exp()).to(scale_dtype).float()
                if packing in ("awq_gemv", "awq_gemv_fast"):
                    # Score the source scales we return: BF16 can round an
                    # FP16 candidate again before the packer stores it.
                    hard_scales = hard_scales.to(scales.dtype)
                candidate = decode(hard_codes, hard_scales).to(weight.dtype)
                candidate_codes = recover_codes(candidate, hard_scales)
                score = float(loss(decode(candidate_codes, hard_scales)))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError("non-finite scalar GSQ hard objective")
                history.append(score)
                if score < best:
                    best, best_weight, best_scales = score, candidate.clone(), hard_scales.clone()
        return ScalarGSQResult(best_weight.detach().to(weight.dtype), best_scales.detach().to(scales.dtype),
                               zeros.detach().clone(), g_idx.detach().clone(), before, best, history)
