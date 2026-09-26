# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in Gumbel-Softmax refinement on an existing GPTQ scalar grid.

The checkpoint still uses GPTQ scales, zero-points, group indices and packers.
This is a local reconstruction objective, not the paper's full block-training
schedule. The baseline is retained unless a hard, export-aware candidate wins.
"""

from dataclasses import dataclass

import pcre
import torch

from ..utils.gemv import awq_gemv_codes
from .config import normalize_gsq_config


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
        stored = scales.to(scale_dtype)[:, g_idx]
        offset = (zeros[:, g_idx] * stored).to(weight.dtype)
        values = (weight + offset) / stored
    elif packing in ("awq_gemv", "awq_gemv_fast"):
        if bits != 4 or scale_dtype != torch.float16:
            raise ValueError("AWQ GEMV GSQ requires 4 bits and FP16 scales")
        return awq_gemv_codes(weight, scales, zeros, g_idx)
    else:
        raise ValueError("Unknown affine GSQ packing format")
    rounded = values.round()
    # GPTQ's pack_original clamps. AWQ GEMM/GEMV packers do not, so hiding an
    # overflowing code here could silently spill bits into adjacent nibbles.
    return rounded.clamp(0, 2**bits - 1) if packing == "gptq" else rounded


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


def _refine_rows(weight, scales, zeros, target, group, factor, bits, choices, config, scale_dtype, row_start):
    rows, columns = weight.shape
    teacher = target.detach().float().clone()
    baseline = weight.detach().clone()
    base_scales = scales.detach().float().clone()
    zero = zeros.detach().float().clone()
    teacher_energy = (teacher @ factor).square().sum()
    denominator = torch.where(
        teacher_energy > torch.finfo(torch.float32).eps,
        teacher_energy,
        torch.ones_like(teacher_energy),
    )

    def loss(matrix):
        return ((matrix.float() - teacher) @ factor).square().sum() / denominator

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
    if choices == 2**bits:
        codes = torch.arange(choices, device=weight.device).float().expand(rows, columns, choices)
        valid = torch.ones((rows, columns, choices), device=weight.device, dtype=torch.bool)
    else:
        offsets = torch.tensor(
            [0] + [((-1) ** i) * ((i + 1) // 2) for i in range(1, choices)],
            device=weight.device,
        )
        codes = base_codes.unsqueeze(-1) + offsets
        valid = (codes >= 0) & (codes < 2**bits)
        codes = codes.clamp(0, 2**bits - 1)

    logits = (-0.5 * (codes - base_codes.unsqueeze(-1)).square()).requires_grad_()
    scale_delta = torch.zeros_like(base_scales, requires_grad=config.learn_scales)
    parameters = [logits] + ([scale_delta] if config.learn_scales else [])
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    rng = torch.Generator(device=weight.device).manual_seed(config.seed + row_start)
    before = float(loss(exported(baseline, scales)))
    if not torch.isfinite(torch.tensor(before)):
        raise ValueError("non-finite GSQ baseline objective")
    best, best_weight, best_scales = before, baseline, scales.detach().clone()
    history = [before]
    for step in range(config.steps):
        temperature = config.temperature_start * (
            config.temperature_end / config.temperature_start
        ) ** (step / max(config.steps - 1, 1))
        uniform = torch.rand(logits.shape, device=weight.device, generator=rng).clamp_(1e-6, 1 - 1e-6)
        gumbel = -(-uniform.log()).log()
        probabilities = ((logits.masked_fill(~valid, -torch.inf) + gumbel) / temperature).softmax(-1)
        current_scales = base_scales * scale_delta.exp()
        soft_codes = (probabilities * codes).sum(-1)
        objective = loss(current_scales[:, group] * (soft_codes - zero[:, group]))
        if not bool(torch.isfinite(objective)):
            raise ValueError("non-finite GSQ relaxed objective")
        optimizer.zero_grad()
        objective.backward()
        if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
            raise ValueError("non-finite GSQ gradient")
        optimizer.step()
        with torch.no_grad():
            chosen = logits.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
            hard_codes = codes.gather(-1, chosen).squeeze(-1)
            hard_scales = (base_scales * scale_delta.exp()).to(scale_dtype).float()
            candidate = decode(hard_codes, hard_scales).to(weight.dtype)
            score = float(loss(exported(candidate, hard_scales)))
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("non-finite GSQ hard objective")
            history.append(score)
            if score < best:
                best, best_weight, best_scales = score, candidate.clone(), hard_scales.clone()
    return ScalarGSQResult(
        best_weight.detach().to(weight.dtype),
        best_scales.detach().to(scales.dtype),
        zeros.detach().clone(),
        group.detach().clone(),
        before,
        best,
        history,
    ), float(teacher_energy)
def refine_affine_scalar(
    weight, scales, zeros, g_idx, *, target, bits, config=None,
    hessian=None, inputs=None, packing="gptq", scale_dtype=torch.float16,
    cross_moment=None, cross_alpha=1.0,
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
    if count * weight.numel() * 4 > config.max_candidate_bytes:
        raise ValueError("scalar GSQ decoded candidates exceed max_candidate_bytes")

    with torch.inference_mode(False), torch.enable_grad():
        teacher = target.detach().float().clone()
        groups = g_idx.detach().long().clone()
        base_scales = scales.detach().float().clone()
        zero = zeros.detach().float().clone()
        baseline = weight.detach().clone()
        factor = None
        diagonal = None
        if hessian is not None:
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

        gemv = packing in ("awq_gemv", "awq_gemv_fast")

        def recover_codes(candidate_weight, candidate_scales):
            source_scales = candidate_scales.to(scales.dtype) if gemv else candidate_scales
            source_zeros = zero.to(zeros.dtype) if gemv else zero
            values = affine_codes(candidate_weight, source_scales, source_zeros, groups, bits,
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
            source_scales = candidate_scales.to(scales.dtype) if gemv else candidate_scales
            recovered = recover_codes(candidate_weight, source_scales)
            if (recovered < 0).any() or (recovered >= 2**bits).any():
                raise ValueError("AWQ GSQ producer arithmetic produced an out-of-range packed code")
            return decode(recovered, source_scales)

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
        rng = torch.Generator(device=weight.device).manual_seed(config.seed)
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
                if gemv:
                    hard_scales = hard_scales.to(scales.dtype)
                candidate = decode(hard_codes, hard_scales).to(weight.dtype)
                candidate_codes = recover_codes(candidate, hard_scales)
                if (candidate_codes < 0).any() or (candidate_codes >= 2**bits).any():
                    history.append(best)
                    continue
                score = float(loss(decode(candidate_codes, hard_scales)))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError("non-finite scalar GSQ hard objective")
                history.append(score)
                if score < best:
                    best, best_weight, best_scales = score, candidate.clone(), hard_scales.clone()
        return ScalarGSQResult(best_weight.detach().to(weight.dtype), best_scales.detach().to(scales.dtype),
                               zeros.detach().clone(), g_idx.detach().clone(), before, best, history)
