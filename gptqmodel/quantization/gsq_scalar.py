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


def gsq_enabled_for(config, module_name: str) -> bool:
    config = normalize_gsq_config(config)
    return config is not None and config.enabled and (
        config.modules is None or any(pcre.search(pattern, module_name) for pattern in config.modules)
    )


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
