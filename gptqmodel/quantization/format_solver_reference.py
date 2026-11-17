# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Small, single-threaded CPU proofs, deliberately outside the model lifecycle.

GPTQ and AWQ are reference generalizations of their reconstruction equations;
GSQ uses the production Gumbel relaxation primitive; RCO uses its production
discrete allocator. None of these functions advertise model/kernel support.
"""

import math
from dataclasses import dataclass

import torch

from .format_reference import (
    FormatTensor,
    _matrix,
    decode_elements,
    encode_elements,
    fit_grid,
    positive_grid,
    quantize_rtn,
)


@dataclass(frozen=True)
class SolverResult:
    weight: FormatTensor
    input_scale: torch.Tensor | None = None
    history: tuple[float, ...] = ()

    def forward(self, inputs):
        inputs = _matrix(inputs)
        if self.input_scale is not None:
            inputs = inputs / self.input_scale.double()
        return inputs @ self.weight.dequantize().T

    @property
    def storage_bits(self):
        extra = 0 if self.input_scale is None else self.input_scale.numel() * self.input_scale.element_size() * 8
        return self.weight.storage_bits + extra


def reconstruction_error(result, weight, inputs):
    target = inputs.double() @ weight.double().T
    energy = target.square().sum()
    error = (result.forward(inputs) - target).square().sum()
    return float(error / energy) if energy > 0 else float(error)


def _problem(weight, inputs):
    weight, inputs = _matrix(weight), _matrix(inputs)
    if inputs.shape[1] != weight.shape[1]:
        raise ValueError("Calibration inputs and weights must share the input dimension")
    if not torch.isfinite(inputs @ weight.T).all():
        raise ValueError("Reference product overflows FP64")
    return weight, inputs


def solve_gptq(weight, inputs, name, *, damping=.01, act_order=False):
    """Sequential GPTQ/OBS update with frozen target-format block scales.

    The Cholesky inverse-Hessian update does not assume a uniform integer
    codebook. Only the scalar projection changes. Physical block membership
    is preserved even when columns are processed in activation order.
    """
    weight, inputs = _problem(weight, inputs)
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("Damping must be finite and positive")
    grid = fit_grid(weight, name)
    hessian = inputs.T @ inputs / inputs.shape[0]
    diagonal = hessian.diagonal()
    if not torch.isfinite(hessian).all():
        raise ValueError("Reference Hessian overflows FP64")
    if not diagonal.any():
        return SolverResult(grid.pack(grid.encode(weight)))
    order = diagonal.argsort(descending=True, stable=True) if act_order else torch.arange(weight.shape[1])
    ridge = damping * max(float(diagonal.mean()), torch.finfo(torch.float64).tiny)
    hessian = hessian + ridge * torch.eye(weight.shape[1], dtype=torch.float64)
    hessian = hessian[order][:, order]
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    upper = torch.linalg.cholesky(inverse, upper=True)
    working = weight[:, order].clone()
    scales, zeros = grid.expanded()[:, order], grid.zero_grid()[:, order]
    codes = torch.empty_like(weight, dtype=torch.uint8)
    for column in range(weight.shape[1]):
        normalized = working[:, column] / scales[:, column]
        if grid.zeros is not None:
            normalized = normalized + zeros[:, column]
        code = encode_elements(normalized, grid.spec.element)
        quantized = (decode_elements(code, grid.spec.element) - zeros[:, column]) * scales[:, column]
        codes[:, order[column]] = code
        error = (working[:, column] - quantized) / upper[column, column]
        working[:, column:] -= error[:, None] * upper[column, column:]
    return SolverResult(grid.pack(codes))


def solve_awq(weight, inputs, name, *, grid_steps=21):
    """AWQ activation-aware diagonal scaling scored on the *target* codebook.

    The winning transform is retained explicitly. Dividing it out of the
    packed weights would destroy target-format membership. Production must
    fold the inverse into a valid upstream producer or retain the transform.
    """
    weight, inputs = _problem(weight, inputs)
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int) or grid_steps < 2:
        raise ValueError("AWQ requires at least two scale candidates")
    activation = inputs.abs().mean(0).clamp(min=1e-12)
    best = SolverResult(quantize_rtn(weight, name))
    best_loss = reconstruction_error(best, weight, inputs)
    scores = [best_loss]
    for alpha in torch.linspace(0, 1, grid_steps, dtype=torch.float64):
        scale = activation.pow(alpha)
        scale = (scale / (scale.min() * scale.max()).sqrt()).float()
        candidate = SolverResult(quantize_rtn(weight * scale.double(), name), scale)
        loss = reconstruction_error(candidate, weight, inputs)
        scores.append(loss)
        if loss < best_loss:
            best, best_loss = candidate, loss
    return SolverResult(best.weight, best.input_scale, tuple(scores))


def _candidate_bank(packed):
    element = packed.grid.spec.element
    positive = positive_grid(element)
    if element == "int4":
        values = positive
    else:
        values = torch.cat((-positive[1:].flip(0), positive))
    raw = decode_elements(packed.codes(), element)
    center = torch.searchsorted(values, raw.contiguous())
    payloads = [packed.codes()]
    for offset in (-1, 1):
        payloads.append(encode_elements(values[(center + offset).clamp(0, len(values) - 1)], element))
    codes = torch.stack(payloads)
    return codes, torch.stack([packed.grid.decode(candidate) for candidate in payloads])


def solve_gsq(weight, inputs, name, *, steps=100, seed=0, learning_rate=.05):
    """GSQ categorical reconstruction over exact legal payload neighbours.

    Uses the repository's author-form Gumbel relaxation and sampling schedule,
    with frozen scales and a hard-payload acceptance guard. This is a local
    linear proof, not the complete staged model-training implementation.
    """
    from .gsq_training import relaxed_scalar_weights, sampling_schedule

    weight, inputs = _problem(weight, inputs)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("GSQ steps must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("GSQ learning rate must be finite and positive")
    packed = quantize_rtn(weight, name)
    best = SolverResult(packed)
    best_loss = reconstruction_error(best, weight, inputs)
    history = [best_loss]
    if best_loss == 0:
        return SolverResult(packed, history=tuple(history))
    codes, candidates = _candidate_bank(packed)
    logits = torch.zeros_like(candidates)
    logits[0] = .1
    logits.requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    group_index = torch.zeros(weight.shape[1], dtype=torch.long)
    scales = torch.ones(weight.shape[0], 1, dtype=torch.float64)
    teacher = inputs @ weight.T
    denominator = max(float(teacher.square().sum()), torch.finfo(torch.float64).tiny)
    for step in range(steps):
        temperature, multiplier = sampling_schedule(step, steps)
        uniform = torch.rand(logits.shape, dtype=torch.float64, generator=generator)
        relaxed = relaxed_scalar_weights(logits, scales, candidates, group_index, uniform=uniform,
                                         temperature=temperature, multiplier=multiplier)
        loss = ((inputs @ relaxed.T - teacher).square().sum() / denominator)
        if not torch.isfinite(loss):
            raise ValueError("GSQ relaxed loss is nonfinite")
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(logits.grad).all():
            raise ValueError("GSQ gradient is nonfinite")
        optimizer.step()
        with torch.no_grad():
            chosen = codes.gather(0, logits.argmax(0, keepdim=True)).squeeze(0)
            hard = SolverResult(packed.grid.pack(chosen))
            score = reconstruction_error(hard, weight, inputs)
            if score < best_loss:
                best, best_loss = hard, score
            history.append(best_loss)
    return SolverResult(best.weight, history=tuple(history))


def allocate_rco(candidates, errors, *, budget_bits):
    """Apply RCO's discrete decoder to solved format choices, with exact costs.

    The small proof uses one cost unit per bit (not the production coarse
    approximation). All payloads, stored scales, and AWQ transforms count.
    Scores are additive negative reconstruction errors, not model KL logits.
    """
    from .rco import budget_constrained_argmax

    if isinstance(budget_bits, bool) or not isinstance(budget_bits, int) or budget_bits <= 0:
        raise ValueError("RCO requires a positive integer bit budget")
    if not candidates or not candidates[0] or any(len(row) != len(candidates[0]) for row in candidates):
        raise ValueError("RCO candidates must be a nonempty rectangular table")
    errors = _matrix(errors)
    if errors.shape != (len(candidates), len(candidates[0])) or (errors < 0).any():
        raise ValueError("RCO errors must match the candidate table and be nonnegative")
    costs = torch.tensor([[candidate.storage_bits for candidate in row] for row in candidates], dtype=torch.float64)
    # weights=1, resolution=number of groups makes each DP unit exactly one bit.
    count = len(candidates)
    assignment = budget_constrained_argmax(-errors, torch.ones(count, dtype=torch.float64), costs,
                                          budget_bits / count, resolution=count)
    used = sum(row[int(index)].storage_bits for row, index in zip(candidates, assignment))
    if used > budget_bits:
        raise RuntimeError("RCO exceeded the exact storage budget")
    return assignment
