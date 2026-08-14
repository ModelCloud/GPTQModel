# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deprecated learned-PGC16 research implementation.

This module is intentionally outside the production QVQ codec package. The
learned table reduced local reconstruction proxies but did not consistently
improve held-out model-output KLD, so production accepts fixed PGC16-v1 only.
Keep this implementation importable solely for reproducing historical studies.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from numbers import Real

import torch

from ...qvq_rates import normalize_qvq_rate
from ..pgc16 import (
    PGC16_LEVEL_COUNT,
    PGC16_NORMALIZATION_RMS,
    canonical_pgc16_levels,
    pgc16_codebook,
    pgc16_decode_states,
    pgc16_mix_states,
    pgc16_scale_factor,
    validate_pgc16_levels,
)


@dataclass(frozen=True)
class PGC16CompanderFitResult:
    """Frozen learned levels and the accepted alternating-fit history."""

    levels: torch.Tensor
    states: torch.Tensor
    weighted_error: torch.Tensor
    error_history: tuple[float, ...]
    iterations: int


@torch.inference_mode()
def collect_qvq_compander_population(
    weights: dict[str, torch.Tensor],
    hessians: dict[str, torch.Tensor],
    *,
    bits: float,
    device: torch.device,
    tiles_per_module: int,
    damp_percent: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Collect a deterministic model population in QVQ's exact search basis."""

    from ...qvq import rht_preprocess_hessian, rht_preprocess_weight

    if isinstance(tiles_per_module, bool) or not isinstance(tiles_per_module, int) or tiles_per_module < 1:
        raise ValueError("QVQ compander tiles per module must be a positive integer.")
    if isinstance(damp_percent, bool) or not isinstance(damp_percent, Real):
        raise TypeError("QVQ compander damping must be a real scalar.")
    damp_percent = float(damp_percent)
    if not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError("QVQ compander damping must be finite and nonnegative.")
    if set(weights) != set(hessians):
        raise ValueError("QVQ compander weights and Hessians must name the same modules.")
    if not weights:
        raise ValueError("QVQ compander population requires at least one module.")

    sampled_sequences = []
    sampled_importance = []
    available_tiles = 0
    for module_name, source_weight in weights.items():
        if not isinstance(module_name, str) or not module_name:
            raise ValueError("QVQ compander module names must be nonempty strings.")
        if source_weight.ndim != 2 or not source_weight.is_floating_point() or not torch.isfinite(source_weight).all():
            raise ValueError(f"QVQ compander weight for `{module_name}` must be a finite floating-point matrix.")
        source_hessian = hessians[module_name]
        if (
            source_hessian.ndim != 2
            or not source_hessian.is_floating_point()
            or not torch.isfinite(source_hessian).all()
        ):
            raise ValueError(f"QVQ compander Hessian for `{module_name}` must be a finite floating-point matrix.")
        weight = source_weight.to(device=device, dtype=torch.float32)
        H = source_hessian.to(device=device, dtype=torch.float32)
        out_features, in_features = weight.shape
        if in_features % 16 or out_features % 16:
            raise ValueError(f"QVQ compander module `{module_name}` dimensions must be divisible by 16.")
        if tuple(H.shape) != (in_features, in_features):
            raise ValueError(f"QVQ compander Hessian for `{module_name}` does not match its input width.")

        seed = zlib.crc32(module_name.encode("utf-8")) & 0x7FFFFFFF
        generator = torch.Generator(device="cpu").manual_seed(seed)
        SU = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
        SV = torch.randint(0, 2, (out_features,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
        SU = SU.to(device=device, dtype=torch.float32)
        SV = SV.to(device=device, dtype=torch.float32)
        transformed_weight = rht_preprocess_weight(weight, SU, SV)
        transformed_H = rht_preprocess_hessian(H, SU)
        transformed_H = (transformed_H + transformed_H.transpose(0, 1)) * 0.5
        mean_diagonal = transformed_H.diagonal().abs().mean()
        damping = torch.maximum(
            mean_diagonal * damp_percent,
            torch.tensor(torch.finfo(torch.float32).eps, device=device),
        )
        transformed_H.diagonal().add_(damping)

        scale = (
            transformed_weight.square().mean().sqrt() / PGC16_NORMALIZATION_RMS * pgc16_scale_factor(bits)
        ).clamp_min(torch.finfo(torch.float32).eps)
        normalized = transformed_weight / scale
        importance_matrix = transformed_H.diagonal().clamp_min(0).unsqueeze(1).expand_as(normalized)
        module_sequences = (
            normalized.reshape(in_features // 16, 16, out_features // 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(-1, 128, 2)
        )
        module_importance = (
            importance_matrix.reshape(in_features // 16, 16, out_features // 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(-1, 128, 2)
        )
        available_tiles += module_sequences.shape[0]
        sample_count = min(tiles_per_module, module_sequences.shape[0])
        indices = (
            torch.linspace(0, module_sequences.shape[0] - 1, steps=sample_count, device=device)
            .round()
            .to(torch.long)
        )
        sampled_sequences.append(module_sequences.index_select(0, indices))
        sampled_importance.append(module_importance.index_select(0, indices))

    sequences = torch.cat(sampled_sequences, dim=0).contiguous()
    importance = torch.cat(sampled_importance, dim=0).contiguous()
    return (
        sequences,
        importance,
        {
            "modules": len(weights),
            "available_tiles": available_tiles,
            "sampled_tiles": sequences.shape[0],
            "vectors_per_tile": sequences.shape[1],
        },
    )


def _importance_like(samples: torch.Tensor, importance: torch.Tensor | None) -> torch.Tensor:
    if importance is None:
        return torch.ones_like(samples, dtype=torch.float32)
    if not importance.is_floating_point():
        raise TypeError("PGC16 compander importance must use a floating-point dtype.")
    if not torch.isfinite(importance).all() or torch.any(importance < 0):
        raise ValueError("PGC16 compander importance must be finite and nonnegative.")
    try:
        return torch.broadcast_to(importance.to(device=samples.device, dtype=torch.float32), samples.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"PGC16 compander importance must broadcast to `{tuple(samples.shape)}`."
        ) from exc


def _validate_training_inputs(
    samples: torch.Tensor,
    states: torch.Tensor,
    levels: torch.Tensor,
) -> None:
    if samples.ndim < 1 or samples.shape[-1] != 2:
        raise ValueError("PGC16 compander samples must have a final vector dimension of two.")
    if tuple(states.shape) != tuple(samples.shape[:-1]):
        raise ValueError("PGC16 compander states must match the sample prefix shape.")
    if not samples.is_floating_point():
        raise TypeError("PGC16 compander samples must use a floating-point dtype.")
    if states.dtype == torch.bool or states.is_floating_point() or states.is_complex():
        raise TypeError("PGC16 compander states must use an integer dtype.")
    if not torch.isfinite(samples).all():
        raise ValueError("PGC16 compander samples must contain only finite values.")
    if states.device != samples.device or levels.device != samples.device:
        raise ValueError("PGC16 compander samples, states, and levels must share one device.")
    validate_pgc16_levels(levels)


def pgc16_weighted_error(
    samples: torch.Tensor,
    states: torch.Tensor,
    levels: torch.Tensor,
    *,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return diagonal-Hessian-weighted reconstruction SSE for fixed states."""

    _validate_training_inputs(samples, states, levels)
    weights = _importance_like(samples, importance)
    reconstructed = pgc16_decode_states(states, levels=levels).to(torch.float32)
    return ((samples.to(torch.float32) - reconstructed).square() * weights).sum()


def _weighted_isotonic(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Project a tiny 1D table onto the nondecreasing cone with weighted PAVA."""

    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values.tolist(), weights.tolist())):
        blocks.append([index, index + 1, float(weight), float(weight) * float(value)])
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous[3]) / float(previous[2])
            current_mean = float(current[3]) / float(current[2])
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                [
                    int(previous[0]),
                    int(current[1]),
                    float(previous[2]) + float(current[2]),
                    float(previous[3]) + float(current[3]),
                ]
            ]
    projected = torch.empty_like(values)
    for start, end, weight, weighted_sum in blocks:
        projected[int(start) : int(end)] = float(weighted_sum) / float(weight)
    return projected


def _strict_fp16(values: torch.Tensor) -> torch.Tensor:
    """Round ordered levels to FP16 while retaining strict uniqueness."""

    frozen = values.to(device="cpu", dtype=torch.float16).clone()
    positive_infinity = torch.tensor(torch.inf, dtype=torch.float16)
    for index in range(1, frozen.numel()):
        if frozen[index] <= frozen[index - 1]:
            frozen[index] = torch.nextafter(frozen[index - 1], positive_infinity)
    if not torch.isfinite(frozen).all():
        raise ValueError("PGC16 learned levels overflowed while freezing to FP16.")
    return frozen


def lloyd_update_pgc16_levels(
    samples: torch.Tensor,
    states: torch.Tensor,
    *,
    importance: torch.Tensor | None = None,
    previous_levels: torch.Tensor | None = None,
    symmetric: bool = True,
) -> torch.Tensor:
    """Optimize the 256 scalar levels for fixed PGC16 state assignments."""

    if not isinstance(symmetric, bool):
        raise TypeError("PGC16 compander symmetry control must be boolean.")
    if previous_levels is None:
        previous_levels = canonical_pgc16_levels().to(device=samples.device)
    _validate_training_inputs(samples, states, previous_levels)
    weights = _importance_like(samples, importance)
    if not torch.any(weights > 0):
        return previous_levels.detach().to(device="cpu", dtype=torch.float16).clone()

    mixed = pgc16_mix_states(states)
    indices = torch.stack((mixed >> 8, mixed & 0xFF), dim=-1).reshape(-1)
    values = samples.to(torch.float32).reshape(-1)
    flat_weights = weights.reshape(-1)
    numerators = torch.zeros(PGC16_LEVEL_COUNT, dtype=torch.float32, device=samples.device)
    denominators = torch.zeros_like(numerators)
    numerators.scatter_add_(0, indices, values * flat_weights)
    denominators.scatter_add_(0, indices, flat_weights)

    numerators = numerators.cpu().to(torch.float64)
    denominators = denominators.cpu().to(torch.float64)
    previous = previous_levels.detach().cpu().to(torch.float64)
    prior_weight = torch.finfo(torch.float64).eps

    if symmetric:
        high = torch.arange(PGC16_LEVEL_COUNT // 2, PGC16_LEVEL_COUNT)
        low = PGC16_LEVEL_COUNT - 1 - high
        pair_weights = denominators[high] + denominators[low]
        raw_magnitudes = (numerators[high] - numerators[low]) / pair_weights.clamp_min(prior_weight)
        unobserved = pair_weights == 0
        raw_magnitudes[unobserved] = previous[high].abs()[unobserved]
        fit_weights = pair_weights.clamp_min(prior_weight)
        magnitudes = _weighted_isotonic(raw_magnitudes.clamp_min(0), fit_weights)
        positive = _strict_fp16(magnitudes.clamp_min(torch.finfo(torch.float16).smallest_normal))
        frozen = torch.cat((-positive.flip(0), positive))
    else:
        raw_levels = numerators / denominators.clamp_min(prior_weight)
        unobserved = denominators == 0
        raw_levels[unobserved] = previous[unobserved]
        frozen = _strict_fp16(_weighted_isotonic(raw_levels, denominators.clamp_min(prior_weight)))

    validate_pgc16_levels(frozen)
    return frozen


@torch.inference_mode()
def fit_pgc16_compander(
    sequences: torch.Tensor,
    *,
    bits: float,
    importance: torch.Tensor | None = None,
    initial_levels: torch.Tensor | None = None,
    iterations: int = 8,
    symmetric: bool = True,
) -> PGC16CompanderFitResult:
    """Alternate QVQ trellis assignment and weighted scalar Lloyd updates."""

    from ...qvq import tail_biting_viterbi_quantize

    if sequences.ndim != 3 or sequences.shape[-1] != 2 or sequences.shape[1] < 2:
        raise ValueError("PGC16 fitting sequences must have shape `[batch, steps>=2, 2]`.")
    if not sequences.is_floating_point() or not torch.isfinite(sequences).all():
        raise ValueError("PGC16 fitting sequences must be finite floating-point values.")
    bits = normalize_qvq_rate(bits)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("PGC16 fitting iterations must be a positive integer.")
    if not isinstance(symmetric, bool):
        raise TypeError("PGC16 fitting symmetry control must be boolean.")

    weights = _importance_like(sequences, importance)
    levels = (
        canonical_pgc16_levels().to(device=sequences.device)
        if initial_levels is None
        else initial_levels.detach().to(device=sequences.device, dtype=torch.float16).contiguous()
    )
    validate_pgc16_levels(levels)

    def encode(candidate_levels: torch.Tensor):
        codebook = pgc16_codebook(device=sequences.device, dtype=torch.float32, levels=candidate_levels)
        return tail_biting_viterbi_quantize(sequences, codebook, bits=bits)

    best = encode(levels)
    best_error = pgc16_weighted_error(sequences, best.states, levels, importance=weights)
    if not torch.isfinite(best_error):
        raise ValueError("PGC16 initial compander objective must be finite.")
    history = [float(best_error.item())]
    accepted = 0
    for _ in range(iterations):
        candidate_cpu = lloyd_update_pgc16_levels(
            sequences,
            best.states,
            importance=weights,
            previous_levels=levels,
            symmetric=symmetric,
        )
        candidate_levels = candidate_cpu.to(device=sequences.device)
        candidate = encode(candidate_levels)
        candidate_error = pgc16_weighted_error(
            sequences,
            candidate.states,
            candidate_levels,
            importance=weights,
        )
        # Both objectives use the exact frozen FP16 tables, so accepting even a
        # tolerance-sized increase would be a real serialized-format regression.
        if not torch.isfinite(candidate_error) or candidate_error > best_error:
            break
        levels = candidate_levels
        best = candidate
        best_error = candidate_error
        history.append(float(best_error.item()))
        accepted += 1

    return PGC16CompanderFitResult(
        levels=levels.detach().cpu().to(torch.float16),
        states=best.states,
        weighted_error=best_error,
        error_history=tuple(history),
        iterations=accepted,
    )


__all__ = [
    "PGC16CompanderFitResult",
    "collect_qvq_compander_population",
    "fit_pgc16_compander",
    "lloyd_update_pgc16_levels",
    "pgc16_weighted_error",
]
