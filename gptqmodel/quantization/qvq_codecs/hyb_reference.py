# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Non-loadable HYB regression oracle and benchmark baseline."""

from __future__ import annotations

from functools import lru_cache

import torch


def fit_hyb_lut(
    *,
    lut_bits: int = 9,
    sample_count: int = 32_768,
    iterations: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fit a deterministic HYB LUT to an empirical two-dimensional Gaussian."""

    if isinstance(lut_bits, bool) or not isinstance(lut_bits, int) or not 1 <= lut_bits <= 15:
        raise ValueError("QVQ HYB `lut_bits` must be an integer in `[1, 15]`.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("QVQ HYB `sample_count` must be an integer.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("QVQ HYB `iterations` must be a positive integer.")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("QVQ HYB LUT dtype must be floating point.")

    center_count = 1 << lut_bits
    if sample_count < center_count:
        raise ValueError("QVQ HYB `sample_count` must be at least the LUT size.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn((sample_count, 2), generator=generator, dtype=torch.float32)
    samples[:, 1].abs_()

    first = int(torch.randint(sample_count, (), generator=generator).item())
    centers = torch.empty((center_count, 2), dtype=torch.float32)
    centers[0] = samples[first]
    closest = (samples - centers[0]).square().sum(dim=1)
    for index in range(1, center_count):
        total = closest.sum()
        if not torch.isfinite(total) or total <= 0:
            centers[index:] = samples[: center_count - index]
            break
        selected = int(torch.multinomial(closest, 1, generator=generator).item())
        centers[index] = samples[selected]
        torch.minimum(closest, (samples - centers[index]).square().sum(dim=1), out=closest)

    chunk_size = max(1, min(sample_count, 4096))
    for _ in range(iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(center_count, dtype=torch.int64)
        for chunk in samples.split(chunk_size):
            assignment = torch.cdist(chunk, centers).square_().argmin(dim=1)
            sums.index_add_(0, assignment, chunk)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.int64))
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]

    return centers.to(dtype=dtype).contiguous()


@lru_cache(maxsize=1)
def canonical_hyb_lut() -> torch.Tensor:
    """Return the process-local deterministic HYB reference LUT."""

    return fit_hyb_lut()


def hyb_decode_states(
    states: torch.Tensor,
    lut: torch.Tensor,
    *,
    trellis_window: int = 16,
    lut_bits: int = 9,
) -> torch.Tensor:
    """Decode states with the QTIP paper's HYB rule for comparison only."""

    if trellis_window < 1 or trellis_window > 32:
        raise ValueError("QVQ `trellis_window` must be in `[1, 32]`.")
    if lut_bits < 1 or lut_bits > 15:
        raise ValueError("QVQ HYB `lut_bits` must be in `[1, 15]`.")
    if lut.ndim != 2 or tuple(lut.shape) != (1 << lut_bits, 2):
        raise ValueError(f"QVQ HYB LUT must have shape `{(1 << lut_bits, 2)}`, got `{tuple(lut.shape)}`.")
    if not lut.is_floating_point():
        raise TypeError("QVQ HYB LUT must use a floating-point dtype.")
    if not torch.isfinite(lut).all():
        raise ValueError("QVQ HYB LUT must contain only finite values.")

    state_mask = (1 << trellis_window) - 1
    states_i64 = states.to(device=lut.device, dtype=torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 > state_mask)):
        raise ValueError(f"QVQ trellis states must be in `[0, {state_mask}]`.")

    hashed = (states_i64 * states_i64 + states_i64) & 0xFFFFFFFF
    lut_indices = (hashed >> (15 - lut_bits)) & ((1 << lut_bits) - 1)
    decoded = lut[lut_indices].clone()
    second_sign = torch.where(
        (hashed & (1 << 15)) == 0,
        torch.ones((), dtype=decoded.dtype, device=decoded.device),
        -torch.ones((), dtype=decoded.dtype, device=decoded.device),
    )
    decoded[..., 1] *= second_sign
    return decoded


def hyb_codebook(
    lut: torch.Tensor,
    *,
    trellis_window: int = 16,
    lut_bits: int = 9,
) -> torch.Tensor:
    """Materialize the non-loadable HYB reference value for every state."""

    states = torch.arange(1 << trellis_window, dtype=torch.int64, device=lut.device)
    return hyb_decode_states(states, lut, trellis_window=trellis_window, lut_bits=lut_bits)


__all__ = ["canonical_hyb_lut", "fit_hyb_lut", "hyb_codebook", "hyb_decode_states"]
