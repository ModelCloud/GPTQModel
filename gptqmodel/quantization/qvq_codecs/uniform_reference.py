# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Zero-table uniform-product PGC16 diagnostic decoder."""

from __future__ import annotations

import math

import torch

from .pgc16 import PGC16_LEVEL_COUNT, pgc16_mix_states


def uniform_decode_states(states: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode with computed unit-RMS uniform levels for a no-LUT baseline."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("QVQ uniform reference dtype must be floating point.")
    mixed = pgc16_mix_states(states)
    indices = torch.stack((mixed >> 8, mixed & 0xFF), dim=-1).to(torch.float64)
    levels = ((indices + 0.5) * (2.0 / PGC16_LEVEL_COUNT) - 1.0) * math.sqrt(3.0)
    return levels.to(dtype=dtype)


def uniform_codebook(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize the zero-table uniform reference codebook."""

    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(1 << 16, dtype=torch.int64, device=target_device)
    return uniform_decode_states(states, dtype=dtype)


__all__ = ["uniform_codebook", "uniform_decode_states"]
