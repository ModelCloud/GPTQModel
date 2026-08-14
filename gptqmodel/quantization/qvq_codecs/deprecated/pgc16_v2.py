# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deprecated learned PGC16-v2 table serialization for research replay."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from ..pgc16 import PGC16_LEVEL_COUNT, canonical_pgc16_levels, validate_pgc16_levels


QVQ_LEARNED_CODEBOOK_VERSION = "pgc16-v2"


def blend_pgc16_compander_levels(learned_levels: torch.Tensor, blend: float) -> torch.Tensor:
    """Reproduce the historical v2-to-v1 table regularization experiment."""

    validate_pgc16_levels(learned_levels)
    if isinstance(blend, bool) or not isinstance(blend, (int, float)):
        raise TypeError("QVQ compander blend must be a real scalar")
    blend = float(blend)
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("QVQ compander blend must be finite and in [0, 1]")
    if not torch.equal(learned_levels, -learned_levels.flip(0)):
        raise ValueError("QVQ compander blending requires an exactly symmetric learned table")

    canonical = canonical_pgc16_levels().to(device=learned_levels.device)
    if blend == 0.0:
        return canonical.clone()
    if blend == 1.0:
        return learned_levels.detach().to(dtype=torch.float16).clone()
    blended = torch.lerp(canonical.to(torch.float32), learned_levels.to(torch.float32), blend).to(torch.float16)
    validate_pgc16_levels(blended)
    if not torch.equal(blended, -blended.flip(0)):
        raise ValueError("QVQ compander blending must preserve exact FP16 symmetry")
    return blended


def freeze_pgc16_level_bits(levels: torch.Tensor) -> tuple[int, ...]:
    """Freeze one historical learned compander as unsigned FP16 bits."""

    validate_pgc16_levels(levels)
    frozen = levels.detach().to(device="cpu", dtype=torch.float16).contiguous()
    validate_pgc16_levels(frozen)
    return tuple(int(value) & 0xFFFF for value in frozen.view(torch.int16).tolist())


@lru_cache(maxsize=128)
def learned_pgc16_levels(level_bits: tuple[int, ...]) -> torch.Tensor:
    """Decode historical learned-compander bits outside production QVQ."""

    if len(level_bits) != PGC16_LEVEL_COUNT:
        raise ValueError(f"PGC16-v2 compander bits must contain {PGC16_LEVEL_COUNT} entries.")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFF for value in level_bits):
        raise ValueError("PGC16-v2 compander bits must be unsigned 16-bit integers.")
    signed = [value if value < 0x8000 else value - 0x10000 for value in level_bits]
    levels = torch.tensor(signed, dtype=torch.int16).view(torch.float16)
    validate_pgc16_levels(levels)
    return levels


__all__ = [
    "QVQ_LEARNED_CODEBOOK_VERSION",
    "blend_pgc16_compander_levels",
    "freeze_pgc16_level_bits",
    "learned_pgc16_levels",
]
