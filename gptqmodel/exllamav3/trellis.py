# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Trellis geometry checks shared by the EXL3 loader and native op."""

from __future__ import annotations


HALF_RATES = (1.5, 2.5, 3.5)
HALF_RATE_MASK = 0xAAAA
Q4NX_RATES = (2.5, 3, 3.5)
EXL3_CODEBOOK_MARKERS = ("mcg", "mul1", "q4nx")


def validate_trellis_k(
    k: int | float, mcg: bool = False, mul1: bool = False, *, q4nx: bool = False, max_integer_rate: int = 8
) -> None:
    """Validate a native rate and its packed codebook marker."""
    if mcg and mul1:
        raise ValueError("EXL3 trellis cannot set both mcg and mul1")
    if q4nx and (mcg or mul1):
        raise ValueError("EXL3 trellis cannot combine q4nx with another codebook marker")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not 1 <= k <= max_integer_rate:
        raise ValueError(f"EXL3 trellis K must be in [1, {max_integer_rate}], got {k}")
    if q4nx and k not in Q4NX_RATES:
        raise ValueError(f"EXL3 q4nx supports 2.5, 3.0, or 3.5 bpw, got {k}")
    if not q4nx and k != int(k) and (k not in HALF_RATES or not mul1):
        raise ValueError(f"EXL3 trellis K={k} requires a supported half-bit rate (1.5, 2.5, 3.5) and mul1")


def trellis_rate(
    words: int, mcg: bool = False, mul1: bool = False, *, q4nx: bool = False, max_integer_rate: int = 8
) -> int | float:
    """Derive the exact rate from the number of int16 words per 256 weights."""
    depth, remainder = divmod(words, 16)
    rate = words / 16 if remainder else depth
    validate_trellis_k(rate, mcg, mul1, q4nx=q4nx, max_integer_rate=max_integer_rate)
    return rate


__all__ = ["EXL3_CODEBOOK_MARKERS", "HALF_RATES", "HALF_RATE_MASK", "Q4NX_RATES", "trellis_rate", "validate_trellis_k"]
