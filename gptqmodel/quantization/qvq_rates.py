# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact rate helpers for QVQ's fixed two-value bitshift trellis."""

from __future__ import annotations

import functools
import math
from fractions import Fraction

QVQ_VECTOR_SIZE = 2
QVQ_MIN_RATE = Fraction(1, 1)
QVQ_MAX_RATE = Fraction(8, 1)
QVQ_TRANSITION_BITS = tuple(range(2, 17))
QVQ_BITS = tuple(value // 2 if value % 2 == 0 else value / 2 for value in QVQ_TRANSITION_BITS)


def _qvq_rate_fraction(rate: float | str | Fraction) -> Fraction:
    if isinstance(rate, bool):
        raise TypeError("QVQ rate must be an integer, float, string, or Fraction.")
    if isinstance(rate, Fraction):
        normalized = rate
    elif isinstance(rate, int):
        normalized = Fraction(rate, 1)
    elif isinstance(rate, float):
        if not math.isfinite(rate):
            raise ValueError("QVQ rate must be finite.")
        normalized = Fraction(str(rate))
    elif isinstance(rate, str):
        raw = rate.strip().lower().removeprefix("w")
        try:
            normalized = Fraction(raw)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Unsupported QVQ rate specification `{rate}`.") from exc
    else:
        raise TypeError(f"Unsupported QVQ rate type `{type(rate).__name__}`.")

    if normalized < QVQ_MIN_RATE or normalized > QVQ_MAX_RATE or (normalized * 2).denominator != 1:
        raise ValueError("QVQ rate must be an integer or half-integer from 1 through 8.")
    return normalized


def normalize_qvq_rate(rate: float | str | Fraction) -> int | float:
    """Return the canonical JSON-safe QVQ rate."""

    normalized = _qvq_rate_fraction(rate)
    if normalized.denominator == 1:
        return normalized.numerator
    return normalized.numerator / normalized.denominator


@functools.lru_cache(maxsize=128)
def qvq_transition_bits(rate: float | str | Fraction, *, vector_size: int = QVQ_VECTOR_SIZE) -> int:
    """Return the integer bits appended by one trellis transition."""

    if isinstance(vector_size, bool) or not isinstance(vector_size, int) or vector_size < 1:
        raise ValueError("QVQ vector size must be a positive integer.")
    transition_width = _qvq_rate_fraction(rate) * vector_size
    if transition_width.denominator != 1:
        raise ValueError("QVQ requires `rate * vector_size` to be an integer transition width.")
    return transition_width.numerator


def qvq_rate_from_transition_bits(
    transition_bits: int,
    *,
    vector_size: int = QVQ_VECTOR_SIZE,
) -> int | float:
    """Return the exact public rate represented by an integer transition width."""

    if isinstance(transition_bits, bool) or not isinstance(transition_bits, int) or transition_bits < 1:
        raise ValueError("QVQ transition width must be a positive integer.")
    if isinstance(vector_size, bool) or not isinstance(vector_size, int) or vector_size < 1:
        raise ValueError("QVQ vector size must be a positive integer.")
    return normalize_qvq_rate(Fraction(transition_bits, vector_size))


def qvq_words_per_tile(
    rate: float | str | Fraction,
    *,
    weight_count: int = 256,
    vector_size: int = QVQ_VECTOR_SIZE,
    word_bits: int = 32,
) -> int:
    """Return the exact packed-word count for one fixed-rate tail-biting tile."""

    if isinstance(weight_count, bool) or not isinstance(weight_count, int) or weight_count < 1:
        raise ValueError("QVQ tile weight count must be a positive integer.")
    if isinstance(word_bits, bool) or not isinstance(word_bits, int) or word_bits < 1:
        raise ValueError("QVQ packed word width must be a positive integer.")
    transition_bits = qvq_transition_bits(rate, vector_size=vector_size)
    if weight_count % vector_size:
        raise ValueError("QVQ tile weight count must be divisible by the vector size.")
    payload_bits = weight_count // vector_size * transition_bits
    if payload_bits % word_bits:
        raise ValueError("QVQ tile payload must contain a whole number of packed words.")
    return payload_bits // word_bits


__all__ = [
    "QVQ_BITS",
    "QVQ_MAX_RATE",
    "QVQ_MIN_RATE",
    "QVQ_TRANSITION_BITS",
    "QVQ_VECTOR_SIZE",
    "normalize_qvq_rate",
    "qvq_rate_from_transition_bits",
    "qvq_transition_bits",
    "qvq_words_per_tile",
]
