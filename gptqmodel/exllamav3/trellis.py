# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Trellis geometry checks shared by the EXL3 loader and native op."""

from __future__ import annotations


def validate_trellis_k(k: int, mcg: bool = False, mul1: bool = False) -> None:
    """Reject a trellis K that the native kernel table cannot represent.

    ``reconstruct_kernel_instances`` has 24 entries: K in ``[1, 8]`` for the
    plain, ``mcg`` and ``mul1`` variants. Anything else indexes past the table
    (or runs the wrong kernel); the GEMM path already enforces this range.
    """

    if not 1 <= int(k) <= 8:
        raise ValueError(f"EXL3 trellis K must be in [1, 8], got {k}")
    if mcg and mul1:
        raise ValueError("EXL3 trellis cannot set both mcg and mul1")


__all__ = ["validate_trellis_k"]
