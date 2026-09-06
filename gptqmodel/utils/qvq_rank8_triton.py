# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""P32 rank8 expansion/add/output transform with explicit historical rounding.

One CTA owns one complete output row. Hidden/B remain FP16, products and
accumulation FP32; the base is added only after the rank reduction finishes.
The output butterfly mirrors qvq_hadamard_cuda.cu modes 3/4, including its
per-operation overflow rescue. No grid barrier or persistent scratch is used.
"""

from __future__ import annotations

import math
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _round_half_finite(value):
    narrowed = value.to(tl.float16).to(tl.float32)
    return tl.where(tl.abs(narrowed) < float("inf"), narrowed, value)


@triton.jit
def _rank8_output_epilogue(
    Hidden,
    B,
    Base,
    SV,
    Bias,
    Output,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    HADAMARD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE_FIRST: tl.constexpr,
    DIVISOR: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, N)
    correction = tl.full((N,), 0, tl.float32)
    # fma contraction is disabled at launch. Products of two FP16 values are
    # exact in FP32; keep the base outside this eight-term reduction.
    for rank in tl.static_range(8):
        hidden = tl.load(Hidden + row * 8 + rank).to(tl.float32)
        weights = tl.load(B + rank * N + column).to(tl.float32)
        correction = correction + hidden * weights
    value = tl.load(Base + row * BASE_STRIDE + column).to(tl.float32) + correction
    if HADAMARD:
        value = _round_half_finite(value)
        if NORMALIZE_FIRST:
            value = _round_half_finite(tl.div_rn(value, DIVISOR))
        for stage in tl.static_range(0, LOG_N):
            partner = tl.gather(value, column ^ (1 << stage), axis=0)
            # Upper element of each pair is lower - upper, not upper - lower.
            value = tl.where(
                (column & (1 << stage)) == 0, value + partner, partner - value
            )
            value = _round_half_finite(value)
        if not NORMALIZE_FIRST:
            value = _round_half_finite(value * RECIPROCAL)
    value = value * tl.load(SV + column).to(tl.float32)
    if HADAMARD:
        value = _round_half_finite(value)
    if HAS_BIAS:
        value = value + tl.load(Bias + column).to(tl.float32)
        if HADAMARD:
            value = _round_half_finite(value)
    tl.store(Output + row * N + column, value)


def rank8_output_epilogue(hidden, b, base, sv, bias=None, *, hadamard=True):
    """Return FP32 full linear output, matching the existing output epilogue."""
    if base.ndim != 2 or hidden.ndim != 2:
        raise ValueError("rank8 epilogue requires matrix inputs")
    m, n = base.shape
    if n < 16 or n > 16384 or n & (n - 1):
        raise ValueError("rank8 fused epilogue requires power-of-two N in [16,16384]")
    if base.device.type != "cuda" or torch.cuda.get_device_capability(base.device) != (
        9,
        0,
    ):
        raise ValueError("rank8 fused epilogue requires SM90")
    if (
        base.dtype != torch.float32
        or hidden.dtype != torch.float16
        or b.dtype != torch.float16
    ):
        raise ValueError("rank8 fused epilogue requires FP32 base and FP16 hidden/B")
    if hidden.shape != (m, 8) or b.shape != (8, n) or sv.shape != (n,):
        raise ValueError("rank8 fused epilogue shape mismatch")
    if base.stride(1) != 1 or not hidden.is_contiguous() or not b.is_contiguous():
        raise ValueError("rank8 epilogue requires contiguous columns and factors")
    values = (hidden, b, sv) if bias is None else (hidden, b, sv, bias)
    if any(value.device != base.device for value in values):
        raise ValueError("rank8 epilogue inputs must share one device")
    if not sv.is_contiguous() or (
        bias is not None and (bias.shape != (n,) or not bias.is_contiguous())
    ):
        raise ValueError("rank8 epilogue requires contiguous SV/bias vectors")
    output = torch.empty((m, n), device=base.device, dtype=torch.float32)
    if not m:
        return output
    divisor = struct.unpack("e", struct.pack("e", math.sqrt(n)))[0]
    root = struct.unpack("f", struct.pack("f", math.sqrt(n)))[0]
    reciprocal = struct.unpack("f", struct.pack("f", 1 / root))[0]
    _rank8_output_epilogue[(m,)](
        hidden,
        b,
        base,
        sv,
        bias,
        output,
        n,
        n.bit_length() - 1,
        base.stride(0),
        hadamard,
        bias is not None,
        n >= 2048,
        divisor,
        reciprocal,
        num_warps=4 if n <= 4096 else 8,
        enable_fp_fusion=False,
    )
    return output
