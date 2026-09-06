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
def _rank8_tensor_core_projection(X, A, U, M: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    columns = tl.arange(0, 64)
    rank = tl.arange(0, 16)
    accumulator = tl.zeros((32, 16), tl.float32)
    for tile in range(tl.cdiv(K, 64)):
        inner = tile * 64 + columns
        x = tl.load(X + rows[:, None] * K + inner[None, :],
                    (rows[:, None] < M) & (inner[None, :] < K), other=0)
        a = tl.load(A + inner[:, None] * 8 + rank[None, :],
                    (inner[:, None] < K) & (rank[None, :] < 8), other=0)
        accumulator = tl.dot(x, a, accumulator)
    tl.store(U + rows[:, None] * 8 + rank[None, :], accumulator,
             (rows[:, None] < M) & (rank[None, :] < 8))


def rank8_tensor_core_projection(transformed, a):
    """Project published FP16 X' with rank padded to 16 and FP32 accumulation."""
    if transformed.ndim != 2 or a.shape != (transformed.shape[-1], 8):
        raise ValueError("rank8 Tensor Core projection requires X[M,K] and A[K,8]")
    if transformed.device.type != "cuda" or torch.cuda.get_device_capability(
        transformed.device
    ) != (9, 0):
        raise ValueError("rank8 Tensor Core projection requires SM90")
    if a.device != transformed.device or any(
        t.dtype != torch.float16 or not t.is_contiguous() for t in (transformed, a)
    ):
        raise ValueError("rank8 Tensor Core projection requires contiguous FP16 inputs on one device")
    m, k = transformed.shape
    if k < 1:
        raise ValueError("rank8 Tensor Core projection requires positive K")
    hidden = torch.empty((m, 8), device=transformed.device, dtype=torch.float16)
    if m:
        _rank8_tensor_core_projection[(triton.cdiv(m, 32),)](
            transformed, a, hidden, m, k, num_warps=4, num_stages=2
        )
    return hidden


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
    HAS_RANK8: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, N)
    # fma contraction is disabled at launch. Products of two FP16 values are
    # exact in FP32; keep the base outside this eight-term reduction.
    if HAS_RANK8:
        correction = tl.full((N,), 0, tl.float32)
        for rank in tl.static_range(8):
            hidden = tl.load(Hidden + row * 8 + rank).to(tl.float32)
            weights = tl.load(B + rank * N + column).to(tl.float32)
            correction = correction + hidden * weights
    value = tl.load(Base + row * BASE_STRIDE + column).to(tl.float32)
    if HAS_RANK8:
        value = value + correction
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


def rank8_output_epilogue(
    hidden, b, base, sv, bias=None, *, hadamard=True, output_dtype=torch.float32, rank8_enabled=True
):
    """Complete the existing output epilogue with an optional final FP16 store."""
    if output_dtype not in (torch.float16, torch.float32):
        raise ValueError("rank8 epilogue output must be FP16 or FP32")
    if base.ndim != 2 or (rank8_enabled and hidden.ndim != 2):
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
        or (rank8_enabled and (hidden.dtype != torch.float16 or b.dtype != torch.float16))
    ):
        raise ValueError("rank8 fused epilogue requires FP32 base and FP16 hidden/B")
    if (rank8_enabled and (hidden.shape != (m, 8) or b.shape != (8, n))) or sv.shape != (n,):
        raise ValueError("rank8 fused epilogue shape mismatch")
    if base.stride(1) != 1 or (rank8_enabled and (not hidden.is_contiguous() or not b.is_contiguous())):
        raise ValueError("rank8 epilogue requires contiguous columns and factors")
    values = (sv,) + ((hidden, b) if rank8_enabled else ()) + (() if bias is None else (bias,))
    if any(value.device != base.device for value in values):
        raise ValueError("rank8 epilogue inputs must share one device")
    if not sv.is_contiguous() or (
        bias is not None and (bias.shape != (n,) or not bias.is_contiguous())
    ):
        raise ValueError("rank8 epilogue requires contiguous SV/bias vectors")
    output = torch.empty((m, n), device=base.device, dtype=output_dtype)
    if not m:
        return output
    divisor = struct.unpack("e", struct.pack("e", math.sqrt(n)))[0]
    root = struct.unpack("f", struct.pack("f", math.sqrt(n)))[0]
    reciprocal = struct.unpack("f", struct.pack("f", 1 / root))[0]
    _rank8_output_epilogue[(m,)](
        hidden if rank8_enabled else None,
        b if rank8_enabled else None,
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
        rank8_enabled,
        num_warps=4 if n <= 4096 else 8,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _rank8_input_producer(
    X,
    SU,
    Factors,
    Transformed,
    Hidden,
    M: tl.constexpr,
    K: tl.constexpr,
    LOG_K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    GROUPS: tl.constexpr,
    HADAMARD: tl.constexpr,
    DIVISOR: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, K)
    value = tl.load(X + row * X_STRIDE + column).to(tl.float32)
    value = value * tl.load(SU + column).to(tl.float32)
    if HADAMARD and K >= 2048:
        # Exact mode2 input semantics: rescue only overflowing SU multiply,
        # normalize, then retain every historical FP16 butterfly boundary.
        value = _round_half_finite(value)
        value = tl.div_rn(value, DIVISOR).to(tl.float16).to(tl.float32)
    else:
        value = value.to(tl.float16).to(tl.float32)
    if HADAMARD:
        for stage in tl.static_range(LOG_K):
            partner = tl.gather(value, column ^ (1 << stage), axis=0)
            value = tl.where(
                (column & (1 << stage)) == 0, value + partner, partner - value
            )
            value = value.to(tl.float16).to(tl.float32)
        if K < 2048:
            value = (value * RECIPROCAL).to(tl.float16).to(tl.float32)
    tl.store(Transformed + row * K + column, value)
    # The exact rounded activation stays live in the CTA. No recovery branch
    # reloads X or repeats SU/H, including independent sibling projections.
    for group in tl.static_range(GROUPS):
        for rank in tl.static_range(8):
            factor = tl.load(Factors[group] + column * 8 + rank).to(tl.float32)
            projected = tl.sum(value * factor, axis=0)
            tl.store(Hidden + group * M * 8 + row * 8 + rank, projected)


@triton.jit
def _rank8_input_producer_masked(
    X,
    SU,
    Factors,
    Transformed,
    Hidden,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    X_STRIDE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Composite-width producer for projections whose input Hadamard is folded."""

    row = tl.program_id(0)
    column = tl.arange(0, BLOCK_K)
    mask = column < K
    value = tl.load(X + row * X_STRIDE + column, mask=mask, other=0).to(tl.float32)
    value = value * tl.load(SU + column, mask=mask, other=0).to(tl.float32)
    value = value.to(tl.float16).to(tl.float32)
    tl.store(Transformed + row * K + column, value, mask=mask)
    for group in tl.static_range(GROUPS):
        for rank in tl.static_range(8):
            factor = tl.load(
                Factors[group] + column * 8 + rank,
                mask=mask,
                other=0,
            ).to(tl.float32)
            projected = tl.sum(value * factor, axis=0)
            tl.store(Hidden + group * M * 8 + row * 8 + rank, projected)


def rank8_input_producer(x, su, factors, *, hadamard=True):
    """Publish shared X-prime and one FP16 rank8 projection per enabled child.

    Each CTA owns an entire row and its reductions. Factors are passed as a
    tuple of pointers, so grouped consumers need no concatenated factor cache.
    """
    factors = (factors,) if isinstance(factors, torch.Tensor) else tuple(factors)
    if x.ndim != 2 or x.device.type != "cuda" or x.dtype != torch.float16:
        raise ValueError("rank8 input producer requires FP16 CUDA [M,K]")
    m, k = x.shape
    if k < 16 or k > 16384 or (hadamard and k & (k - 1)) or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError(
            "rank8 input producer requires SM90 and K in [16,16384]; Hadamard mode requires power-of-two K"
        )
    if not 1 <= len(factors) <= 3:
        raise ValueError("rank8 input producer requires one to three enabled children")
    if (
        su.shape != (k,)
        or su.dtype != torch.float16
        or su.device != x.device
        or not su.is_contiguous()
    ):
        raise ValueError(
            "rank8 input producer requires contiguous FP16 SU on the input device"
        )
    if x.stride(1) != 1 or any(
        a.shape != (k, 8)
        or a.dtype != torch.float16
        or a.device != x.device
        or not a.is_contiguous()
        for a in factors
    ):
        raise ValueError(
            "rank8 input producer requires contiguous columns and FP16 [K,8] factors"
        )
    transformed = torch.empty((m, k), device=x.device, dtype=torch.float16)
    hidden = torch.empty((len(factors), m, 8), device=x.device, dtype=torch.float16)
    if m and not hadamard and k & (k - 1):
        block_k = 1 << (k - 1).bit_length()
        _rank8_input_producer_masked[(m,)](
            x,
            su,
            factors,
            transformed,
            hidden,
            m,
            k,
            block_k,
            x.stride(0),
            len(factors),
            num_warps=4 if k <= 4096 else 8,
            enable_fp_fusion=False,
        )
    elif m:
        divisor = struct.unpack("e", struct.pack("e", math.sqrt(k)))[0]
        root = struct.unpack("f", struct.pack("f", math.sqrt(k)))[0]
        reciprocal = struct.unpack("f", struct.pack("f", 1 / root))[0]
        _rank8_input_producer[(m,)](
            x,
            su,
            factors,
            transformed,
            hidden,
            m,
            k,
            k.bit_length() - 1,
            x.stride(0),
            len(factors),
            hadamard,
            divisor,
            reciprocal,
            num_warps=4 if k <= 4096 else 8,
            enable_fp_fusion=False,
        )
    return transformed, hidden.unbind(0)
