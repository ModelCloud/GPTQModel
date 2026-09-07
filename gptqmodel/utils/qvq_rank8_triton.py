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
import threading

import torch
import triton
import triton.language as tl

from gptqmodel.quantization.rotation.hadamard_utils import get_hadK, matmul_hadU_stable

_GRAPH_WARM_KEYS: set[tuple[object, ...]] = set()
_GRAPH_WARM_KEYS_LOCK = threading.Lock()


def _rank8_num_warps(n: int, requested: int | None) -> int:
    """Resolve the epilogue launch width from the selected BM/BN policy.

    The rank8 epilogue is part of the complete tuned operator.  When the base
    window candidate supplies a BN-sized warp policy, carry that decision into
    the correction epilogue instead of silently launching a different fixed
    geometry.  A zero/None request preserves the historical width heuristic.
    """

    if requested in (None, 0):
        return 4 if n <= 4096 else 8
    if requested not in (1, 2, 4, 8):
        raise ValueError("rank8 epilogue num_warps must be one of 1, 2, 4, or 8")
    return requested


def _rank8_graph_key(device: torch.device, *parts: object) -> tuple[object, ...]:
    return (device.type, device.index, *parts)


def _require_rank8_kernel_warm(key: tuple[object, ...]) -> None:
    """Reject a first Triton compilation/launch from inside graph capture."""

    if (
        torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        with _GRAPH_WARM_KEYS_LOCK:
            warm = key in _GRAPH_WARM_KEYS
        if not warm:
            raise RuntimeError(
                "QVQ rank8 Triton kernel must be warmed before CUDA Graph capture"
            )


def _mark_rank8_kernel_warm(key: tuple[object, ...]) -> None:
    with _GRAPH_WARM_KEYS_LOCK:
        _GRAPH_WARM_KEYS.add(key)


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


def rank8_tensor_core_projection(transformed, a, *, num_warps=None):
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
        # Keep the published Tensor Core projection default at four warps;
        # only an explicit BM/BN policy may widen it.
        launch_warps = _rank8_num_warps(2048, num_warps)
        key = _rank8_graph_key(
            transformed.device, "tensor_core_projection", m, k, launch_warps
        )
        _require_rank8_kernel_warm(key)
        _rank8_tensor_core_projection[(triton.cdiv(m, 32),)](
            transformed, a, hidden, m, k, num_warps=launch_warps, num_stages=2
        )
        _mark_rank8_kernel_warm(key)
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


@triton.jit
def _rank8_output_epilogue_masked(
    Hidden,
    B,
    Base,
    SV,
    Bias,
    Output,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RANK8: tl.constexpr,
):
    """Fused non-Hadamard epilogue for composite output widths."""

    row = tl.program_id(0)
    column = tl.arange(0, BLOCK_N)
    mask = column < N
    if HAS_RANK8:
        correction = tl.zeros((BLOCK_N,), tl.float32)
        for rank in tl.static_range(8):
            hidden = tl.load(Hidden + row * 8 + rank).to(tl.float32)
            weights = tl.load(B + rank * N + column, mask=mask, other=0).to(tl.float32)
            correction = correction + hidden * weights
    value = tl.load(Base + row * BASE_STRIDE + column, mask=mask, other=0).to(tl.float32)
    if HAS_RANK8:
        value = value + correction
    value = value * tl.load(SV + column, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        value = value + tl.load(Bias + column, mask=mask, other=0).to(tl.float32)
    tl.store(Output + row * N + column, value, mask=mask)


@triton.jit
def _rank8_project_output_epilogue(
    X,
    A,
    B,
    Base,
    SV,
    Bias,
    Output,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    HADAMARD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE_FIRST: tl.constexpr,
    DIVISOR: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    """Project X' and expand the rank-8 correction in one row-owned CTA.

    This is the no-hidden-buffer fused path.  The projection accumulator is
    explicitly narrowed to FP16 before B expansion, matching the versioned
    rank8 contract.  The output transform is kept byte-for-byte in the same
    ordering as :func:`_rank8_output_epilogue`.
    """

    row = tl.program_id(0)
    column = tl.arange(0, N)
    rank = tl.arange(0, 8)
    projected = tl.zeros((8,), tl.float32)
    for offset in tl.range(0, K, BLOCK_K):
        k = offset + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + k, mask=k < K, other=0).to(tl.float32)
        a = tl.load(
            A + k[:, None] * 8 + rank[None, :],
            mask=k[:, None] < K,
            other=0,
        ).to(tl.float32)
        projected += tl.sum(x[:, None] * a, axis=0)
    # Preserve the declared hidden FP16 boundary before the expansion.
    hidden = projected.to(tl.float16).to(tl.float32)
    weights = tl.load(B + rank[:, None] * N + column[None, :]).to(tl.float32)
    correction = tl.sum(hidden[:, None] * weights, axis=0)
    value = tl.load(Base + row * BASE_STRIDE + column).to(tl.float32) + correction
    if HADAMARD:
        value = _round_half_finite(value)
        if NORMALIZE_FIRST:
            value = _round_half_finite(tl.div_rn(value, DIVISOR))
        for stage in tl.static_range(0, LOG_N):
            partner = tl.gather(value, column ^ (1 << stage), axis=0)
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


@triton.jit
def _rank8_project_output_epilogue_masked(
    X,
    A,
    B,
    Base,
    SV,
    Bias,
    Output,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Composite-width variant of the fused projection/output epilogue."""

    row = tl.program_id(0)
    column = tl.arange(0, BLOCK_N)
    mask = column < N
    rank = tl.arange(0, 8)
    projected = tl.zeros((8,), tl.float32)
    for offset in tl.range(0, K, BLOCK_K):
        k = offset + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + k, mask=k < K, other=0).to(tl.float32)
        a = tl.load(
            A + k[:, None] * 8 + rank[None, :],
            mask=k[:, None] < K,
            other=0,
        ).to(tl.float32)
        projected += tl.sum(x[:, None] * a, axis=0)
    hidden = projected.to(tl.float16).to(tl.float32)
    weights = tl.load(
        B + rank[:, None] * N + column[None, :],
        mask=mask[None, :],
        other=0,
    ).to(tl.float32)
    correction = tl.sum(hidden[:, None] * weights, axis=0)
    value = tl.load(
        Base + row * BASE_STRIDE + column, mask=mask, other=0
    ).to(tl.float32) + correction
    value = value * tl.load(SV + column, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        value = value + tl.load(Bias + column, mask=mask, other=0).to(tl.float32)
    tl.store(Output + row * N + column, value, mask=mask)


def rank8_output_epilogue(
    hidden,
    b,
    base,
    sv,
    bias=None,
    *,
    hadamard=True,
    output_dtype=torch.float32,
    rank8_enabled=True,
    num_warps=None,
):
    """Complete the existing output epilogue with an optional final FP16 store."""
    if output_dtype not in (torch.float16, torch.float32):
        raise ValueError("rank8 epilogue output must be FP16 or FP32")
    if base.ndim != 2 or (rank8_enabled and hidden.ndim != 2):
        raise ValueError("rank8 epilogue requires matrix inputs")
    m, n = base.shape
    composite_hadamard = bool(hadamard and n & (n - 1))
    if n < 16 or n > 17408:
        raise ValueError(
            "rank8 fused epilogue requires N in [16,17408]"
        )
    if composite_hadamard:
        try:
            had_n, base_n = get_hadK(n)
        except AssertionError as error:
            raise ValueError("unsupported composite Hadamard output width") from error
        power_two_width = n // base_n
        if had_n is None or power_two_width < 2 or power_two_width & (power_two_width - 1):
            raise ValueError("unsupported composite Hadamard output width")
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
    launch_warps = _rank8_num_warps(n, num_warps)
    key = _rank8_graph_key(
        base.device,
        "output_epilogue",
        m,
        n,
        bool(hadamard),
        bias is not None,
        bool(rank8_enabled),
        str(output_dtype),
        str(sv.dtype),
        None if bias is None else str(bias.dtype),
        "composite" if composite_hadamard else ("masked" if not hadamard and n & (n - 1) else "butterfly"),
        launch_warps,
    )
    _require_rank8_kernel_warm(key)
    if composite_hadamard:
        # Keep the same FP32 base/correction arithmetic, then delegate the
        # existing factored Hadamard implementation for the composite width.
        # This fallback is intentionally separate from the power-of-two fused
        # kernel until a native composite output kernel is certified.
        added = base
        if rank8_enabled:
            added = base + hidden.float() @ b.float()
        narrowed = added.to(torch.float16)
        historical = matmul_hadU_stable(narrowed)
        if sv is not None:
            historical = historical * sv.to(torch.float16)
        if bias is not None:
            historical = historical + bias.to(torch.float16)
        historical_finite = torch.isfinite(historical).all()
        if not torch.cuda.is_current_stream_capturing() and bool(historical_finite):
            output = historical.to(output_dtype)
        else:
            rescue = matmul_hadU_stable(added)
            if sv is not None:
                rescue = rescue * sv
            if bias is not None:
                rescue = rescue + bias
            output = torch.where(historical_finite, historical.to(added.dtype), rescue)
            output = output.to(output_dtype)
        _mark_rank8_kernel_warm(key)
        return output
    output = torch.empty((m, n), device=base.device, dtype=output_dtype)
    if not m:
        return output
    if not hadamard and n & (n - 1):
        block_n = 1 << (n - 1).bit_length()
        _rank8_output_epilogue_masked[(m,)](
            hidden if rank8_enabled else None,
            b if rank8_enabled else None,
            base,
            sv,
            bias,
            output,
            n,
            block_n,
            base.stride(0),
            bias is not None,
            rank8_enabled,
            num_warps=launch_warps,
            enable_fp_fusion=False,
        )
    else:
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
            num_warps=launch_warps,
            enable_fp_fusion=False,
        )
    _mark_rank8_kernel_warm(key)
    return output


def rank8_project_output_epilogue(
    transformed,
    a,
    b,
    base,
    sv,
    bias=None,
    *,
    hadamard=True,
    output_dtype=torch.float32,
    num_warps=None,
):
    """Fuse ``X' @ A`` with rank8 expansion and the output epilogue.

    ``transformed`` is already the exact P32 activation domain.  The kernel
    therefore removes the intermediate hidden allocation and the separate
    projection launch while preserving the FP16 hidden rounding boundary.
    Preparation/warming is tracked independently for every shape so the first
    Triton compile cannot occur during CUDA Graph capture.
    """

    if output_dtype not in (torch.float16, torch.float32):
        raise ValueError("rank8 fused epilogue output must be FP16 or FP32")
    if (
        transformed.ndim != 2
        or a.ndim != 2
        or b.ndim != 2
        or base.ndim != 2
        or transformed.dtype != torch.float16
        or a.dtype != torch.float16
        or b.dtype != torch.float16
        or base.dtype != torch.float32
    ):
        raise ValueError(
            "fused rank8 projection requires FP16 X'/A/B and FP32 base"
        )
    m, k = transformed.shape
    base_m, n = base.shape
    if (
        base_m != m
        or a.shape != (k, 8)
        or b.shape != (8, n)
        or sv.shape != (n,)
        or n < 16
        or n > 16384
        or (hadamard and n & (n - 1))
    ):
        raise ValueError("fused rank8 projection/output shapes are unsupported")
    values = (transformed, a, b, base, sv) + (() if bias is None else (bias,))
    if any(value.device != base.device for value in values):
        raise ValueError("fused rank8 projection inputs must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("fused rank8 projection inputs must be contiguous")
    if bias is not None and (bias.shape != (n,)):
        raise ValueError("fused rank8 projection bias shape mismatch")
    if base.device.type != "cuda" or torch.cuda.get_device_capability(base.device) != (
        9,
        0,
    ):
        raise ValueError("fused rank8 projection requires SM90")
    output = torch.empty((m, n), device=base.device, dtype=output_dtype)
    if not m:
        return output
    launch_warps = _rank8_num_warps(n, num_warps)
    key = _rank8_graph_key(
        base.device,
        "project_output_epilogue",
        m,
        k,
        n,
        bool(hadamard),
        bias is not None,
        str(output_dtype),
        str(sv.dtype),
        None if bias is None else str(bias.dtype),
        "masked" if not hadamard and n & (n - 1) else "butterfly",
        launch_warps,
    )
    _require_rank8_kernel_warm(key)
    if not hadamard and n & (n - 1):
        block_n = 1 << (n - 1).bit_length()
        _rank8_project_output_epilogue_masked[(m,)](
            transformed,
            a,
            b,
            base,
            sv,
            bias,
            output,
            n,
            block_n,
            k,
            64,
            base.stride(0),
            bias is not None,
            num_warps=launch_warps,
            num_stages=2,
            enable_fp_fusion=False,
        )
    else:
        root = struct.unpack("f", struct.pack("f", math.sqrt(n)))[0]
        divisor = struct.unpack("e", struct.pack("e", math.sqrt(n)))[0]
        reciprocal = struct.unpack("f", struct.pack("f", 1 / root))[0]
        _rank8_project_output_epilogue[(m,)](
            transformed,
            a,
            b,
            base,
            sv,
            bias,
            output,
            n,
            n.bit_length() - 1,
            k,
            64,
            base.stride(0),
            hadamard,
            bias is not None,
            n >= 2048,
            divisor,
            reciprocal,
            output_dtype == torch.float16,
            num_warps=launch_warps,
            num_stages=2,
            enable_fp_fusion=False,
        )
    _mark_rank8_kernel_warm(key)
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
    APPLY_SU: tl.constexpr,
):
    """Composite-width producer for projections whose input Hadamard is folded."""

    row = tl.program_id(0)
    column = tl.arange(0, BLOCK_K)
    mask = column < K
    value = tl.load(X + row * X_STRIDE + column, mask=mask, other=0).to(tl.float32)
    if APPLY_SU:
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
    if k < 16 or k > 17408 or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError(
            "rank8 input producer requires SM90 and K in [16,17408]"
        )
    composite_hadamard = bool(hadamard and k & (k - 1))
    if composite_hadamard:
        try:
            had_k, base_k = get_hadK(k)
        except AssertionError as error:
            raise ValueError("unsupported composite Hadamard input width") from error
        power_two_width = k // base_k
        if had_k is None or power_two_width < 2 or power_two_width & (power_two_width - 1):
            raise ValueError("unsupported composite Hadamard input width")
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
    key = _rank8_graph_key(
        x.device,
        "input_producer",
        m,
        k,
        len(factors),
        bool(hadamard),
        "composite" if composite_hadamard else ("masked" if not hadamard and k & (k - 1) else "butterfly"),
    )
    _require_rank8_kernel_warm(key)
    if m and composite_hadamard:
        # Composite Hadamards are factored as the existing stable transform:
        # SU is applied once, the power-of-two stages use the established CUDA
        # path, and the small base transform remains in the same FP16 domain.
        # The masked producer then publishes X' and projects all children
        # without reapplying SU. The transform/cache must be warmed eagerly.
        transformed_input = matmul_hadU_stable(x.mul(su))
        block_k = 1 << (k - 1).bit_length()
        _rank8_input_producer_masked[(m,)](
            transformed_input,
            su,
            factors,
            transformed,
            hidden,
            m,
            k,
            block_k,
            transformed_input.stride(0),
            len(factors),
            APPLY_SU=False,
            num_warps=4 if k <= 4096 else 8,
            enable_fp_fusion=False,
        )
    elif m and not hadamard and k & (k - 1):
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
            APPLY_SU=True,
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
    _mark_rank8_kernel_warm(key)
    return transformed, hidden.unbind(0)
