# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CUDA execution over out-axis Tenstorrent BF4_B and BF8_B checkpoint bytes.

The checkpoint stores W.T as [K, N] with one signed exponent per 16 columns.
The kernels decode a tile at a time and add the bias to the FP32 reduction
before casting to the activation dtype. Split-K decode keeps only partial
outputs in scratch.
"""

from __future__ import annotations

import torch

from . import tt_blockfloat_cuda


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _decode_tile(
        Codes,
        Exponents,
        k,
        pid_n,
        K,
        N: tl.constexpr,
        BITS: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        n = pid_n * BN + tl.arange(0, BN)
        in_bounds = (k[:, None] < K) & (n[None, :] < N)
        # An output head's flattened weight can exceed 32-bit element offsets.
        row = k.to(tl.int64)[:, None]
        group = pid_n * (BN // 16) + tl.arange(0, BN // 16)
        exponent = tl.load(
            Exponents + row * (N // 16) + group[None, :], (k[:, None] < K) & (group[None, :] < N // 16), other=0
        ).to(tl.int32)
        # Build exact 2**e, including the subnormal bins, without a fast exp2
        # instruction that can flush the two lowest exponent bins to zero.
        scale_bits = tl.where(exponent >= -126, (exponent + 127) << 23, tl.where(exponent == -127, 1 << 22, 1 << 21))
        scale = tl.broadcast_to(scale_bits.to(tl.float32, bitcast=True)[:, :, None], (BK, BN // 16, 16)).reshape(
            (BK, BN)
        )

        if BITS == 4:
            lane = pid_n * (BN // 2) + tl.arange(0, BN // 2)
            word = tl.load(
                Codes + row * (N // 2) + lane[None, :], (k[:, None] < K) & (lane[None, :] < N // 2), other=0
            ).to(tl.int32)
            code = tl.interleave(word & 15, word >> 4)
        else:
            code = tl.load(Codes + row * N + n[None, :], in_bounds, other=0).to(tl.int32)
        levels: tl.constexpr = 1 << (BITS - 2)
        unit = (code - (2 * levels - 1)).to(tl.float32) * (1.0 / levels)
        return tl.where(in_bounds, unit * scale, 0.0)

    @triton.jit
    def _gemv_partials(
        X,
        Codes,
        Exponents,
        Partial,
        K,
        N: tl.constexpr,
        stride_m,
        stride_k,
        M: tl.constexpr,
        SPLITS: tl.constexpr,
        BITS: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(0)
        k = pid_k * BK + tl.arange(0, BK)
        n = pid_n * BN + tl.arange(0, BN)
        w = _decode_tile(Codes, Exponents, k, pid_n, K, N, BITS, BK, BN)
        for m in tl.static_range(M):
            x = tl.load(X + m * stride_m + k.to(tl.int64) * stride_k, k < K, other=0.0).to(tl.float32)
            part = tl.sum(x[:, None] * w, axis=0)
            tl.store(Partial + (m * SPLITS + pid_k.to(tl.int64)) * N + n, part, n < N)

    @triton.jit
    def _split_k_epilogue(
        Partial,
        Bias,
        Out,
        N: tl.constexpr,
        stride_bias,
        SPLITS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BP: tl.constexpr,
        BN: tl.constexpr,
    ):
        m = tl.program_id(1).to(tl.int64)
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        p = tl.arange(0, BP)
        part = tl.load(
            Partial + (m * SPLITS + p[:, None]) * N + n[None, :], (p[:, None] < SPLITS) & (n[None, :] < N), other=0.0
        )
        y = tl.sum(part, axis=0)
        if HAS_BIAS:
            y += tl.load(Bias + n * stride_bias, n < N, other=0.0).to(tl.float32)
        tl.store(Out + m * N + n, y, n < N)

    @triton.jit
    def _gemm(
        X,
        Codes,
        Exponents,
        Bias,
        Out,
        M,
        K,
        N: tl.constexpr,
        SPLITS: tl.constexpr,
        stride_m,
        stride_k,
        stride_bias,
        BITS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        BN: tl.constexpr,
    ):
        m = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
        pid_n = tl.program_id(1)
        n = pid_n * BN + tl.arange(0, BN)
        partition = tl.program_id(2)
        blocks = tl.cdiv(K, BK * SPLITS)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for block in range(blocks):
            k = (partition * blocks + block) * BK + tl.arange(0, BK)
            x = tl.load(
                X + m[:, None] * stride_m + k[None, :].to(tl.int64) * stride_k,
                (m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            w = _decode_tile(Codes, Exponents, k, pid_n, K, N, BITS, BK, BN)
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if HAS_BIAS:
            acc += tl.load(Bias + n * stride_bias, n < N, other=0.0).to(tl.float32)[None, :]
        offsets = (m[:, None] * SPLITS + partition) * N + n[None, :]
        tl.store(Out + offsets, acc, (m[:, None] < M) & (n[None, :] < N))


def linear_out_axis(
    x: torch.Tensor,
    codes: torch.Tensor,
    exponents: torch.Tensor,
    bits: int,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run a packed out-axis projection, or return ``None`` for the Torch path."""

    if triton is None or x.device.type != "cuda" or not tt_blockfloat_cuda.supported(x.device):
        return None
    if bits not in (4, 8) or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad)):
        return None
    if codes.device != x.device or exponents.device != x.device or (bias is not None and bias.device != x.device):
        return None
    if codes.dtype != torch.uint8 or exponents.dtype != torch.int8:
        return None
    if not codes.is_contiguous() or not exponents.is_contiguous() or exponents.ndim != 2:
        return None

    k, groups = exponents.shape
    n = groups * tt_blockfloat_cuda.BLOCKFLOAT_BLOCK
    if x.shape[-1] != k or codes.numel() != k * n * bits // 8 or not x.numel() or not n:
        return None
    if bias is not None and bias.shape != (n,):
        return None
    bn = 64
    tiles_n = triton.cdiv(n, bn)
    if tiles_n > 65535:  # CUDA's second grid dimension limit in _gemm
        return None
    activation = x.reshape(-1, k)
    m = activation.size(0)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    stride_bias = bias.stride(0) if bias is not None else 0
    with torch.cuda.device(x.device):
        gemv_bk = 128
        gemv_splits = triton.cdiv(k, gemv_bk)
        # Bound the GEMV accumulation and epilogue tile. Very wide inputs use
        # GEMM's tiled loop with at most four reduction partitions.
        if m <= 4 and gemv_splits <= 64:
            splits = gemv_splits
            partial = torch.empty((m, splits, n), dtype=torch.float32, device=x.device)
            _gemv_partials[(tiles_n, splits)](
                activation,
                codes,
                exponents,
                partial,
                k,
                n,
                *activation.stride(),
                m,
                splits,
                bits,
                gemv_bk,
                bn,
            )
        else:
            bm, bk = 16, 32
            tiles_m = triton.cdiv(m, bm)
            # A short prefill has few output tiles. Split its K axis to expose
            # more parallel work without expanding or storing any weight tile.
            splits = 4 if tiles_m * tiles_n < 128 and k >= 512 else 1
            partial = torch.empty((m, splits, n), dtype=torch.float32, device=x.device) if splits > 1 else out
            _gemm[(tiles_m, tiles_n, splits)](
                activation,
                codes,
                exponents,
                bias,
                partial,
                m,
                k,
                n,
                splits,
                *activation.stride(),
                stride_bias,
                bits,
                bias is not None and splits == 1,
                bm,
                bk,
                bn,
            )
        if partial is not out:
            _split_k_epilogue[(tiles_n, m)](
                partial,
                bias,
                out,
                n,
                stride_bias,
                splits,
                bias is not None,
                triton.next_power_of_2(splits),
                bn,
            )
    return out.reshape(*x.shape[:-1], n)
