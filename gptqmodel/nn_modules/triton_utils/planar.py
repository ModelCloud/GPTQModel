# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Triton kernels for the split-plane (planar, `gptq_p`) GPTQ layout.

Planar storage (see `gptqmodel/utils/planar_packing.py`): every 32 logical
codes occupy `bits` adjacent int32 words, low plane first. Within a plane of
width `w`, word `i` holds codes `[i*(32//w), (i+1)*(32//w))` at shifts `w*j`.
Decode per element is fixed shifts/masks OR-merged across planes — branch-free
and uniform per lane. `qweight` packs along rows, `qzeros` along columns, and
zero points use v2 semantics (no +1 bias).

Two entry points:

- ``planar_dequant``: decode the whole weight matrix to fp16/bf16 (matmul is
  then delegated to cuBLAS). Best for prefill-shape GEMM.
- ``planar_matmul``: fused dequant+matmul that reads the packed words
  directly, avoiding the dense fp16 weight round-trip through DRAM. Best for
  decode-shape (small-M) GEMM where the weight traffic dominates.
"""

import itertools
import os
from typing import List

import torch
import triton
import triton.language as tl

from ...utils.pangolin import g_idx_block_uniform
from ...utils.torch import HAS_XPU


# bits -> (width0, width1, width2); width2 == 0 means the plane is absent.
# Bit offsets equal the cumulative widths of the preceding planes, and so do
# the word-row offsets of each plane inside a 32-code block.
_PLANAR_PLANE_WIDTHS = {
    3: (2, 1, 0),
    5: (4, 1, 0),
    6: (4, 2, 0),
    7: (4, 2, 1),
}

PLANAR_TRITON_BITS = tuple(sorted(_PLANAR_PLANE_WIDTHS))


@triton.jit
def _planar_decode_rows(
    qweight_ptr,
    row_idx,
    col_idx,
    out_features,
    mask,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
):
    """Decode logical codes packed along rows: word row = (k//32)*BITS + plane offset."""
    base = (row_idx // 32) * BITS
    idx = row_idx % 32

    PF0: tl.constexpr = 32 // W0
    w0 = tl.load(
        qweight_ptr + (base + idx // PF0) * out_features + col_idx,
        mask=mask,
        other=0,
        eviction_policy="evict_last",
    )
    codes = (w0 >> (W0 * (idx % PF0))) & ((1 << W0) - 1)

    PF1: tl.constexpr = 32 // W1
    w1 = tl.load(
        qweight_ptr + (base + W0 + idx // PF1) * out_features + col_idx,
        mask=mask,
        other=0,
        eviction_policy="evict_last",
    )
    codes |= ((w1 >> (W1 * (idx % PF1))) & ((1 << W1) - 1)) << W0

    if W2 > 0:
        PF2: tl.constexpr = 32 // W2
        w2 = tl.load(
            qweight_ptr + (base + W0 + W1 + idx // PF2) * out_features + col_idx,
            mask=mask,
            other=0,
            eviction_policy="evict_last",
        )
        codes |= ((w2 >> (W2 * (idx % PF2))) & ((1 << W2) - 1)) << (W0 + W1)

    return codes


@triton.jit
def _planar_decode_cols(
    qzeros_ptr,
    group_idx,
    col_idx,
    zeros_words_per_row,
    mask,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
):
    """Decode logical codes packed along columns: word col = (n//32)*BITS + plane offset."""
    row_base = group_idx * zeros_words_per_row + (col_idx // 32) * BITS
    idx = col_idx % 32

    PF0: tl.constexpr = 32 // W0
    w0 = tl.load(
        qzeros_ptr + row_base + idx // PF0,
        mask=mask,
        other=0,
        eviction_policy="evict_last",
    )
    codes = (w0 >> (W0 * (idx % PF0))) & ((1 << W0) - 1)

    PF1: tl.constexpr = 32 // W1
    w1 = tl.load(
        qzeros_ptr + row_base + W0 + idx // PF1,
        mask=mask,
        other=0,
        eviction_policy="evict_last",
    )
    codes |= ((w1 >> (W1 * (idx % PF1))) & ((1 << W1) - 1)) << W0

    if W2 > 0:
        PF2: tl.constexpr = 32 // W2
        w2 = tl.load(
            qzeros_ptr + row_base + W0 + W1 + idx // PF2,
            mask=mask,
            other=0,
            eviction_policy="evict_last",
        )
        codes |= ((w2 >> (W2 * (idx % PF2))) & ((1 << W2) - 1)) << (W0 + W1)

    return codes


def _make_dequant_configs(block_sizes: List[int], num_warps: List[int]):
    return [
        triton.Config({"X_BLOCK": bs}, num_warps=ws)
        for bs, ws in itertools.product(block_sizes, num_warps)
    ]


# Wider space than the continuous dequant's fixed 1024/1-warp: ncu showed that
# config reaching only ~9% DRAM SOL at ~18% occupancy on A100 (see
# pangolin_kernel.md); the planar decode does 2-3 word loads per element, so it
# needs more in-flight warps per SM to keep the memory system busy.
_PLANAR_DEQUANT_CONFIGS = _make_dequant_configs([1024, 2048, 4096], [1, 2, 4, 8])


@triton.autotune(_make_dequant_configs([1024, 2048], [1, 4]), key=["numels"])
@triton.jit
def planar_zeros_kernel(
    qzeros_ptr,
    out_ptr,
    numels,
    out_features: tl.constexpr,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    """Decode column-packed qzeros to a dense int32 [num_groups, out_features] buffer."""
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels

    group_idx = x_index // out_features
    col_idx = x_index % out_features

    ZEROS_WORDS_PER_ROW: tl.constexpr = (out_features // 32) * BITS
    zeros = _planar_decode_cols(
        qzeros_ptr, group_idx, col_idx, ZEROS_WORDS_PER_ROW, xmask, BITS, W0, W1, W2
    )
    tl.store(out_ptr + x_index, zeros, mask=xmask)


def _make_row_configs():
    return [
        triton.Config({"BLOCK_N": bn}, num_warps=ws)
        for bn, ws in itertools.product([512, 1024, 2048], [2, 4, 8])
    ]


@triton.autotune(_make_row_configs(), key=["out_features"])
@triton.jit
def planar_dequant_block_kernel(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    zeros_dense_ptr,
    out_ptr,
    out_dtype: tl.constexpr,
    num_groups,
    out_features: tl.constexpr,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """32-row-block-per-program planar weight decode.

    Each program decodes one 32-code block over BLOCK_N columns. The `BITS`
    packed word rows are re-read per unrolled row but stay resident in L1, so
    each word row leaves L2/DRAM once per program instead of once per output
    row, and the static unroll makes every shift amount and word-row offset a
    compile-time constant (no per-element index ALU at all).
    """
    pid_n = tl.program_id(0)
    blk = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < out_features

    base = blk * BITS
    row0 = blk * 32

    PF0: tl.constexpr = 32 // W0
    PF1: tl.constexpr = 32 // W1
    # max(W2, 1) avoids division by zero; PF2 is only used when W2 > 0.
    PF2: tl.constexpr = 32 // (W2 if W2 > 0 else 1)

    for k in tl.static_range(32):
        g = tl.load(g_idx_ptr + row0 + k)
        g = tl.where(g < 0, g + num_groups, g)
        g = tl.cast(g, tl.int32)

        scales = tl.cast(
            tl.load(scales_ptr + g * out_features + offs_n, mask=mask, eviction_policy="evict_last"),
            tl.float32,
        )
        zeros = tl.load(
            zeros_dense_ptr + g * out_features + offs_n, mask=mask, eviction_policy="evict_last"
        )

        w0 = tl.load(
            qweight_ptr + (base + k // PF0) * out_features + offs_n,
            mask=mask,
            eviction_policy="evict_last",
        )
        codes = (w0 >> (W0 * (k % PF0))) & ((1 << W0) - 1)

        w1 = tl.load(
            qweight_ptr + (base + W0 + k // PF1) * out_features + offs_n,
            mask=mask,
            eviction_policy="evict_last",
        )
        codes |= ((w1 >> (W1 * (k % PF1))) & ((1 << W1) - 1)) << W0

        if W2 > 0:
            w2 = tl.load(
                qweight_ptr + (base + W0 + W1 + k // PF2) * out_features + offs_n,
                mask=mask,
                eviction_policy="evict_last",
            )
            codes |= ((w2 >> (W2 * (k % PF2))) & ((1 << W2) - 1)) << (W0 + W1)

        result = tl.cast(codes - zeros, tl.float32) * scales
        tl.store(out_ptr + (row0 + k) * out_features + offs_n, tl.cast(result, out_dtype), mask=mask)


@triton.autotune(_PLANAR_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def planar_dequant_kernel(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    out_dtype: tl.constexpr,
    numels,
    num_groups,
    out_features: tl.constexpr,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels

    row_idx = x_index // out_features
    col_idx = x_index % out_features

    g_idx = tl.load(g_idx_ptr + row_idx, mask=xmask, eviction_policy="evict_last")
    groups = tl.where(g_idx < 0, g_idx + num_groups, g_idx)

    scales = tl.cast(
        tl.load(
            scales_ptr + (col_idx + out_features * groups),
            mask=xmask,
            eviction_policy="evict_last",
        ),
        tl.float32,
    )

    ZEROS_WORDS_PER_ROW: tl.constexpr = (out_features // 32) * BITS
    zeros = _planar_decode_cols(
        qzeros_ptr, groups, col_idx, ZEROS_WORDS_PER_ROW, xmask, BITS, W0, W1, W2
    )
    weights = _planar_decode_rows(
        qweight_ptr, row_idx, col_idx, out_features, xmask, BITS, W0, W1, W2
    )

    result = (tl.cast(weights, tl.float32) - tl.cast(zeros, tl.float32)) * scales
    tl.store(out_ptr + x_index, tl.cast(result, out_dtype), mask=xmask)


def _torch_dtype_to_triton(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _plane_widths(bits: int):
    widths = _PLANAR_PLANE_WIDTHS.get(bits)
    if widths is None:
        raise ValueError(
            f"planar Triton kernels support bits {PLANAR_TRITON_BITS}, got bits={bits}"
        )
    return widths


def planar_dequant(dtype, qweight, scales, qzeros, g_idx, bits: int) -> torch.Tensor:
    """Decode a planar-packed weight matrix to a dense `[in_features, out_features]` tensor."""
    w0, w1, w2 = _plane_widths(bits)

    num_groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = g_idx.shape[0]
    # The planar layout stores whole 32-code blocks; the block kernel covers
    # exactly in_features // 32 blocks, so unaligned rows would stay
    # uninitialized in the `torch.empty` output.
    if in_features % 32 != 0:
        raise ValueError(f"planar_dequant requires in_features divisible by 32, got {in_features}.")

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=dtype)

    device_ctx = torch.xpu.device(qweight.device) if HAS_XPU else torch.cuda.device(qweight.device)
    # The block-per-program kernel maps 32-row blocks to grid dim 1 (max
    # 65535); fall back to the elementwise kernel for larger in_features.
    if in_features // 32 <= 65535:
        grid = lambda meta: (triton.cdiv(out_features, meta["BLOCK_N"]), in_features // 32)  # noqa: E731
        with device_ctx:
            zeros_dense = _zeros_dense(qzeros, num_groups, out_features, bits, w0, w1, w2)
            planar_dequant_block_kernel[grid](
                g_idx,
                scales,
                qweight,
                zeros_dense,
                out,
                _torch_dtype_to_triton(dtype),
                num_groups,
                out_features=out_features,
                BITS=bits,
                W0=w0,
                W1=w1,
                W2=w2,
            )
        return out

    numels = out.numel()
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    with device_ctx:
        planar_dequant_kernel[grid](
            g_idx,
            scales,
            qweight,
            qzeros,
            out,
            _torch_dtype_to_triton(dtype),
            numels,
            num_groups,
            out_features=out_features,
            BITS=bits,
            W0=w0,
            W1=w1,
            W2=w2,
        )
    return out


def _make_gemv_configs():
    # Config space is kept tiny on purpose: the 32-row static unroll makes each
    # specialization expensive to compile, and autotuning reruns per (N, K,
    # BLOCK_M) key at first use in real inference.
    # Deep split-K is essential: with BLOCK_N=256 an N=4096 GEMV has only 16
    # n-tiles, so the K split must supply the rest of the SM-level parallelism
    # (ncu: SPLIT_K=4 left a 124-SM A100 at 6% occupancy and 45 GB/s).
    return [
        triton.Config({"BLOCK_N": bn, "SPLIT_K": sk}, num_warps=4)
        for bn, sk in itertools.product([128, 256], [16, 32])
    ]


@triton.autotune(_make_gemv_configs(), key=["N", "K", "BLOCK_M", "GROUP_UNIFORM"], reset_to_zero=["out_ptr"])
@triton.jit
def planar_gemv_kernel(
    x_ptr,
    qweight_ptr,
    zeros_dense_ptr,
    scales_ptr,
    g_idx_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_om,
    num_groups,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_UNIFORM: tl.constexpr,
):
    """Fused decode-regime GEMV: out[M, N] = x[M, K] @ dequant(qweight)[K, N] for small M.

    Bandwidth-first: weight-side DRAM traffic is only the packed words
    (`bits/8` bytes per element) — no dense fp16 weight round-trip. Each
    program walks 32-code blocks with the same static unroll as
    `planar_dequant_block_kernel` (word rows L1-resident, compile-time
    shifts) and accumulates `x[m, k] * w[k, n]` with broadcast FMAs instead
    of `tl.dot`, so BLOCK_M can be 1. Split-K spreads the K-blocks across
    SMs and combines with fp32 atomic adds.

    GROUP_UNIFORM=True (canonical monotone g_idx with a group size that is a
    multiple of 32) hoists the g_idx/scales/zeros loads out of the 32-row
    unroll: one scales and one zeros vector per block instead of 32 of each.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    PF0: tl.constexpr = 32 // W0
    PF1: tl.constexpr = 32 // W1
    PF2: tl.constexpr = 32 // (W2 if W2 > 0 else 1)

    NUM_BLOCKS: tl.constexpr = K // 32
    for blk in range(pid_k, NUM_BLOCKS, SPLIT_K):
        base = blk * BITS
        row0 = blk * 32

        if GROUP_UNIFORM:
            g_blk = tl.load(g_idx_ptr + row0)
            g_blk = tl.where(g_blk < 0, g_blk + num_groups, g_blk)
            g_blk = tl.cast(g_blk, tl.int32)
            scales_blk = tl.cast(
                tl.load(scales_ptr + g_blk * N + offs_n, mask=mask_n, eviction_policy="evict_last"),
                tl.float32,
            )
            zeros_blk = tl.load(
                zeros_dense_ptr + g_blk * N + offs_n, mask=mask_n, eviction_policy="evict_last"
            )

        for k in tl.static_range(32):
            if GROUP_UNIFORM:
                scales = scales_blk
                zeros = zeros_blk
            else:
                g = tl.load(g_idx_ptr + row0 + k)
                g = tl.where(g < 0, g + num_groups, g)
                g = tl.cast(g, tl.int32)

                scales = tl.cast(
                    tl.load(scales_ptr + g * N + offs_n, mask=mask_n, eviction_policy="evict_last"),
                    tl.float32,
                )
                zeros = tl.load(
                    zeros_dense_ptr + g * N + offs_n, mask=mask_n, eviction_policy="evict_last"
                )

            w0 = tl.load(
                qweight_ptr + (base + k // PF0) * N + offs_n,
                mask=mask_n,
                eviction_policy="evict_last",
            )
            codes = (w0 >> (W0 * (k % PF0))) & ((1 << W0) - 1)

            w1 = tl.load(
                qweight_ptr + (base + W0 + k // PF1) * N + offs_n,
                mask=mask_n,
                eviction_policy="evict_last",
            )
            codes |= ((w1 >> (W1 * (k % PF1))) & ((1 << W1) - 1)) << W0

            if W2 > 0:
                w2 = tl.load(
                    qweight_ptr + (base + W0 + W1 + k // PF2) * N + offs_n,
                    mask=mask_n,
                    eviction_policy="evict_last",
                )
                codes |= ((w2 >> (W2 * (k % PF2))) & ((1 << W2) - 1)) << (W0 + W1)

            w = tl.cast(codes - zeros, tl.float32) * scales

            xv = tl.cast(
                tl.load(x_ptr + offs_m * stride_xm + row0 + k, mask=mask_m, other=0.0),
                tl.float32,
            )
            acc += xv[:, None] * w[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=out_mask)


def _make_gemm_configs():
    # Small-M decode GEMM is DRAM-bound on the packed words; split-K spreads the
    # K-dimension reads across SMs so bandwidth is not limited by N//BLOCK_N
    # programs. Tile sizes stay small to keep shared memory well under the
    # 164KB/SM Ampere limit with multi-stage pipelining.
    # BLOCK_N is capped at 64: the decode gathers buffer a full
    # [BLOCK_K, BLOCK_N] int32 tile per plane per pipeline stage, and a 3-plane
    # (7-bit) 64x128 tile at num_stages=3 already needs ~232KB of shared memory
    # versus the ~164KB Ampere limit (measured: OutOfResources 231680/166912).
    configs = []
    for block_n, block_k, ws, ns, sk in itertools.product(
        [64], [64], [4, 8], [2, 3], [1, 4, 8]
    ):
        configs.append(
            triton.Config(
                {"BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": sk},
                num_warps=ws,
                num_stages=ns,
            )
        )
    return configs


# BLOCK_M is part of the key: a config tuned at BLOCK_M=16 can exceed shared
# memory at BLOCK_M=32, so each BLOCK_M must be tuned (and validated) separately.
@triton.autotune(_make_gemm_configs(), key=["N", "K", "BLOCK_M"], reset_to_zero=["out_ptr"])
@triton.jit
def planar_gemm_kernel(
    x_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    g_idx_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_om,
    num_groups,
    BITS: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Fused planar dequant + matmul: out[M, N] = x[M, K] @ dequant(qweight)[K, N].

    Packed words are read once per (K, N) tile and decoded in registers, so the
    weight-side DRAM traffic stays at `bits/8` bytes per element instead of the
    2-byte fp16 round-trip a separate dequant would add.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    ZEROS_WORDS_PER_ROW: tl.constexpr = (N // 32) * BITS

    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_kn = mask_k[:, None] & mask_n[None, :]

        g_idx = tl.load(g_idx_ptr + offs_k, mask=mask_k, other=0, eviction_policy="evict_last")
        groups = tl.where(g_idx < 0, g_idx + num_groups, g_idx)

        scales = tl.load(
            scales_ptr + groups[:, None] * N + offs_n[None, :],
            mask=mask_kn,
            other=0.0,
            eviction_policy="evict_last",
        )
        zeros = _planar_decode_cols(
            qzeros_ptr,
            groups[:, None],
            offs_n[None, :],
            ZEROS_WORDS_PER_ROW,
            mask_kn,
            BITS,
            W0,
            W1,
            W2,
        )
        codes = _planar_decode_rows(
            qweight_ptr,
            offs_k[:, None],
            offs_n[None, :],
            N,
            mask_kn,
            BITS,
            W0,
            W1,
            W2,
        )

        w = (tl.cast(codes, tl.float32) - tl.cast(zeros, tl.float32)) * tl.cast(scales, tl.float32)

        x_blk = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(x_blk, tl.cast(w, x_blk.dtype), out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=out_mask)


# Input rows at or below this threshold route to the fused GEMM; everything
# else (and the default of 0, i.e. disabled) uses planar_dequant + cuBLAS.
# On A100 the per-element word gathers keep the fused path behind the dequant
# + cuBLAS route even at M=1 (see pangolin_kernel.md benchmarks), so it stays
# opt-in until the decode is restructured around shared/register word reuse.
PLANAR_FUSED_MAX_M = int(os.environ.get("GPTQMODEL_PLANAR_FUSED_MAX_M", "0"))

# Input rows at or below this threshold route to the fused GEMV, which reads
# only the packed words on the weight side (no dense fp16 round-trip).
PLANAR_GEMV_MAX_M = int(os.environ.get("GPTQMODEL_PLANAR_GEMV_MAX_M", "0"))


def _zeros_dense(qzeros, num_groups: int, out_features: int, bits: int, w0: int, w1: int, w2: int):
    zeros_dense = torch.empty((num_groups, out_features), device=qzeros.device, dtype=torch.int32)
    znumels = zeros_dense.numel()
    zgrid = lambda meta: (triton.cdiv(znumels, meta["X_BLOCK"]),)  # noqa: E731
    planar_zeros_kernel[zgrid](
        qzeros,
        zeros_dense,
        znumels,
        out_features=out_features,
        BITS=bits,
        W0=w0,
        W1=w1,
        W2=w2,
    )
    return zeros_dense




def planar_gemv(x: torch.Tensor, qweight, scales, qzeros, g_idx, bits: int) -> torch.Tensor:
    """Fused decode-regime GEMV for a small-M 2D input `x[M, K]`; returns `[M, N]` fp32."""
    w0, w1, w2 = _plane_widths(bits)

    m, k = x.shape
    n = scales.shape[1]
    num_groups = scales.shape[0]

    if k % 32 != 0 or n % 32 != 0:
        raise ValueError(f"planar_gemv requires K and N divisible by 32, got K={k}, N={n}")

    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    block_m = triton.next_power_of_2(m)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])  # noqa: E731

    with torch.xpu.device(x.device) if HAS_XPU else torch.cuda.device(x.device):
        zeros_dense = _zeros_dense(qzeros, num_groups, n, bits, w0, w1, w2)
        planar_gemv_kernel[grid](
            x,
            qweight,
            zeros_dense,
            scales,
            g_idx,
            out,
            m,
            N=n,
            K=k,
            stride_xm=x.stride(0),
            stride_om=out.stride(0),
            num_groups=num_groups,
            BITS=bits,
            W0=w0,
            W1=w1,
            W2=w2,
            BLOCK_M=block_m,
            GROUP_UNIFORM=g_idx_block_uniform(g_idx),
        )
    return out.to(x.dtype)


def planar_matmul(x: torch.Tensor, qweight, scales, qzeros, g_idx, bits: int) -> torch.Tensor:
    """Fused planar dequant+matmul for a 2D input `x[M, K]`; returns `[M, N]`."""
    w0, w1, w2 = _plane_widths(bits)

    m, k = x.shape
    n = scales.shape[1]
    num_groups = scales.shape[0]

    if k % 32 != 0 or n % 32 != 0:
        raise ValueError(f"planar_matmul requires K and N divisible by 32, got K={k}, N={n}")

    if not x.is_contiguous():
        x = x.contiguous()

    # fp32 workspace: split-K partial sums accumulate with atomic adds.
    out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    block_m = min(max(16, triton.next_power_of_2(m)), 32)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )

    with torch.xpu.device(x.device) if HAS_XPU else torch.cuda.device(x.device):
        planar_gemm_kernel[grid](
            x,
            qweight,
            qzeros,
            scales,
            g_idx,
            out,
            m,
            N=n,
            K=k,
            stride_xm=x.stride(0),
            stride_om=out.stride(0),
            num_groups=num_groups,
            BITS=bits,
            W0=w0,
            W1=w1,
            W2=w2,
            BLOCK_M=block_m,
        )
    return out.to(x.dtype)


__all__ = [
    "PLANAR_FUSED_MAX_M",
    "PLANAR_GEMV_MAX_M",
    "PLANAR_TRITON_BITS",
    "planar_dequant",
    "planar_gemv",
    "planar_matmul",
]
