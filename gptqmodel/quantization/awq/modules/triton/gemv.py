# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Portable Triton GEMV for the AWQ ``FORMAT.GEMV`` / ``GEMV_FAST`` layout.

The native ``gptqmodel_ext/awq`` GEMV kernels are CUDA-only: their sources
include ``cuda_fp16.h`` and CUDA inline asm, and the hipified warp reduction
requires a 64-bit sync mask, which wave32 RDNA4 rejects.  This module provides
the same computation with plain Triton so a decode-shaped (``M <= 8``) AWQ GEMV
checkpoint can run on ROCm (and on CUDA Triton) without the extension.

Layout
------
The GEMV packing (`AwqGEMVLinear.pack` -> `pack_awq_columns`) is **not** the
``FORMAT.GEMM`` layout:

* ``qweight`` is ``[N, K/8]`` int32.  Nibble ``i`` of word ``w`` holds input
  column ``w*8 + i`` -- plain little-nibble order, no AWQ ``[0,2,4,6,1,3,5,7]``
  interleave.  (``FORMAT.GEMM`` is ``[K, N/8]`` with that interleave.)
* ``qzeros`` is ``[N, ceil(groups/8)]`` packed the same way, and its group axis
  is padded up to a multiple of 8, so it needs an explicit stride.
* ``scales`` is ``[N, stride]`` FP16 with the same padded group axis and the
  real group count in ``[0, groups)``.

Numerics follow the rest of the repository: weights stay packed and are read
once, the dot product accumulates in FP32 over bounded ``BLOCK_K`` tiles, and
the result is stored in the input dtype.

Measured on AMD Radeon AI PRO R9700 (gfx1201, wave32) with
``scripts/benchmark_rdna4_inference.py`` methodology (median of 21 iterations
after 5 warm-ups, 256 MiB L2 flush before each timed call,
``torch.cuda.synchronize`` around every call), ``M == 1``, FP16, group 128:

======  ======  ==========  =========  ========
K       N       split us    gemv us    gain
======  ======  ==========  =========  ========
2048    2048    679.8       48.3       14.1x
4096    4096    3164.1      93.2       33.9x
5120    8192    7730.3      152.6      50.7x
14336   4096    10771.2     229.3      47.0x
4096    11008   8284.2      164.5      50.4x
======  ======  ==========  =========  ========

``split`` is ``torch.matmul`` against a dense weight materialized from the same
packed buffers -- the cost floor the fused kernel has to beat, not a shipped
kernel.  The ``FORMAT.GEMM`` Triton GEMM at ``M == 1`` is the nearest shipped
comparison and is 1.7x-2.4x slower than this kernel at the same shapes.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from gptqmodel.utils.arch import arch_class, normalize_arch


class AwqGemvLaunch(NamedTuple):
    """Resolved launch constants for :func:`awq_gemv_triton` on one arch class."""

    block_n: int
    max_block_k: int
    num_warps: int
    num_stages: int


# ``block_k`` is capped at ``max_block_k`` and additionally clamped to the group
# size: the kernel loads one scale/zero group per K tile and broadcasts it across
# the tile, so a tile may never span more than one group.
_AWQ_GEMV_LAUNCH_CONFIGS: dict[str, AwqGemvLaunch] = {
    # Measured on gfx1201 across K,N in {(2048,2048), (4096,4096), (5120,8192),
    # (14336,4096), (4096,11008), (896,128)}.  block_n=64/block_k=512 was within
    # 1% of the per-shape winner everywhere except (2048,2048), where block_n=32
    # was 0.5% faster, and (5120,8192), where block_n=128 was 0.4% faster.
    "rdna4": AwqGemvLaunch(block_n=64, max_block_k=512, num_warps=4, num_stages=1),
}

# Portable fallback: the same measured tile.  Triton's own default of four warps
# and the 512-deep K tile are not CUDA-specific, and no non-ROCm sweep was run
# here, so this is deliberately not presented as a CUDA tuning result.
_DEFAULT_AWQ_GEMV_LAUNCH = AwqGemvLaunch(block_n=64, max_block_k=512, num_warps=4, num_stages=1)


def launch_config(arch=None) -> AwqGemvLaunch:
    """Return the AWQ GEMV launch constants for an architecture string or device."""

    if isinstance(arch, str):
        name = normalize_arch(arch)
        if not name.startswith("gfx"):
            return _DEFAULT_AWQ_GEMV_LAUNCH
        arch = name
    return _AWQ_GEMV_LAUNCH_CONFIGS.get(arch_class(arch), _DEFAULT_AWQ_GEMV_LAUNCH)


@triton.jit
def _awq_gemv_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    out_ptr,
    M,
    N,
    K,
    group_size,
    stride_z,
    stride_s,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pack_factor: tl.constexpr = 8
    words_k = K // pack_factor
    num_words: tl.constexpr = BLOCK_K // pack_factor

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    offs_words = tl.arange(0, num_words)
    nibbles = tl.arange(0, pack_factor)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        base_k = k_tile * BLOCK_K
        group = base_k // group_size

        # Load one nibble-expanded slice: [BLOCK_N, BLOCK_K].
        words = tl.load(
            qweight_ptr
            + cols[:, None, None] * words_k
            + (base_k // pack_factor + offs_words)[None, :, None],
            mask=col_mask[:, None, None],
            other=0,
        )
        codes = (words >> (nibbles[None, None, :] * 4)) & 0xF
        codes = tl.reshape(codes, (BLOCK_N, BLOCK_K))

        scale = tl.load(scales_ptr + cols * stride_s + group, mask=col_mask, other=0.0)
        zero_word = tl.load(
            zeros_ptr + cols * stride_z + group // pack_factor, mask=col_mask, other=0,
        )
        zero = (zero_word >> ((group % pack_factor) * 4)) & 0xF

        rows = base_k + tl.arange(0, BLOCK_K)
        activation = tl.load(
            x_ptr + pid_m * K + rows, mask=rows < K, other=0.0,
        ).to(tl.float32)

        weights = (codes.to(tl.float32) - zero.to(tl.float32)[:, None]) * scale.to(tl.float32)[:, None]
        accumulator += tl.sum(weights * activation[None, :], axis=1)

    tl.store(out_ptr + pid_m * N + cols, accumulator.to(out_ptr.type.element_ty), mask=col_mask)


def awq_gemv_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused AWQ GEMV for decode-shaped ``input`` (``M <= 8``).

    ``qweight`` is the ``[N, K/8]`` GEMV packing, ``qzeros`` the ``[N, groups/8]``
    padded packing, and ``scales`` the ``[N, padded_groups]`` FP16 table.  Weights
    are read from their packed representation; nothing dense is materialized.
    """

    if input.ndim != 2 or qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ GEMV Triton tensors must be rank-2")
    M, K = input.shape
    N = qweight.shape[0]
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("AWQ GEMV Triton requires positive M, N and K")
    if qweight.shape[1] * 8 != K:
        raise ValueError("AWQ GEMV Triton qweight must be [N, K/8]")
    if K % group_size:
        raise ValueError("AWQ GEMV Triton group_size must divide K")
    groups = K // group_size
    if qzeros.shape[0] != N or scales.shape[0] != N:
        raise ValueError("AWQ GEMV Triton qzeros/scales must be [N, *]")
    if qzeros.shape[1] * 8 < groups or scales.shape[1] < groups:
        raise ValueError("AWQ GEMV Triton qzeros/scales group axis is too short")
    if qzeros.shape[1] * 8 < groups:
        raise ValueError("AWQ GEMV Triton qzeros group axis is too short")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("AWQ GEMV Triton requires int32 packed weights and zeros")
    if scales.dtype != torch.float16:
        raise ValueError("AWQ GEMV Triton requires float16 scales")
    if any(t.device != input.device for t in (qweight, qzeros, scales)):
        raise ValueError("AWQ GEMV Triton tensors must be on the same device")
    if not all(t.is_contiguous() for t in (input, qweight, qzeros, scales)):
        raise ValueError("AWQ GEMV Triton tensors must be contiguous")

    config = launch_config(input.device)
    block_n = config.block_n
    block_k = min(config.max_block_k, group_size)
    out_dtype = input.dtype if output_dtype is None else output_dtype
    output = torch.empty((M, N), device=input.device, dtype=out_dtype)
    with torch.cuda.device(input.device):
        _awq_gemv_kernel[(triton.cdiv(N, block_n), M)](
            input,
            qweight,
            scales,
            qzeros,
            output,
            M,
            N,
            K,
            group_size,
            qzeros.stride(0),
            scales.stride(0),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return output


__all__ = ["awq_gemv_triton", "launch_config", "AwqGemvLaunch"]
