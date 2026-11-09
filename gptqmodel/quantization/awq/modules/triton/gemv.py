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

Measured on AMD Radeon AI PRO R9700 (gfx1201, wave32): ``M == 1``, FP16,
group 128, median of 41 iterations after 20 warm-ups, a 256 MiB L2 flush before
every timed call and ``torch.cuda.synchronize`` around every call.  Every column
below is a real module forward (``AwqGEMVTritonLinear`` vs the shipped
``AwqGEMMTritonLinear``), not a raw kernel launch, so the ~15 us of Python
launch overhead each path pays is included:

======  ======  =========  =========  =========  ========  =========
K       N       gemv us    gemm us    split us   vs GEMM   vs split
======  ======  =========  =========  =========  ========  =========
2048    2048    68.0       93.5       448.6      1.38x     6.6x
4096    4096    111.8      117.1      2134.0     1.05x     19.1x
5120    8192    170.2      165.1      5222.8     0.97x     30.7x
14336   4096    248.4      200.5      7286.3     0.81x     29.3x
4096    11008   186.0      175.6      5593.1     0.94x     30.1x
======  ======  =========  =========  =========  ========  =========

``split`` is this package's own dequantize-to-dense + ``torch.matmul`` from the
same packed buffers: the cost floor the fused kernel replaces, which is what the
29-31x headline measures.  The ``FORMAT.GEMM`` Triton GEMM at ``M == 1`` is the
nearest *shipped* comparison and is only at parity (0.81x-1.38x, i.e. the two
kernels are within noise of each other on large shapes); this GEMV kernel is a
correctness/portability path for ROCm, not a decode speedup over ``FORMAT.GEMM``.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from gptqmodel.utils.arch import tuning_family


class AwqGemvLaunch(NamedTuple):
    """Resolved launch constants for :func:`awq_gemv_triton` on one arch class."""

    block_n: int
    max_block_k: int
    num_warps: int
    num_stages: int


# ``block_k`` is capped at ``max_block_k`` and must divide the group size:
# the kernel loads one scale/zero group per K tile and broadcasts it across
# the tile, so a tile may never span more than one group.
_AWQ_GEMV_LAUNCH_CONFIGS: dict[str, AwqGemvLaunch] = {
    # Measured on gfx1201 across K,N in {(2048,2048), (4096,4096), (5120,8192),
    # (14336,4096), (4096,11008), (896,128)}.  block_n=64/block_k=512 was within
    # 1% of the per-shape winner everywhere except (2048,2048), where block_n=32
    # was 0.5% faster, and (5120,8192), where block_n=128 was 0.4% faster.
    "rdna4": AwqGemvLaunch(block_n=64, max_block_k=512, num_warps=4, num_stages=1),
    # Measured on an NVIDIA GeForce RTX 4090 (SM 8.9, torch 2.15.0+cu132,
    # Triton 3.9.0), FORMAT.GEMV packing, group_size 128 (so ``block_k`` clamps
    # to 128 and the max_block_k axis is a no-op for this group size), FP16,
    # M == 1: median of 41 iterations after 15 warm-ups, a 256 MiB L2 flush
    # before every timed call and ``torch.cuda.synchronize`` around each call.
    # Swept block_n {32,64,128} x max_block_k {256,512,1024} x num_warps
    # {2,4,8} x num_stages {1,2,3} on K,N in {(2048,2048), (4096,4096),
    # (5120,8192), (14336,4096), (4096,11008), (896,128)}.
    #
    # Findings: the portable 4-warp/block_n=64 tile is warp-starved on Ada -- a
    # 32-wide N tile with eight warps is 1.02x-1.40x faster hot and
    # 1.00x-1.14x faster L2-flushed on the five real projections
    # (2048x2048 1.10x/1.12x, 4096x4096 1.19x/1.08x, 5120x8192 1.05x/1.06x,
    # 14336x4096 1.40x/1.14x, 4096x11008 1.02x/1.00x).  The one 896x128 case is
    # a launch-bound ~2% loss (0.981x hot / 0.987x cold): N=128 is only four
    # 32-wide blocks, so the tile choice is inside the launch-cost noise there
    # and the shipped tile is kept uniform rather than special-cased.
    # block_n=128 always lost.  num_stages and max_block_k only move <=1.5%
    # once block_k is clamped to the group, so the entry keeps the shallowest
    # pipeline.  Measured with 120 interleaved hot/flushed iterations after 20
    # warm-ups on top of two independent sweeps.
    "ada": AwqGemvLaunch(block_n=32, max_block_k=512, num_warps=8, num_stages=1),
}

# Portable fallback: the same measured tile.  Triton's own default of four warps
# and the 512-deep K tile are not CUDA-specific, and no non-ROCm sweep was run
# here, so this is deliberately not presented as a CUDA tuning result.
_DEFAULT_AWQ_GEMV_LAUNCH = AwqGemvLaunch(block_n=64, max_block_k=512, num_warps=4, num_stages=1)


def launch_config(device_or_arch=None) -> AwqGemvLaunch:
    """Return the AWQ GEMV launch constants for an architecture string or device.

    The table is keyed by :func:`gptqmodel.utils.arch.tuning_family`, the same
    vocabulary every other tuned Triton table uses, so a CUDA capability tuple
    or an ``sm_89`` string resolves to the Ada entry instead of falling through
    to the portable default.  ``None``/empty input asks the current device and
    on a GPU host therefore resolves to that device's family; an unregistered
    family still keeps :data:`_DEFAULT_AWQ_GEMV_LAUNCH`.
    """

    if isinstance(device_or_arch, str) and not device_or_arch.strip():
        return _DEFAULT_AWQ_GEMV_LAUNCH
    return _AWQ_GEMV_LAUNCH_CONFIGS.get(tuning_family(device_or_arch), _DEFAULT_AWQ_GEMV_LAUNCH)


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
    if group_size < 8 or group_size % 8:
        raise ValueError("AWQ GEMV Triton group_size must be a positive multiple of 8")
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
    # A non-power-of-two group still has a power-of-two factor that aligns
    # every tile with a scale/zero boundary (e.g. group 96 -> tile 32).
    block_k = min(config.max_block_k, group_size & -group_size)
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
