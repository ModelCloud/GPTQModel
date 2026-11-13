# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Copied/adapted from the AWQ Triton kernels used in vLLM and GPT-QModel.
#
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gptqmodel.utils.arch import tuning_family
from gptqmodel.utils.env import env_flag


PAROQUANT_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
# Shared runtime default: fp32 accumulation trades a little speed for lower numerical drift.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)


# Per-tuning-family launch tiles for the fused ParoQuant GEMM.
#
# Before this table the kernel hardcoded one pair for every GPU: decode
# (4,128,32,warps=4,stages=2) and prefill (32,128,32,warps=8,stages=4).  Those
# stay the fallback for families nobody measured here (Hopper, Blackwell,
# cdna4 and rdna3) so no unmeasured target is silently retuned.
#
# ``ada`` (RTX 4090, SM 8.9) has its own measured entry below.
#
# ``rdna4`` was measured on an AMD Radeon AI PRO R9700 (gfx1201, wave32,
# HIP 7.16.26385, torch 2.16.0.dev20261025+rocm10.1): 4-bit ParoQuant,
# group_size 128, FP16 activations, median of 21 iterations after 5 warm-up
# iterations, a 256 MiB L2 flush before every timed call and
# ``torch.cuda.synchronize`` around each call (scripts/benchmark_rdna4_inference.py
# methodology).  Swept BLOCK_M {1,2,4,8,16,32,64}, BLOCK_N {64,128},
# BLOCK_K {32,64}, num_warps {2,4,8}, num_stages {1,2,4} on
# (K,N) in {(2048,2048), (4096,4096), (5120,8192), (14336,4096), (4096,11008)}.
#
# Findings (all reproducible with the shipped tiles):
# * ``num_warps=8`` never won a decode shape and lost the prefill band badly
#   (K=4096/N=4096/M=16: 83.2 us at warps=4 vs 137.6 us at warps=8).  On wave32
#   four warps already cover 128 lanes, so eight mainly add scheduling overhead.
# * ``BLOCK_N=64`` beat 128 up to ~64 rows; above that 128 amortises better.
# * The M band matters more than the decode/prefill label: rows <= 8 want a
#   narrow BLOCK_M and no pipelining, 9..64 want BLOCK_M 32, and > 64 rows want
#   BLOCK_M 64 / BLOCK_N 128.
# Representative medians (small / medium / large vs the old tile):
#   M=1   K=14336 N=4096  181.30 vs 269.13 us  (1.48x)
#   M=4   K=4096  N=4096   77.72 vs 102.18 us  (1.31x)
#   M=16  K=14336 N=4096  206.87 vs 288.76 us  (1.40x)
#   M=64  K=14336 N=4096  222.37 vs 325.10 us  (1.46x)
#   M=128 K=5120  N=8192  186.06 vs 296.69 us  (1.59x)
#   M=1024 K=14336 N=4096 1680.99 vs 2893.65 us (1.72x)
_PAROQUANT_GEMM_LAUNCH_CONFIGS: dict[str, dict[str, tuple[int, int, int, int, int]]] = {
    # ``ada`` was measured on an NVIDIA GeForce RTX 4090 (SM 8.9, torch
    # 2.15.0+cu132, Triton 3.9.0): 4-bit ParoQuant, group_size 128, FP16
    # activations, median of 41 iterations after 15 warm-up iterations, a 256 MiB
    # L2 flush before every timed call and ``torch.cuda.synchronize`` around each
    # call (scripts/benchmark_rdna4_inference.py methodology).  Swept the same
    # grid as ``rdna4``: BLOCK_M {1,2,4,8,16,32,64}, BLOCK_N {64,128},
    # BLOCK_K {32,64}, num_warps {2,4,8}, num_stages {1,2,4} (252 tiles) on
    # (K,N) in {(2048,2048), (4096,4096), (5120,8192), (14336,4096),
    # (4096,11008)} at M in {1,2,4,8} / {16,32,64} / {65,128,1024}.
    #
    # Findings: ``num_stages=4`` wins the small-M bands (Triton pipelines the
    # K-loop across Ada's cp.async stages) while the >64-row band prefers four
    # warps with a 64-wide N tile; ``BLOCK_N=128`` only wins at the 1024-row tail
    # of the largest (K=14336/5120/4096-wide) shapes and loses badly below it,
    # so the band keeps 64 for a single uniform tile.  The historical default
    # (decode (4,128,32,4,2), prefill (32,128,32,8,4)) is 1.14x-3.29x slower
    # across the swept matrix (hot; 1.11x-3.29x L2-flushed); these tiles are its
    # per-band measured winner.  Worst reference-side oracle error over the same
    # sweep is 4.7e-4 relative against an FP64 oracle.
    "ada": {
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        "decode": (16, 64, 64, 4, 4),
        "prefill": (32, 64, 64, 8, 4),
        "prefill_large": (64, 64, 64, 8, 4),
    },
    "rdna4": {
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        "decode": (4, 64, 64, 4, 1),
        "prefill": (32, 64, 64, 4, 2),
        "prefill_large": (64, 128, 64, 4, 2),
    },
}
# Historical tile shipped by every pre-table release: (M, N, K, warps, stages).
# Kept verbatim so unmeasured families, and callers that do not pass a row
# count, behave exactly as before.
_PAROQUANT_GEMM_DEFAULT_CONFIGS: dict[str, tuple[int, int, int, int, int]] = {
    "decode": (4, 128, 32, 4, 2),
    "prefill": (32, 128, 32, 8, 4),
}

# Row band at which a banded family (``ada``/``rdna4``) switches from the medium
# to the large tile.  The small/medium split is the module's own decode
# boundary (8 rows), so the constant is shared and its meaning is unchanged for
# non-Ada families.
_RDNA4_MEDIUM_MAX_ROWS = 64


def paroquant_gemm_launch_config(
    kind: str, device_or_arch=None, rows: int | None = None
) -> tuple[int, int, int, int, int]:
    """Return ``(block_m, block_n, block_k, num_warps, num_stages)``.

    ``kind`` is ``"decode"`` (the module's M<=8 fused path) or ``"prefill"``.
    ``rows`` is the flattened row count of the current input; when it is given,
    families with a row-banded table (``ada`` and ``rdna4``) pick the tile from
    the measured band instead of from the decode/prefill label.  Families
    without a measured entry always keep the historical tile.
    """

    family = tuning_family(device_or_arch)
    table = _PAROQUANT_GEMM_LAUNCH_CONFIGS.get(family)
    if table is None:
        return _PAROQUANT_GEMM_DEFAULT_CONFIGS[kind]
    if "prefill_large" not in table:
        return table[kind]
    if rows is not None:
        if rows <= 8:
            return table["decode"]
        if rows <= _RDNA4_MEDIUM_MAX_ROWS:
            return table["prefill"]
        return table["prefill_large"]
    # Without a row count keep the label's tile; the label itself already means
    # "decode rows" vs "prefill rows".
    return table[kind]


def get_same_device_cm(t):
    if t.device.type == "xpu":
        return torch.xpu.device(t.device.index)
    return torch.cuda.device(t.device.index)


@triton.jit
def paroquant_dequantize_kernel(
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    group_size,
    result_ptr,
    num_cols,
    num_rows,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols
    masks = masks_y[:, None] & masks_x[None, :]

    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = 8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    iweights = tl.load(qweight_ptr + offsets, masks)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
    iweights = (iweights >> shifts) & 0xF

    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
    zeros = (zeros >> shifts) & 0xF

    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    scale_offsets = num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    scales = tl.load(scales_ptr + scale_offsets, scale_masks)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)
    tl.store(result_ptr + result_offsets, iweights, result_masks)


@triton.jit
def paroquant_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    bias_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if USE_FP32_ACCUM:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.type.element_ty)

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    masks_am = offsets_am < M
    masks_bn = offsets_bn < N // 8
    masks_zn = offsets_zn < N // 8
    masks_sn = offsets_sn < N

    a_ptrs = a_ptr + K * offsets_am[:, None] + offsets_k[None, :]
    b_ptrs = b_ptr + (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        offsets_szk = k * BLOCK_SIZE_K // group_size + tl.arange(0, 1)
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros = tl.load(zeros_ptr + offsets_z, mask=masks_z, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_s = masks_zk[:, None] & masks_sn[None, :]
        scales = tl.load(scales_ptr + offsets_s, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = ((b - zeros) * scales).to(a.dtype)

        if USE_FP32_ACCUM:
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        else:
            accumulator = tl.dot(a, b, accumulator, out_dtype=c_ptr.type.element_ty)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * (N // 8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0).to(tl.float32)
        c = (accumulator.to(tl.float32) + bias[None, :]).to(c_ptr.type.element_ty)
    else:
        c = accumulator.to(c_ptr.type.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def _validate_shapes(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
):
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert group_size <= K
    assert group_size in PAROQUANT_TRITON_SUPPORTED_GROUP_SIZES or group_size == K
    return M, N, K, group_size


def paroquant_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    K = qweight.shape[0]
    N = scales.shape[1]
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert K > 0 and N > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert group_size <= K
    assert group_size in PAROQUANT_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # The dequant kernel uses dense offsets for all packed operands.
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()

    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    y = qweight.shape[0]
    x = qweight.shape[1]

    def grid(meta):
        return (
            triton.cdiv(x, meta["BLOCK_SIZE_X"]),
            triton.cdiv(y, meta["BLOCK_SIZE_Y"]),
        )

    with get_same_device_cm(qweight):
        paroquant_dequantize_kernel[grid](
            qweight,
            scales,
            qzeros,
            group_size,
            result,
            x,
            y,
            BLOCK_SIZE_X=block_size_x,
            BLOCK_SIZE_Y=block_size_y,
        )

    return result


def _paroquant_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    num_warps: int,
    num_stages: int,
    fp32_accum: bool = FP32_ACCUM,
) -> torch.Tensor:
    M, N, K, group_size = _validate_shapes(input, qweight, scales, qzeros)
    # The fused GEMM uses dense row and column offsets for every operand.
    input = input.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    if bias is not None:
        if bias.shape != (N,) or bias.device != input.device or not bias.is_floating_point():
            raise ValueError("ParoQuant Triton bias must be a floating [N] tensor on the input device.")
        bias = bias.contiguous()

    # ``paroquant_gemm_kernel`` loads one scale/zero group per K tile and
    # broadcasts it across the whole tile, so a tile may never span more than
    # one group: ``BLOCK_SIZE_K <= group_size``.  Supported group sizes are
    # powers of two, so clamping keeps the tile group-aligned.
    block_size_k = min(block_size_k, group_size)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    result = torch.empty((M, N), dtype=input.dtype, device=input.device)

    with get_same_device_cm(qweight):
        paroquant_gemm_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            bias if bias is not None else scales,
            M,
            N,
            K,
            group_size,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            USE_FP32_ACCUM=fp32_accum or bias is not None,
            HAS_BIAS=bias is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return result


def paroquant_gemm_triton_decode(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    block_size_m, block_size_n, block_size_k, num_warps, num_stages = (
        paroquant_gemm_launch_config("decode", input.device, rows=input.shape[0])
    )
    return _paroquant_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        bias=bias,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
        fp32_accum=FP32_ACCUM,
    )


def paroquant_gemm_triton_prefill(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    block_size_m, block_size_n, block_size_k, num_warps, num_stages = (
        paroquant_gemm_launch_config("prefill", input.device, rows=input.shape[0])
    )
    return _paroquant_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        bias=bias,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        block_size_k=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
        fp32_accum=FP32_ACCUM,
    )


__all__ = [
    "paroquant_dequantize_triton",
    "paroquant_gemm_launch_config",
    "paroquant_gemm_triton_decode",
    "paroquant_gemm_triton_prefill",
]
