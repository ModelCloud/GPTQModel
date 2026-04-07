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

from gptqmodel.utils.env import env_flag


PAROQUANT_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
# Shared runtime default: fp32 accumulation trades a little speed for lower numerical drift.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)


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
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
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

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
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
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
    num_warps: int,
    num_stages: int,
    fp32_accum: bool = FP32_ACCUM,
) -> torch.Tensor:
    M, N, K, group_size = _validate_shapes(input, qweight, scales, qzeros)

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
            M,
            N,
            K,
            group_size,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            USE_FP32_ACCUM=fp32_accum,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return result


def paroquant_gemm_triton_decode(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    return _paroquant_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        block_size_m=4,
        block_size_n=128,
        block_size_k=32,
        num_warps=4,
        num_stages=2,
        fp32_accum=FP32_ACCUM,
    )


def paroquant_gemm_triton_prefill(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    return _paroquant_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        block_size_m=32,
        block_size_n=128,
        block_size_k=32,
        num_warps=8,
        num_stages=4,
        fp32_accum=FP32_ACCUM,
    )


__all__ = [
    "paroquant_dequantize_triton",
    "paroquant_gemm_triton_decode",
    "paroquant_gemm_triton_prefill",
]
