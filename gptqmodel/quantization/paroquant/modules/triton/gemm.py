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

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from gptqmodel.utils.env import env_flag

try:
    from triton.runtime import driver as triton_driver
except ImportError:  # pragma: no cover - retained for older supported Triton releases
    triton_driver = None


PAROQUANT_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
PAROQUANT_MEGAKERNEL_GROUP_SIZE = 128
PAROQUANT_SPLITK_PAIR_PREFETCH_SECOND_WEIGHT = False
PAROQUANT_SPLITK_PAIR_ATOMIC_RESET = False
# Shared runtime default: fp32 accumulation trades a little speed for lower numerical drift.
FP32_ACCUM = env_flag("GPTQMODEL_FP32_ACCUM", default=True)


def get_same_device_cm(t):
    if t.device.type == "xpu":
        return torch.xpu.device(t.device.index)
    return torch.cuda.device(t.device.index)


def _paroquant_triton_current_stream(device: torch.device) -> int:
    """Return Triton's raw current-stream handle, with a public PyTorch fallback."""
    if triton_driver is not None:
        try:
            return triton_driver.active.get_current_stream(device.index)
        except (AttributeError, RuntimeError, TypeError):
            pass
    return torch.cuda.current_stream(device).cuda_stream


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


@triton.jit
def paroquant_rotation_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    partner_ptr,
    cos_ptr,
    sin_ptr,
    channel_scales_ptr,
    bias_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    KROT: tl.constexpr,
    PARTNER_IS_LOCAL: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    EXPLICIT_FMA: tl.constexpr,
    PREFETCH_FIRST_PARTNER: tl.constexpr,
    PREFETCH_PACKED_WEIGHT: tl.constexpr,
    USE_FP32_ACCUM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """Fuse ParoQuant rotation, int4 dequantization, GEMM, and bias."""
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

    for k_block in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), loop_unroll_factor=LOOP_UNROLL_FACTOR):
        global_k = k_block * BLOCK_SIZE_K + offsets_k
        masks_k = global_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)
        if PREFETCH_FIRST_PARTNER:
            first_partner = tl.load(partner_ptr + global_k, mask=masks_k, other=offsets_k)
        channel_scales = tl.load(channel_scales_ptr + global_k, mask=masks_k, other=1.0)
        if INPUT_IS_BF16:
            a = (a.to(tl.float16) * channel_scales[None, :].to(tl.float16)).to(tl.float16)
        else:
            a = (a * channel_scales[None, :]).to(a_ptr.type.element_ty)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        if PREFETCH_PACKED_WEIGHT:
            b = tl.load(b_ptrs, mask=masks_b, other=0)

        for rot_idx in range(0, KROT):
            lookup_offsets = rot_idx * K + global_k
            if PREFETCH_FIRST_PARTNER and rot_idx == 0:
                partner = first_partner
            else:
                partner = tl.load(partner_ptr + lookup_offsets, mask=masks_k, other=offsets_k)
            if not PARTNER_IS_LOCAL:
                partner = partner - k_block * BLOCK_SIZE_K
            partner = tl.broadcast_to(partner[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_K))
            paired = tl.gather(a, partner, axis=1)
            cos_value = tl.load(cos_ptr + lookup_offsets, mask=masks_k, other=1.0)
            sin_value = tl.load(sin_ptr + lookup_offsets, mask=masks_k, other=0.0)
            if EXPLICIT_FMA:
                rotated = tl.fma(a, cos_value[None, :], paired * sin_value[None, :])
            else:
                rotated = a * cos_value[None, :] + paired * sin_value[None, :]
            if INPUT_IS_BF16:
                a = rotated.to(tl.float16)
            else:
                a = rotated.to(a_ptr.type.element_ty)

        if INPUT_IS_BF16:
            a_dot = a.to(tl.bfloat16)
        else:
            a_dot = a

        if not PREFETCH_PACKED_WEIGHT:
            b = tl.load(b_ptrs, mask=masks_b, other=0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        offsets_z = (N // 8) * k_block + offsets_zn
        zeros = tl.load(zeros_ptr + offsets_z, mask=masks_zn, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * k_block + offsets_sn
        weight_scales = tl.load(scales_ptr + offsets_s, mask=masks_sn, other=0.0)
        weight_scales = tl.broadcast_to(weight_scales[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = ((b - zeros) * weight_scales).to(a_dot.dtype)

        if USE_FP32_ACCUM:
            accumulator = tl.dot(a_dot, b, accumulator, out_dtype=tl.float32)
        else:
            accumulator = tl.dot(a_dot, b, accumulator, out_dtype=c_ptr.type.element_ty)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offsets_sn, mask=masks_sn, other=0.0)
        c += bias[None, :].to(c_ptr.type.element_ty)

    c_ptrs = c_ptr + N * offsets_am[:, None] + offsets_sn[None, :]
    c_mask = masks_am[:, None] & masks_sn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def paroquant_rotation_gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    partner_ptr,
    cos_ptr,
    sin_ptr,
    channel_scales_ptr,
    bias_ptr,
    partials_ptr,
    counters_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    KROT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_BLOCKS_PER_SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """Run one-launch split-K decode with an atomic last-CTA reduction."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = pid // SPLIT_K
    split_id = pid % SPLIT_K
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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

    first_k_block = split_id * K_BLOCKS_PER_SPLIT
    a_ptrs = a_ptr + K * offsets_am[:, None] + first_k_block * BLOCK_SIZE_K + offsets_k[None, :]
    b_ptrs = b_ptr + (N // 8) * (first_k_block * BLOCK_SIZE_K + offsets_k[:, None]) + offsets_bn[None, :]

    for local_k_block in range(0, K_BLOCKS_PER_SPLIT):
        k_block = first_k_block + local_k_block
        global_k = k_block * BLOCK_SIZE_K + offsets_k
        masks_k = global_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)
        first_partner = tl.load(partner_ptr + global_k, mask=masks_k, other=offsets_k)
        channel_scales = tl.load(channel_scales_ptr + global_k, mask=masks_k, other=1.0)
        a = (a.to(tl.float16) * channel_scales[None, :].to(tl.float16)).to(tl.float16)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0)
        for rot_idx in range(0, KROT):
            lookup_offsets = rot_idx * K + global_k
            if rot_idx == 0:
                partner = first_partner
            else:
                partner = tl.load(partner_ptr + lookup_offsets, mask=masks_k, other=offsets_k)
            partner = tl.broadcast_to(partner[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_K))
            paired = tl.gather(a, partner, axis=1)
            cos_value = tl.load(cos_ptr + lookup_offsets, mask=masks_k, other=1.0)
            sin_value = tl.load(sin_ptr + lookup_offsets, mask=masks_k, other=0.0)
            a = tl.fma(a, cos_value[None, :], paired * sin_value[None, :]).to(tl.float16)

        if INPUT_IS_BF16:
            a_dot = a.to(tl.bfloat16)
        else:
            a_dot = a
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        offsets_z = (N // 8) * k_block + offsets_zn
        zeros = tl.load(zeros_ptr + offsets_z, mask=masks_zn, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * k_block + offsets_sn
        weight_scales = tl.load(scales_ptr + offsets_s, mask=masks_sn, other=0.0)
        weight_scales = tl.broadcast_to(weight_scales[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        if INPUT_IS_BF16:
            b = ((b - zeros) * weight_scales).to(tl.bfloat16)
        else:
            b = ((b - zeros) * weight_scales).to(tl.float16)
        accumulator = tl.dot(a_dot, b, accumulator, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * (N // 8)

    partial_offsets = (
        (tile_id * SPLIT_K + split_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_N)[None, :]
    )
    output_mask = masks_am[:, None] & masks_sn[None, :]
    tl.store(partials_ptr + partial_offsets, accumulator, mask=output_mask)

    prior_count = tl.atomic_add(counters_ptr + tile_id, 1, sem="acq_rel", scope="gpu")
    if prior_count == SPLIT_K - 1:
        reduced = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for partial_id in range(0, SPLIT_K):
            reduction_offsets = (
                (tile_id * SPLIT_K + partial_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N)[None, :]
            )
            reduced += tl.load(partials_ptr + reduction_offsets, mask=output_mask, other=0.0)
        c = reduced.to(c_ptr.type.element_ty)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offsets_sn, mask=masks_sn, other=0.0)
            c += bias[None, :].to(c_ptr.type.element_ty)
        c_ptrs = c_ptr + N * offsets_am[:, None] + offsets_sn[None, :]
        tl.store(c_ptrs, c, mask=output_mask)
        tl.atomic_xchg(counters_ptr + tile_id, 0, sem="release", scope="gpu")


@triton.jit
def paroquant_rotation_gemm_splitk_two_n_tiles_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    partner_ptr,
    cos_ptr,
    sin_ptr,
    channel_scales_ptr,
    bias_ptr,
    partials_ptr,
    counters_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    KROT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_BLOCKS_PER_SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """Process two output tiles per CTA while reusing one rotated activation tile.

    The host selector restricts this schedule to one K block per split.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_n_groups = tl.cdiv(num_pid_n, 2)
    cta_tile_id = pid // SPLIT_K
    split_id = pid % SPLIT_K
    pid_m = cta_tile_id // num_pid_n_groups
    pid_n_group = cta_tile_id % num_pid_n_groups

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    masks_am = offsets_am < M
    first_k_block = split_id * K_BLOCKS_PER_SPLIT
    global_k = first_k_block * BLOCK_SIZE_K + offsets_k
    masks_k = global_k < K
    masks_a = masks_am[:, None] & masks_k[None, :]
    a_ptrs = a_ptr + K * offsets_am[:, None] + global_k[None, :]

    first_pid_n = pid_n_group * 2
    first_offsets_bn = first_pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    first_masks_bn = first_offsets_bn < N // 8
    first_b_ptrs = b_ptr + (N // 8) * global_k[:, None] + first_offsets_bn[None, :]
    second_offsets_bn = (first_pid_n + 1) * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    second_masks_bn = second_offsets_bn < N // 8
    second_b_ptrs = b_ptr + (N // 8) * global_k[:, None] + second_offsets_bn[None, :]

    a = tl.load(a_ptrs, mask=masks_a, other=0.0)
    first_partner = tl.load(partner_ptr + global_k, mask=masks_k, other=offsets_k)
    channel_scales = tl.load(channel_scales_ptr + global_k, mask=masks_k, other=1.0)
    a = (a.to(tl.float16) * channel_scales[None, :].to(tl.float16)).to(tl.float16)
    first_b = tl.load(first_b_ptrs, mask=masks_k[:, None] & first_masks_bn[None, :], other=0)
    second_b = tl.load(second_b_ptrs, mask=masks_k[:, None] & second_masks_bn[None, :], other=0)

    for rot_idx in range(0, KROT):
        lookup_offsets = rot_idx * K + global_k
        if rot_idx == 0:
            partner = first_partner
        else:
            partner = tl.load(partner_ptr + lookup_offsets, mask=masks_k, other=offsets_k)
        partner = tl.broadcast_to(partner[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        paired = tl.gather(a, partner, axis=1)
        cos_value = tl.load(cos_ptr + lookup_offsets, mask=masks_k, other=1.0)
        sin_value = tl.load(sin_ptr + lookup_offsets, mask=masks_k, other=0.0)
        a = tl.fma(a, cos_value[None, :], paired * sin_value[None, :]).to(tl.float16)

    if INPUT_IS_BF16:
        a_dot = a.to(tl.bfloat16)
    else:
        a_dot = a

    for n_tile_index in range(0, 2):
        pid_n = pid_n_group * 2 + n_tile_index
        offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
        offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        masks_zn = offsets_zn < N // 8
        masks_sn = offsets_sn < N

        if n_tile_index == 0:
            b = first_b
        else:
            b = second_b
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        offsets_z = (N // 8) * first_k_block + offsets_zn
        zeros = tl.load(zeros_ptr + offsets_z, mask=masks_zn, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * first_k_block + offsets_sn
        weight_scales = tl.load(scales_ptr + offsets_s, mask=masks_sn, other=0.0)
        weight_scales = tl.broadcast_to(weight_scales[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        if INPUT_IS_BF16:
            b = ((b - zeros) * weight_scales).to(tl.bfloat16)
        else:
            b = ((b - zeros) * weight_scales).to(tl.float16)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator = tl.dot(a_dot, b, accumulator, out_dtype=tl.float32)
        tile_id = pid_m * num_pid_n + pid_n
        partial_offsets = (
            (tile_id * SPLIT_K + split_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_N)[None, :]
        )
        output_mask = masks_am[:, None] & masks_sn[None, :]
        tl.store(partials_ptr + partial_offsets, accumulator, mask=output_mask)

        prior_count = tl.atomic_add(counters_ptr + tile_id, 1, sem="acq_rel", scope="gpu")
        if prior_count == SPLIT_K - 1:
            reduced = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for partial_id in range(0, SPLIT_K):
                reduction_offsets = (
                    (tile_id * SPLIT_K + partial_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
                    + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
                    + tl.arange(0, BLOCK_SIZE_N)[None, :]
                )
                reduced += tl.load(partials_ptr + reduction_offsets, mask=output_mask, other=0.0)
            c = reduced.to(c_ptr.type.element_ty)
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offsets_sn, mask=masks_sn, other=0.0)
                c += bias[None, :].to(c_ptr.type.element_ty)
            c_ptrs = c_ptr + N * offsets_am[:, None] + offsets_sn[None, :]
            tl.store(c_ptrs, c, mask=output_mask)
            tl.atomic_xchg(counters_ptr + tile_id, 0, sem="release", scope="gpu")


@triton.jit
def paroquant_rotation_gemm_splitk_two_n_tiles_pair_counter_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    partner_ptr,
    cos_ptr,
    sin_ptr,
    channel_scales_ptr,
    bias_ptr,
    partials_ptr,
    counters_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    KROT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_BLOCKS_PER_SPLIT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
    PREFETCH_FIRST_WEIGHT: tl.constexpr,
    PREFETCH_SECOND_WEIGHT: tl.constexpr,
    ATOMIC_COUNTER_RESET: tl.constexpr,
):
    """Process two output tiles with one completion counter after both partials are visible."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_n_groups = tl.cdiv(num_pid_n, 2)
    cta_tile_id = pid // SPLIT_K
    split_id = pid % SPLIT_K
    pid_m = cta_tile_id // num_pid_n_groups
    pid_n_group = cta_tile_id % num_pid_n_groups

    reverse_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)
    shifts = reverse_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    masks_am = offsets_am < M
    first_k_block = split_id * K_BLOCKS_PER_SPLIT
    global_k = first_k_block * BLOCK_SIZE_K + offsets_k
    masks_k = global_k < K
    masks_a = masks_am[:, None] & masks_k[None, :]
    a_ptrs = a_ptr + K * offsets_am[:, None] + global_k[None, :]

    first_pid_n = pid_n_group * 2
    first_offsets_bn = first_pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    first_masks_bn = first_offsets_bn < N // 8
    first_b_ptrs = b_ptr + (N // 8) * global_k[:, None] + first_offsets_bn[None, :]
    second_offsets_bn = (first_pid_n + 1) * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    second_masks_bn = second_offsets_bn < N // 8
    second_b_ptrs = b_ptr + (N // 8) * global_k[:, None] + second_offsets_bn[None, :]

    a = tl.load(a_ptrs, mask=masks_a, other=0.0)
    first_partner = tl.load(partner_ptr + global_k, mask=masks_k, other=offsets_k)
    channel_scales = tl.load(channel_scales_ptr + global_k, mask=masks_k, other=1.0)
    a = (a.to(tl.float16) * channel_scales[None, :].to(tl.float16)).to(tl.float16)
    if PREFETCH_FIRST_WEIGHT:
        first_b = tl.load(first_b_ptrs, mask=masks_k[:, None] & first_masks_bn[None, :], other=0)
    if PREFETCH_SECOND_WEIGHT:
        second_b = tl.load(second_b_ptrs, mask=masks_k[:, None] & second_masks_bn[None, :], other=0)

    for rot_idx in range(0, KROT):
        lookup_offsets = rot_idx * K + global_k
        if rot_idx == 0:
            partner = first_partner
        else:
            partner = tl.load(partner_ptr + lookup_offsets, mask=masks_k, other=offsets_k)
        partner = tl.broadcast_to(partner[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        paired = tl.gather(a, partner, axis=1)
        cos_value = tl.load(cos_ptr + lookup_offsets, mask=masks_k, other=1.0)
        sin_value = tl.load(sin_ptr + lookup_offsets, mask=masks_k, other=0.0)
        a = tl.fma(a, cos_value[None, :], paired * sin_value[None, :]).to(tl.float16)

    if not PREFETCH_FIRST_WEIGHT:
        first_b = tl.load(first_b_ptrs, mask=masks_k[:, None] & first_masks_bn[None, :], other=0)
    if not PREFETCH_SECOND_WEIGHT:
        second_b = tl.load(second_b_ptrs, mask=masks_k[:, None] & second_masks_bn[None, :], other=0)

    if INPUT_IS_BF16:
        a_dot = a.to(tl.bfloat16)
    else:
        a_dot = a

    for n_tile_index in range(0, 2):
        pid_n = pid_n_group * 2 + n_tile_index
        offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
        offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        masks_zn = offsets_zn < N // 8
        masks_sn = offsets_sn < N

        if n_tile_index == 0:
            b = first_b
        else:
            b = second_b
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        offsets_z = (N // 8) * first_k_block + offsets_zn
        zeros = tl.load(zeros_ptr + offsets_z, mask=masks_zn, other=0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * first_k_block + offsets_sn
        weight_scales = tl.load(scales_ptr + offsets_s, mask=masks_sn, other=0.0)
        weight_scales = tl.broadcast_to(weight_scales[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        if INPUT_IS_BF16:
            b = ((b - zeros) * weight_scales).to(tl.bfloat16)
        else:
            b = ((b - zeros) * weight_scales).to(tl.float16)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator = tl.dot(a_dot, b, accumulator, out_dtype=tl.float32)
        tile_id = pid_m * num_pid_n + pid_n
        partial_offsets = (
            (tile_id * SPLIT_K + split_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_N)[None, :]
        )
        output_mask = masks_am[:, None] & masks_sn[None, :]
        tl.store(partials_ptr + partial_offsets, accumulator, mask=output_mask)

    pair_counter_id = pid_m * num_pid_n_groups + pid_n_group
    prior_count = tl.atomic_add(counters_ptr + pair_counter_id, 1, sem="acq_rel", scope="gpu")
    if prior_count == SPLIT_K - 1:
        for n_tile_index in range(0, 2):
            pid_n = pid_n_group * 2 + n_tile_index
            offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            masks_sn = offsets_sn < N
            tile_id = pid_m * num_pid_n + pid_n
            output_mask = masks_am[:, None] & masks_sn[None, :]
            reduced = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for partial_id in range(0, SPLIT_K):
                reduction_offsets = (
                    (tile_id * SPLIT_K + partial_id) * BLOCK_SIZE_M * BLOCK_SIZE_N
                    + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
                    + tl.arange(0, BLOCK_SIZE_N)[None, :]
                )
                reduced += tl.load(partials_ptr + reduction_offsets, mask=output_mask, other=0.0)
            c = reduced.to(c_ptr.type.element_ty)
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offsets_sn, mask=masks_sn, other=0.0)
                c += bias[None, :].to(c_ptr.type.element_ty)
            c_ptrs = c_ptr + N * offsets_am[:, None] + offsets_sn[None, :]
            tl.store(c_ptrs, c, mask=output_mask)
        if ATOMIC_COUNTER_RESET:
            tl.atomic_xchg(counters_ptr + pair_counter_id, 0, sem="release", scope="gpu")
        else:
            tl.store(counters_ptr + pair_counter_id, 0)


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


def _validate_rotation_lookup(
    input: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
) -> int:
    if partner.dim() != 2:
        raise ValueError(f"ParoQuant megakernel partner lookup must be rank 2, got {tuple(partner.shape)}.")
    if cos.shape != partner.shape or sin.shape != partner.shape:
        raise ValueError(
            "ParoQuant megakernel trigonometric lookups must match partner shape, got "
            f"partner={tuple(partner.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}."
        )
    if partner.shape[1] != input.shape[1]:
        raise ValueError(
            f"ParoQuant megakernel lookup width ({partner.shape[1]}) must match K ({input.shape[1]})."
        )
    if channel_scales.numel() != input.shape[1]:
        raise ValueError(
            "ParoQuant megakernel channel scales must contain one value per input feature, got "
            f"{channel_scales.numel()} for K={input.shape[1]}."
        )
    tensors = (partner, cos, sin, channel_scales)
    if any(tensor.device != input.device for tensor in tensors):
        raise ValueError("ParoQuant megakernel inputs and rotation metadata must share one device.")
    # int8/int16 store offsets local to each 128-channel group; int32 stores absolute channel indices.
    if partner.dtype not in {torch.int8, torch.int16, torch.int32}:
        raise ValueError(f"ParoQuant megakernel partner lookup must be int8, int16, or int32, got {partner.dtype}.")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("ParoQuant megakernel cos/sin lookups must use float32.")
    return int(partner.shape[0])


def _paroquant_rotation_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    block_size_m: int,
    block_size_n: int,
    num_warps: int,
    num_stages: int,
    loop_unroll_factor: int = 1,
    explicit_fma: bool = False,
    prefetch_first_partner: bool = False,
    prefetch_packed_weight: bool = False,
    fp32_accum: bool = FP32_ACCUM,
) -> torch.Tensor:
    M, N, K, group_size = _validate_shapes(input, qweight, scales, qzeros)
    if group_size != PAROQUANT_MEGAKERNEL_GROUP_SIZE:
        raise ValueError(
            "ParoQuant megakernel requires group_size=128, "
            f"got group_size={group_size}."
        )
    krot = _validate_rotation_lookup(input, partner, cos, sin, channel_scales)
    if krot not in {1, 8}:
        raise ValueError(f"ParoQuant megakernel supports krot 1 or 8, got {krot}.")
    if input.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"ParoQuant megakernel requires fp16 or bf16 input, got {input.dtype}.")
    if bias is not None and (bias.device != input.device or bias.dtype != input.dtype or bias.numel() != N):
        raise ValueError(
            "ParoQuant megakernel bias must match the input device/dtype and output width, got "
            f"device={bias.device}, dtype={bias.dtype}, shape={tuple(bias.shape)}."
        )

    result = input.new_empty((M, N))
    bias_arg = result if bias is None else bias

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    with get_same_device_cm(qweight):
        paroquant_rotation_gemm_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            partner,
            cos,
            sin,
            channel_scales,
            bias_arg,
            M,
            N,
            K,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=PAROQUANT_MEGAKERNEL_GROUP_SIZE,
            KROT=krot,
            PARTNER_IS_LOCAL=partner.dtype != torch.int32,
            LOOP_UNROLL_FACTOR=loop_unroll_factor,
            EXPLICIT_FMA=explicit_fma,
            PREFETCH_FIRST_PARTNER=prefetch_first_partner,
            PREFETCH_PACKED_WEIGHT=prefetch_packed_weight,
            USE_FP32_ACCUM=fp32_accum,
            HAS_BIAS=bias is not None,
            INPUT_IS_BF16=input.dtype == torch.bfloat16,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return result


def _paroquant_splitk_compiled_launch_supported(compiled_kernel: object) -> bool:
    """Use Triton's direct launcher only for the exact internal ABI validated here."""
    try:
        version_parts = tuple(int(part) for part in triton.__version__.split(".")[:2])
        knobs = triton.knobs
        launcher = compiled_kernel.run
        return (
            version_parts == (3, 7)
            and triton_driver is not None
            and not paroquant_rotation_gemm_splitk_kernel.pre_run_hooks
            and not paroquant_rotation_gemm_splitk_two_n_tiles_kernel.pre_run_hooks
            and not paroquant_rotation_gemm_splitk_two_n_tiles_pair_counter_kernel.pre_run_hooks
            and not knobs.runtime.debug
            and not knobs.compilation.instrumentation_mode
            and callable(getattr(compiled_kernel, "run", None))
            and callable(getattr(compiled_kernel, "launch_metadata", None))
            and hasattr(compiled_kernel, "function")
            and hasattr(compiled_kernel, "packed_metadata")
            and callable(getattr(launcher, "launch", None))
            and launcher.global_scratch_size == 0
            and launcher.profile_scratch_size == 0
            and hasattr(launcher, "launch_cooperative_grid")
            and hasattr(launcher, "launch_pdl")
            and hasattr(launcher, "arg_annotations")
            and hasattr(launcher, "kernel_signature")
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _paroquant_splitk_fp16_prefill_shape(*, rows: int, out_features: int) -> bool:
    """Return whether an FP16 prefill shape has a measured split-K schedule."""
    measured_row_bands = {
        512: (9, 992),
        1920: (9, 256),
        2048: (9, 256),
        2560: (97, 192),
        3072: (81, 160),
        4096: (49, 96),
    }
    row_band = measured_row_bands.get(out_features)
    return row_band is not None and row_band[0] <= rows <= row_band[1]


def _paroquant_splitk_launch_config(
    input_dtype: torch.dtype,
    *,
    rows: int,
    in_features: int = 2048,
    out_features: int,
    split_k: int = 16,
) -> tuple[int, int]:
    """Return the measured row tile and warp count for one split-K shape."""
    if input_dtype == torch.bfloat16:
        if rows in {2, 4} and in_features == 4096 and out_features == 4096 and split_k == 32:
            return 4, 4
        if rows == 32 and in_features == 4096 and out_features == 4096 and split_k == 32:
            return 32, 8
        if rows == 1 and in_features == 12288 and out_features == 4096 and split_k == 96:
            return 1, 4
        if (
            rows == 1
            and in_features == 4096
            and out_features in {1024, 4096, 12288}
            and split_k == 32
        ):
            return 4, 4
        if (
            rows == 8
            and in_features in {4096, 12288}
            and out_features in {1024, 4096, 12288}
            and split_k == 32
        ):
            return 8, 8
    if input_dtype == torch.float16:
        if _paroquant_splitk_fp16_prefill_shape(rows=rows, out_features=out_features):
            return 32, 4
        if out_features == 8192 and rows in {1, 2}:
            return 2, 4
        if out_features == 8192 and rows in {3, 4}:
            return 4, 4
        if rows == 1 and out_features == 2048:
            return 2, 4
        if rows in {2, 3, 4} and out_features == 2048:
            return 4, 4
        if rows in {5, 6, 7} and out_features == 2048:
            return 8, 4
    if rows == 1 and out_features == 8192:
        return 4, 4
    return 8, 8


def _paroquant_splitk_output_config(
    input_dtype: torch.dtype,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    split_k: int,
) -> tuple[int, int, int, int | None, bool, bool]:
    """Return the output schedule, including whether paired CTAs prefetch their first weight tile."""
    if (
        input_dtype == torch.bfloat16
        and rows == 2
        and in_features == 4096
        and out_features == 4096
        and split_k == 32
    ):
        return 128, 2, 2, 144, True, True
    if (
        input_dtype == torch.bfloat16
        and rows == 32
        and in_features == 4096
        and out_features == 4096
        and split_k == 32
    ):
        return 128, 1, 2, 128, True, True
    if (
        input_dtype == torch.float16
        and in_features == 2048
        and split_k == 16
        and _paroquant_splitk_fp16_prefill_shape(rows=rows, out_features=out_features)
    ):
        return 128, 1, 1, None, False, False
    if (
        input_dtype == torch.float16
        and in_features == 2048
        and out_features == 8192
        and 1 <= rows <= 8
        and split_k == 16
    ):
        maxnreg = {1: 136, 2: 168, 3: 144, 4: 120}.get(rows, 76)
        return 128, 1, 2, maxnreg, True, rows >= 6
    if (
        input_dtype == torch.bfloat16
        and in_features == 4096
        and out_features == 12288
        and rows == 1
        and split_k == 32
    ):
        return 128, 2, 2, 128, True, False
    if (
        input_dtype == torch.bfloat16
        and in_features == 12288
        and out_features == 4096
        and rows == 1
        and split_k == 96
    ):
        return 128, 2, 2, 128, True, False
    if (
        input_dtype == torch.bfloat16
        and in_features == 4096
        and out_features == 4096
        and rows == 1
        and split_k == 32
    ):
        return 128, 2, 1, 128, False, False
    if (
        input_dtype == torch.bfloat16
        and in_features == 4096
        and out_features == 12288
        and rows == 8
        and split_k == 32
    ):
        return 128, 2, 2, None, False, False
    return 128, 2, 1, None, False, False


def _paroquant_splitk_jit_kernel(n_tiles_per_cta: int, pair_counter: bool):
    """Select the JIT function implementing one measured split-K output schedule."""
    if pair_counter:
        if n_tiles_per_cta != 2:
            raise ValueError("ParoQuant paired counters require two output tiles per CTA.")
        return paroquant_rotation_gemm_splitk_two_n_tiles_pair_counter_kernel
    if n_tiles_per_cta == 1:
        return paroquant_rotation_gemm_splitk_kernel
    if n_tiles_per_cta == 2:
        return paroquant_rotation_gemm_splitk_two_n_tiles_kernel
    raise ValueError(f"Unsupported ParoQuant split-K output tiles per CTA: {n_tiles_per_cta}.")


def _paroquant_prepare_splitk_compiled_launch(
    compiled_kernel: object,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    partials: torch.Tensor,
    counters: torch.Tensor,
    *,
    stream: int,
    input_dtype: torch.dtype,
    rows: int,
    in_features: int,
    split_k: int,
) -> Callable[..., torch.Tensor]:
    """Bind invariant launch state into the warm eager decode submission."""
    if triton_driver is None:
        raise RuntimeError("Triton's compiled split-K launcher is unavailable.")
    n = qweight.shape[1] * 8
    krot = partner.shape[0]
    block_size_m, _ = _paroquant_splitk_launch_config(
        input_dtype,
        rows=rows,
        in_features=in_features,
        out_features=n,
        split_k=split_k,
    )
    block_size_n, _, n_tiles_per_cta, _, pair_counter, prefetch_first_weight = (
        _paroquant_splitk_output_config(
            input_dtype,
            rows=rows,
            in_features=in_features,
            out_features=n,
            split_k=split_k,
        )
    )
    jit_kernel = _paroquant_splitk_jit_kernel(n_tiles_per_cta, pair_counter)
    if (
        jit_kernel.pre_run_hooks
        or triton.knobs.runtime.debug
        or triton.knobs.compilation.instrumentation_mode
    ):
        raise TypeError("Triton's compiled split-K launcher is disabled by active JIT instrumentation.")
    num_pid_m = (rows + block_size_m - 1) // block_size_m
    num_pid_n = (n + block_size_n - 1) // block_size_n
    num_cta_tiles = num_pid_m * (
        (num_pid_n + n_tiles_per_cta - 1) // n_tiles_per_cta
    )
    grid = (num_cta_tiles * split_k,)
    launcher = compiled_kernel.run

    def launch(
        input: torch.Tensor,
        *,
        result_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if (
            jit_kernel.pre_run_hooks
            or triton.knobs.runtime.debug
            or triton.knobs.compilation.instrumentation_mode
        ):
            raise TypeError("Triton's compiled split-K launcher is disabled by active JIT instrumentation.")
        result = input.new_empty((rows, n) if result_shape is None else result_shape)
        bias_arg = result if bias is None else bias
        launch_args = (
            input,
            qweight,
            result,
            qzeros,
            scales,
            partner,
            cos,
            sin,
            channel_scales,
            bias_arg,
            partials,
            counters,
            rows,
            n,
            in_features,
            block_size_m,
            block_size_n,
            PAROQUANT_MEGAKERNEL_GROUP_SIZE,
            krot,
            split_k,
            (in_features // PAROQUANT_MEGAKERNEL_GROUP_SIZE) // split_k,
            bias is not None,
            input_dtype == torch.bfloat16,
        )
        if pair_counter:
            launch_args += (
                prefetch_first_weight,
                PAROQUANT_SPLITK_PAIR_PREFETCH_SECOND_WEIGHT,
                PAROQUANT_SPLITK_PAIR_ATOMIC_RESET,
            )

        launch_enter_hook = triton.knobs.runtime.launch_enter_hook
        launch_exit_hook = triton.knobs.runtime.launch_exit_hook
        enter_calls = getattr(launch_enter_hook, "calls", None)
        exit_calls = getattr(launch_exit_hook, "calls", None)
        hooks_inactive = (
            (launch_enter_hook is None or (enter_calls is not None and not enter_calls))
            and (launch_exit_hook is None or (exit_calls is not None and not exit_calls))
        )
        if hooks_inactive:
            # Triton 3.7 installs empty HookChain objects by default; passing them still enters Python twice.
            launch_metadata = None
            launch_enter_hook = None
            launch_exit_hook = None
        else:
            launch_metadata = compiled_kernel.launch_metadata(
                grid,
                stream,
                *launch_args,
            )
        # This kernel has no compiler-managed scratch. Calling the generated C launcher directly avoids the
        # zero-sized global/profile scratch allocation wrapper on the exact Triton ABI checked above.
        launcher.launch(
            grid[0],
            1,
            1,
            stream,
            compiled_kernel.function,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            compiled_kernel.packed_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            None,
            None,
            launcher.arg_annotations,
            launcher.kernel_signature,
            launch_args,
        )
        return result

    return launch


def _paroquant_rotation_gemm_splitk_triton_prepare(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    partials: torch.Tensor,
    counters: torch.Tensor,
    *,
    split_k: int,
    result_shape: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, object | None]:
    """Launch through JIT once and return a compatible cached launcher when available."""
    M, K = input.shape
    N = qweight.shape[1] * 8
    krot = partner.shape[0]
    block_size_m, num_warps = _paroquant_splitk_launch_config(
        input.dtype,
        rows=M,
        in_features=K,
        out_features=N,
        split_k=split_k,
    )
    (
        block_size_n,
        num_stages,
        n_tiles_per_cta,
        maxnreg,
        pair_counter,
        prefetch_first_weight,
    ) = _paroquant_splitk_output_config(
        input.dtype,
        rows=M,
        in_features=K,
        out_features=N,
        split_k=split_k,
    )
    result = input.new_empty((M, N) if result_shape is None else result_shape)
    bias_arg = result if bias is None else bias
    num_pid_m = (M + block_size_m - 1) // block_size_m
    num_pid_n = (N + block_size_n - 1) // block_size_n
    num_cta_tiles = num_pid_m * (
        (num_pid_n + n_tiles_per_cta - 1) // n_tiles_per_cta
    )
    grid = (num_cta_tiles * split_k,)
    jit_kernel = _paroquant_splitk_jit_kernel(n_tiles_per_cta, pair_counter)
    with get_same_device_cm(qweight):
        compiled_kernel = jit_kernel[grid](
            input,
            qweight,
            result,
            qzeros,
            scales,
            partner,
            cos,
            sin,
            channel_scales,
            bias_arg,
            partials,
            counters,
            M,
            N,
            K,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=PAROQUANT_MEGAKERNEL_GROUP_SIZE,
            KROT=krot,
            SPLIT_K=split_k,
            K_BLOCKS_PER_SPLIT=(K // PAROQUANT_MEGAKERNEL_GROUP_SIZE) // split_k,
            HAS_BIAS=bias is not None,
            INPUT_IS_BF16=input.dtype == torch.bfloat16,
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
            **(
                {
                    "PREFETCH_FIRST_WEIGHT": prefetch_first_weight,
                    "PREFETCH_SECOND_WEIGHT": PAROQUANT_SPLITK_PAIR_PREFETCH_SECOND_WEIGHT,
                    "ATOMIC_COUNTER_RESET": PAROQUANT_SPLITK_PAIR_ATOMIC_RESET,
                }
                if pair_counter
                else {}
            ),
        )
    if not _paroquant_splitk_compiled_launch_supported(compiled_kernel):
        compiled_kernel = None
    return result, compiled_kernel


def _paroquant_rotation_gemm_splitk_triton_compiled(
    compiled_kernel: object,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    partials: torch.Tensor,
    counters: torch.Tensor,
    *,
    stream: int,
    split_k: int,
    input_rows: int | None = None,
    result_shape: tuple[int, ...] | None = None,
    prepared_launch: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Launch a validated Triton 3.7 compiled kernel without repeating JIT specialization binding."""
    if triton_driver is None:
        raise RuntimeError("Triton's compiled split-K launcher is unavailable.")
    if prepared_launch is not None:
        return _paroquant_rotation_gemm_splitk_triton_prepared(
            prepared_launch,
            input,
            qweight,
            result_shape=result_shape,
        )
    if input_rows is None:
        M, K = input.shape
    else:
        M = input_rows
        K = input.shape[-1]
    N = qweight.shape[1] * 8
    krot = partner.shape[0]
    block_size_m, _ = _paroquant_splitk_launch_config(
        input.dtype,
        rows=M,
        in_features=K,
        out_features=N,
        split_k=split_k,
    )
    block_size_n, _, n_tiles_per_cta, _, pair_counter, prefetch_first_weight = (
        _paroquant_splitk_output_config(
            input.dtype,
            rows=M,
            in_features=K,
            out_features=N,
            split_k=split_k,
        )
    )
    jit_kernel = _paroquant_splitk_jit_kernel(n_tiles_per_cta, pair_counter)
    if (
        jit_kernel.pre_run_hooks
        or triton.knobs.runtime.debug
        or triton.knobs.compilation.instrumentation_mode
    ):
        raise TypeError("Triton's compiled split-K launcher is disabled by active JIT instrumentation.")
    result = input.new_empty((M, N) if result_shape is None else result_shape)
    bias_arg = result if bias is None else bias
    num_pid_m = (M + block_size_m - 1) // block_size_m
    num_pid_n = (N + block_size_n - 1) // block_size_n
    num_cta_tiles = num_pid_m * (
        (num_pid_n + n_tiles_per_cta - 1) // n_tiles_per_cta
    )
    grid = (num_cta_tiles * split_k,)
    launch_args = (
        input,
        qweight,
        result,
        qzeros,
        scales,
        partner,
        cos,
        sin,
        channel_scales,
        bias_arg,
        partials,
        counters,
        M,
        N,
        K,
        block_size_m,
        block_size_n,
        PAROQUANT_MEGAKERNEL_GROUP_SIZE,
        krot,
        split_k,
        (K // PAROQUANT_MEGAKERNEL_GROUP_SIZE) // split_k,
        bias is not None,
        input.dtype == torch.bfloat16,
    )
    if pair_counter:
        launch_args += (
            prefetch_first_weight,
            PAROQUANT_SPLITK_PAIR_PREFETCH_SECOND_WEIGHT,
            PAROQUANT_SPLITK_PAIR_ATOMIC_RESET,
        )
    launcher = compiled_kernel.run

    def launch() -> None:
        launch_enter_hook = triton.knobs.runtime.launch_enter_hook
        launch_exit_hook = triton.knobs.runtime.launch_exit_hook
        enter_calls = getattr(launch_enter_hook, "calls", None)
        exit_calls = getattr(launch_exit_hook, "calls", None)
        hooks_inactive = (
            (launch_enter_hook is None or (enter_calls is not None and not enter_calls))
            and (launch_exit_hook is None or (exit_calls is not None and not exit_calls))
        )
        if hooks_inactive:
            # Triton 3.7 installs empty HookChain objects by default; passing them still enters Python twice.
            launch_metadata = None
            launch_enter_hook = None
            launch_exit_hook = None
        else:
            launch_metadata = compiled_kernel.launch_metadata(grid, stream, *launch_args)
        # This kernel has no compiler-managed scratch. Calling the generated C launcher directly avoids the
        # zero-sized global/profile scratch allocation wrapper on the exact Triton ABI checked above.
        launcher.launch(
            grid[0],
            1,
            1,
            stream,
            compiled_kernel.function,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            compiled_kernel.packed_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            None,
            None,
            launcher.arg_annotations,
            launcher.kernel_signature,
            launch_args,
        )

    device_index = qweight.device.index
    if torch.cuda.current_device() == device_index:
        launch()
    else:
        with get_same_device_cm(qweight):
            launch()
    return result


def _paroquant_rotation_gemm_splitk_triton_prepared(
    prepared_launch: Callable[..., torch.Tensor],
    input: torch.Tensor,
    qweight: torch.Tensor,
    *,
    result_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Submit a prepared split-K launch on its validated device and stream."""
    if triton_driver is None:
        raise RuntimeError("Triton's compiled split-K launcher is unavailable.")
    if torch.cuda.current_device() == qweight.device.index:
        return prepared_launch(input, result_shape=result_shape)
    with get_same_device_cm(qweight):
        return prepared_launch(input, result_shape=result_shape)


def _paroquant_rotation_gemm_splitk_triton_unchecked(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    partials: torch.Tensor,
    counters: torch.Tensor,
    *,
    split_k: int,
    result_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Launch split-K after the module or public wrapper has validated every invariant."""
    result, _ = _paroquant_rotation_gemm_splitk_triton_prepare(
        input,
        qweight,
        scales,
        qzeros,
        partner,
        cos,
        sin,
        channel_scales,
        bias,
        partials,
        counters,
        split_k=split_k,
        result_shape=result_shape,
    )
    return result


def _paroquant_rotation_gemm_splitk_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None,
    partials: torch.Tensor,
    counters: torch.Tensor,
    *,
    split_k: int,
) -> torch.Tensor:
    """Run a measured one-launch FP16/BF16 split-K schedule with caller-owned scratch."""
    M, N, K, group_size = _validate_shapes(input, qweight, scales, qzeros)
    krot = _validate_rotation_lookup(input, partner, cos, sin, channel_scales)
    measured_fp16_prefill = (
        input.dtype == torch.float16
        and K == 2048
        and split_k == 16
        and _paroquant_splitk_fp16_prefill_shape(rows=M, out_features=N)
    )
    if (
        split_k <= 0
        or input.dtype not in {torch.float16, torch.bfloat16}
        or group_size != PAROQUANT_MEGAKERNEL_GROUP_SIZE
        or partner.dtype != (torch.int8 if input.dtype == torch.bfloat16 else torch.int16)
        or krot != 8
        or (M > 8 and not measured_fp16_prefill)
        or K % PAROQUANT_MEGAKERNEL_GROUP_SIZE != 0
        or (K // PAROQUANT_MEGAKERNEL_GROUP_SIZE) % split_k != 0
    ):
        raise ValueError(
            "ParoQuant split-K requires a measured FP16/BF16 shape, local partner indices, and even K splits."
        )
    if bias is not None and (bias.device != input.device or bias.dtype != input.dtype or bias.numel() != N):
        raise ValueError("ParoQuant split-K bias must match the input device, dtype, and output width.")
    block_size_m, _ = _paroquant_splitk_launch_config(
        input.dtype,
        rows=M,
        in_features=K,
        out_features=N,
        split_k=split_k,
    )
    block_size_n, _, _, _, _, _ = _paroquant_splitk_output_config(
        input.dtype,
        rows=M,
        in_features=K,
        out_features=N,
        split_k=split_k,
    )
    num_tiles = ((M + block_size_m - 1) // block_size_m) * (
        (N + block_size_n - 1) // block_size_n
    )
    expected_partials = num_tiles * split_k * block_size_m * block_size_n
    if (
        partials.device != input.device
        or partials.dtype != torch.float32
        or not partials.is_contiguous()
        or partials.numel() < expected_partials
    ):
        raise ValueError("ParoQuant split-K partial scratch does not match the requested launch.")
    if (
        counters.device != input.device
        or counters.dtype != torch.int32
        or not counters.is_contiguous()
        or counters.numel() < num_tiles
    ):
        raise ValueError("ParoQuant split-K counter scratch does not match the requested launch.")

    return _paroquant_rotation_gemm_splitk_triton_unchecked(
        input,
        qweight,
        scales,
        qzeros,
        partner,
        cos,
        sin,
        channel_scales,
        bias,
        partials,
        counters,
        split_k=split_k,
    )


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


def paroquant_rotation_gemm_triton_decode(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    K = input.shape[1]
    krot = partner.shape[0]
    # Eight warps improve decode across the supported K range. BF16 and the lighter krot=1 path also benefit from
    # processing all supported decode rows in one tile. FP16 krot=8 needs narrower tiles to control register
    # pressure, with a measured two-row specialization for one- and two-token decode.
    if input.dtype == torch.float16 and krot == 8:
        block_size_m = 2 if input.shape[0] <= 2 else 4
    else:
        block_size_m = 8
    # Starting BF16's first gather while channel scaling is still in flight reduces long-scoreboard stalls.
    # For K>=1024, issuing the packed-weight load before the rotations hides another scoreboard dependency.
    # The old factor-two K-loop unroll only adds live state once these prefetches are enabled.
    prefetch_first_partner = input.dtype == torch.bfloat16 and krot == 8 and K >= 384
    prefetch_packed_weight = input.dtype == torch.bfloat16 and krot == 8 and K >= 1024
    loop_unroll_factor = 2 if input.dtype == torch.bfloat16 and krot == 8 and not prefetch_first_partner else 1
    explicit_fma = krot == 8 and K >= 1024
    return _paroquant_rotation_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        partner,
        cos,
        sin,
        channel_scales,
        bias,
        block_size_m=block_size_m,
        block_size_n=128,
        num_warps=8,
        num_stages=2,
        loop_unroll_factor=loop_unroll_factor,
        explicit_fma=explicit_fma,
        prefetch_first_partner=prefetch_first_partner,
        prefetch_packed_weight=prefetch_packed_weight,
        fp32_accum=FP32_ACCUM,
    )


def _prefill_bm32_halves_cta_waves(
    input: torch.Tensor,
    *,
    m: int,
    n: int,
    sm_count: int | None = None,
) -> bool:
    """Use BM32 only across measured bands where it halves the CTA wave count."""
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(input.device).multi_processor_count
    n_tiles = (n + 127) // 128
    bm16_ctas = ((m + 15) // 16) * n_tiles
    bm32_ctas = ((m + 31) // 32) * n_tiles
    bm16_waves = (bm16_ctas + sm_count - 1) // sm_count
    bm32_waves = (bm32_ctas + sm_count - 1) // sm_count
    halves_waves = bm16_waves == 2 * bm32_waves
    max_measured_waves = 4 if n == 2048 else 1
    return halves_waves and (bm32_waves == 1 or (sm_count == 124 and bm32_waves <= max_measured_waves))


def _prefill_small_n_bm32_collapses_waves(input: torch.Tensor, *, m: int, n: int) -> bool:
    """Use BM32 only in measured 124-SM small-N bands where BM8 needs at least three waves."""
    sm_count = torch.cuda.get_device_properties(input.device).multi_processor_count
    if sm_count != 124:
        return False
    n_tiles = (n + 127) // 128
    bm8_ctas = ((m + 7) // 8) * n_tiles
    bm32_ctas = ((m + 31) // 32) * n_tiles
    bm8_waves = (bm8_ctas + sm_count - 1) // sm_count
    bm32_waves = (bm32_ctas + sm_count - 1) // sm_count
    return bm8_waves >= 3 and bm32_waves == 1


def paroquant_rotation_gemm_triton_prefill(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    partner: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    channel_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    wide_fp16_candidate = (
        N > 512
        and input.dtype == torch.float16
        and K == 2048
        and 640 <= N <= 4096
        and N % 128 == 0
        and partner.shape[0] == 8
    )
    wide_fp16_sm_count = (
        torch.cuda.get_device_properties(input.device).multi_processor_count if wide_fp16_candidate else None
    )
    wide_fp16_bm32 = (
        wide_fp16_candidate
        and (N <= 2048 or wide_fp16_sm_count == 124)
        and _prefill_bm32_halves_cta_waves(input, m=M, n=N, sm_count=wide_fp16_sm_count)
    )
    if N <= 512:
        block_size_m = (
            32
            if (
                input.dtype in {torch.float16, torch.bfloat16}
                and K == 2048
                and N in {128, 256, 384, 512}
                and partner.shape[0] == 8
                and _prefill_small_n_bm32_collapses_waves(input, m=M, n=N)
            )
            else 8
        )
    elif wide_fp16_bm32:
        block_size_m = 32
    else:
        block_size_m = 16
    wide_fp16_bm32_w16 = wide_fp16_bm32 and wide_fp16_sm_count == 124
    if (K, N, partner.shape[0]) == (2048, 512, 8):
        loop_unroll_factor = 2 if input.dtype == torch.float16 else 4
    else:
        loop_unroll_factor = 1
    explicit_fma = partner.shape[0] == 8 and K >= 1024 and N <= 512
    # Keep the prefetch on the small-N BF16 route where duplicated rotation latency dominates.
    prefetch_first_partner = input.dtype == torch.bfloat16 and partner.shape[0] == 8 and K >= 1024 and N <= 512
    prefetch_packed_weight = prefetch_first_partner
    return _paroquant_rotation_gemm_triton(
        input,
        qweight,
        scales,
        qzeros,
        partner,
        cos,
        sin,
        channel_scales,
        bias,
        block_size_m=block_size_m,
        block_size_n=128,
        num_warps=16 if wide_fp16_bm32_w16 else 8,
        num_stages=1 if wide_fp16_bm32_w16 else 2,
        loop_unroll_factor=loop_unroll_factor,
        explicit_fma=explicit_fma,
        prefetch_first_partner=prefetch_first_partner,
        prefetch_packed_weight=prefetch_packed_weight,
        fp32_accum=FP32_ACCUM,
    )


__all__ = [
    "paroquant_dequantize_triton",
    "paroquant_gemm_triton_decode",
    "paroquant_gemm_triton_prefill",
    "paroquant_rotation_gemm_triton_decode",
    "paroquant_rotation_gemm_triton_prefill",
]
