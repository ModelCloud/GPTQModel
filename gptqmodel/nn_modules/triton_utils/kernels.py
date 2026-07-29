# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd

from ...utils.logger import setup_logger
from ...utils.torch import HAS_XPU
from . import custom_autotune


log = setup_logger()


@triton.jit
def _load_int3_10_1_10_1_10(base_ptr, base_offsets, idx, mask, word_stride):
    word_offset = tl.where(idx <= 10, 0, tl.where(idx <= 21, 1, 2))
    shift = tl.where(idx <= 10, idx * 3, tl.where(idx <= 21, 1 + 3 * (idx - 11), 2 + 3 * (idx - 22)))
    word = tl.load(base_ptr + base_offsets + word_offset * word_stride, mask=mask, other=0)
    value = (word >> shift) & 0x7

    split_mask = (idx == 10) | (idx == 21)
    next_word = tl.load(base_ptr + base_offsets + (word_offset + 1) * word_stride, mask=mask & split_mask, other=0)
    value_10 = ((word >> 30) & 0x3) | ((next_word << 2) & 0x4)
    value_21 = ((word >> 31) & 0x1) | ((next_word << 1) & 0x6)
    value = tl.where(idx == 10, value_10, value)
    value = tl.where(idx == 21, value_21, value)
    return tl.cast(value, tl.int32)


# code based https://github.com/fpgaminer/GPTQ-triton
@custom_autotune.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 1,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 1,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 1,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
    nearest_power_of_two=True,
    prune_configs_by={
        "early_config_prune": custom_autotune.matmul248_kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)

@triton.jit
def quant_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    g_ptr,
    M,
    N,
    K,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """

    features_per_word: tl.constexpr = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis pack_factor times.
    b_ptrs = b_ptr + (
        (offs_k[:, None] // features_per_word) * stride_bk + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_k
    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + (offs_bn[None, :] // features_per_word)

    shifter = (offs_k % features_per_word) * bits
    zeros_shifter = (offs_bn % features_per_word) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        k_idxs = k * BLOCK_SIZE_K + offs_k
        k_mask = k_idxs < K
        n_mask = offs_bn < N
        load_mask = k_mask[:, None] & n_mask[None, :]
        g_idx = tl.load(g_ptrs, mask=k_mask, other=0)

        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales, mask=load_mask, other=0.0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        if bits == 3:
            zero_word_base = (offs_bn // 32) * 3
            zero_idx = offs_bn % 32
            zero_base_offsets = g_idx[:, None] * stride_zeros + zero_word_base[None, :]
            zeros = _load_int3_10_1_10_1_10(zeros_ptr, zero_base_offsets, zero_idx[None, :], load_mask, 1)

            weight_word_base = (k_idxs // 32) * 3
            weight_idx = k_idxs % 32
            weight_base_offsets = weight_word_base[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            b = _load_int3_10_1_10_1_10(b_ptr, weight_base_offsets, weight_idx[:, None], load_mask, stride_bk)
        else:
            zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros, mask=load_mask, other=0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = (zeros >> zeros_shifter[None, :]) & maxq

            b = tl.load(b_ptrs, mask=load_mask, other=0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
            # Now we need to unpack b (which is N-bit values) into 32-bit values.
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values

        a = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = ((b - zeros) * scales).to(a.dtype)  # Scale and shift, match A dtype for tl.dot

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // features_per_word) * stride_bk
        g_ptrs += BLOCK_SIZE_K

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@custom_autotune.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
    nearest_power_of_two=True,
)

@triton.jit
def transpose_quant_matmul_248_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    g_ptr,
    M,
    N,
    K,
    bits: tl.constexpr,
    maxq: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales,
    stride_zeros,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, N) float16
    B is of shape (K//8, N) int32
    C is of shape (M, K) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    features_per_word: tl.constexpr = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    a_mask = offs_am[:, None] < M
    # b_ptrs is set up such that it repeats elements along the K axis pack_factor times.
    b_ptrs = b_ptr + (
        (offs_bk[:, None] // features_per_word) * stride_bk + offs_n[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    g_ptrs = g_ptr + offs_bk
    bk_mask = offs_bk < K
    g_idx = tl.load(g_ptrs, mask=bk_mask, other=0)

    # shifter is used to extract the N bits of each element in the 32-bit word from B
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + (offs_n[None, :] // features_per_word) + g_idx[:, None] * stride_zeros

    shifter = (offs_bk % features_per_word) * bits
    zeros_shifter = (offs_n % features_per_word) * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k in range(0, num_pid_n):
        n_idxs = k * BLOCK_SIZE_N + offs_n
        n_mask = n_idxs < N
        load_mask = bk_mask[:, None] & n_mask[None, :]
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs, mask=load_mask, other=0.0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

        if bits == 3:
            zero_word_base = (n_idxs // 32) * 3
            zero_idx = n_idxs % 32
            zero_base_offsets = g_idx[:, None] * stride_zeros + zero_word_base[None, :]
            zeros = _load_int3_10_1_10_1_10(zeros_ptr, zero_base_offsets, zero_idx[None, :], load_mask, 1)

            weight_word_base = (offs_bk // 32) * 3
            weight_idx = offs_bk % 32
            weight_base_offsets = weight_word_base[:, None] * stride_bk + n_idxs[None, :] * stride_bn
            b = _load_int3_10_1_10_1_10(b_ptr, weight_base_offsets, weight_idx[:, None], load_mask, stride_bk)
        else:
            zeros = tl.load(zeros_ptrs, mask=load_mask, other=0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = (zeros >> zeros_shifter[None, :]) & maxq

            b = tl.load(b_ptrs, mask=load_mask, other=0)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
            # Now we need to unpack b (which is N-bit values) into 32-bit values.
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values

        a = tl.load(a_ptrs, mask=a_mask & n_mask[None, :], other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        b = ((b - zeros) * scales).to(a.dtype)  # Scale and shift, match A dtype for tl.dot
        b = tl.trans(b)

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += BLOCK_SIZE_N // features_per_word

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=input.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"]),
        )
        quant_matmul_248_kernel[grid](
            input,
            qweight,
            output,
            scales.to(input.dtype),
            qzeros,
            g_idx,
            input.shape[0],
            qweight.shape[1],
            input.shape[1],
            bits,
            maxq,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scales.stride(0),
            qzeros.stride(0),
        )
        return output


def transpose_quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=input.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(output_dim, META["BLOCK_SIZE_K"]),
        )
        transpose_quant_matmul_248_kernel[grid](
            input,
            qweight,
            output,
            scales.to(input.dtype),
            qzeros,
            g_idx,
            input.shape[0],
            qweight.shape[1],
            output_dim,
            bits,
            maxq,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scales.stride(0),
            qzeros.stride(0),
        )
        return output


class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="xpu" if HAS_XPU else "cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd(device_type="xpu" if HAS_XPU else "cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_quant_matmul_248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


def quant_matmul_inference_only_248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=input.dtype)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(input.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(qweight.shape[1], META["BLOCK_SIZE_N"]),
        )
        quant_matmul_248_kernel[grid](
            input,
            qweight,
            output,
            scales,
            qzeros,
            g_idx,
            input.shape[0],
            qweight.shape[1],
            input.shape[1],
            bits,
            maxq,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            scales.stride(0),
            qzeros.stride(0),
        )
        return output


class QuantLinearInferenceOnlyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="xpu" if HAS_XPU else "cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        return output
