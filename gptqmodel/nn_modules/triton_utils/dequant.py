# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import itertools
from typing import List

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd

from ...utils.torch import HAS_XPU


def make_dequant_configs(block_sizes: List[int], num_warps: List[int], num_stages: List[int]):
    configs = []
    for bs, ws, ns in itertools.product(block_sizes, num_warps, num_stages):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws, num_stages=ns))
    return configs

# tested on A100 with [Llama 3.2 1B and Falcon 7B] bits:4, group_size:128
DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([1024], [1], [1])
#DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8], [2]) <- slower
@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    out_dtype: tl.constexpr,
    numels,
    pack_bits: tl.constexpr,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    # Block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // out_features
    col_idx = x_index % out_features

    pack_scale: tl.constexpr = pack_bits // bits
    qzero_ncols: tl.constexpr = out_features // pack_scale

    # Load group indices
    g_idx = tl.load(g_idx_ptr + row_idx, mask=xmask, eviction_policy="evict_last")
    groups = tl.where(g_idx < 0, g_idx + num_groups, g_idx)

    # Load scales
    scales = tl.cast(
        tl.load(scales_ptr + (col_idx + out_features * groups), mask=xmask, eviction_policy="evict_last"),
        tl.float32)

    # Load zeros
    if bits == 3:
        # For 3-bit, we need to calculate the correct position in the packed zeros
        zero_bit_pos = (groups * out_features + col_idx) * 3
        zero_word_idx = zero_bit_pos // 32
        zero_bit_offset = zero_bit_pos % 32

        zero_word = tl.load(qzeros_ptr + zero_word_idx, mask=xmask, eviction_policy="evict_last")

        # Handle case where 3-bit value is fully within current 32-bit word
        if zero_bit_offset <= 29:
            zeros = (zero_word >> zero_bit_offset) & 0b111
        else:
            # 3-bit value spans two 32-bit words
            next_zero_word = tl.load(qzeros_ptr + zero_word_idx + 1, mask=xmask, eviction_policy="evict_last")
            combined = (zero_word >> zero_bit_offset) | (next_zero_word << (32 - zero_bit_offset))
            zeros = combined & 0b111
    else:
        qzeros = tl.load(qzeros_ptr + (qzero_ncols * groups + col_idx // pack_scale), mask=xmask,
                         eviction_policy="evict_last")
        wf_zeros = (col_idx % pack_scale) * bits
        zeros = (qzeros >> wf_zeros) & maxq

    # Load weights
    if bits == 3:
        # For 3-bit, we need to calculate the correct position in the packed weights
        weight_bit_pos = (row_idx * out_features + col_idx) * 3
        weight_word_idx = weight_bit_pos // 32
        weight_bit_offset = weight_bit_pos % 32

        weight_word = tl.load(qweight_ptr + weight_word_idx, mask=xmask, eviction_policy="evict_last")

        # Handle case where 3-bit value is fully within current 32-bit word
        if weight_bit_offset <= 29:
            weights = (weight_word >> weight_bit_offset) & 0b111
        else:
            # 3-bit value spans two 32-bit words
            next_weight_word = tl.load(qweight_ptr + weight_word_idx + 1, mask=xmask, eviction_policy="evict_last")
            combined = (weight_word >> weight_bit_offset) | (next_weight_word << (32 - weight_bit_offset))
            weights = combined & 0b111
    else:
        qweights = tl.load(qweight_ptr + (col_idx + out_features * (row_idx // pack_scale)), mask=xmask,
                           eviction_policy="evict_last")
        wf_weights = (row_idx % pack_scale) * bits
        weights = (qweights >> wf_weights) & maxq

    # Dequantize
    weights = (weights - zeros).to(tl.float32) * scales
    weights = tl.cast(weights, out_dtype)
    tl.store(out_ptr + x_index, weights, mask=xmask)


def torch_dtype_to_triton(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.int32:
        return tl.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def dequant(dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
    """
    Launcher for triton dequant kernel. Supports bits = 2, 3, 4, 8
    """
    assert bits in [2, 3, 4, 8], "Only 2, 3, 4, 8 bits are supported"

    num_groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = g_idx.shape[0]

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=dtype)
    out_dtype = dtype

    numels = out.numel()
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    dequant_kernel[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        torch_dtype_to_triton(out_dtype),
        numels,
        pack_bits=pack_bits,
        maxq=maxq,
        bits=bits,
        out_features=out_features,
        num_groups=num_groups,
    )
    return out


def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False):
    W = dequant(input.dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    if transpose:
        return input @ W.t()
    return input @ W

class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="xpu" if HAS_XPU else "cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits = bits, maxq, pack_bits
        return output

    @staticmethod
    @custom_bwd(device_type="xpu" if HAS_XPU else "cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits = ctx.bits, ctx.maxq, ctx.pack_bits
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
