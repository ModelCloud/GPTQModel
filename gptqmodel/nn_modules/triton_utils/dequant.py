# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from functools import lru_cache
from typing import List

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd


def make_dequant_configs(block_sizes: List[int], num_warps: List[int], num_stages: List[int]):
    configs = []
    for bs, ws, ns in itertools.product(block_sizes, num_warps, num_stages):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws, num_stages=ns))
    return configs

# tested on A100 with [Llama 3.2 1B and Falcon 7B] bits:4, group_size:128
DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([512], [1], [1])
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
    # For 3-bit, we need to handle packing specially since 32 isn't divisible by 3
    is_3bit: tl.constexpr = bits == 3

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (out_features * (row_idx // pack_scale))),
        None,
    )

    if is_3bit:
        # FIXME: pack_dtype is different
        weight_offset = row_idx % pack_scale
        # 3 bits per weight, 10 weights per 32-bit int32 (with 2 bits unused)
        word_idx = weight_offset // pack_scale
        bit_pos = (weight_offset % pack_scale) * 3
        # Load the entire packed(32-bit) word containing our weight
        weight_word = tl.load(qweight_ptr + (col_idx + (out_features * word_idx)))
        # Extract our 3-bit weight
        weights = (weight_word >> bit_pos) & maxq
    else:
        # Standard handling for 2/4/8 bits
        wf_weights = (row_idx % pack_scale) * bits
        weights = (qweights >> wf_weights) & maxq

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (col_idx + (out_features * groups)), None).to(tl.float32)

    # Unpack zeros
    qzero_ncols: tl.constexpr = out_features // pack_scale
    qzeros = tl.load(
        qzeros_ptr + ((qzero_ncols * groups) + (col_idx // pack_scale)),
        None,
        eviction_policy="evict_last",
    )

    if is_3bit:
        # FIXME: pack_dtype is different
        # Special handling for 3-bit zeros
        zero_offset = col_idx % pack_scale
        word_idx = zero_offset // pack_scale
        bit_pos = (zero_offset % pack_scale) * 3
        zero_word = tl.load(qzeros_ptr + ((qzero_ncols * groups) + word_idx))
        zeros = (zero_word >> bit_pos) & maxq
    else:
        wf_zeros = (col_idx % pack_scale) * bits
        zeros = (qzeros >> wf_zeros) & maxq

    # Dequantize
    weights = (weights - zeros).to(tl.float32) * scales
    weights = tl.cast(weights, out_dtype)
    tl.store(out_ptr + (x_index), weights, mask=xmask)
    #tl.store(out_ptr + (x_index), weights.to(tl.float16 if out_ptr.dtype == tl.float16 else tl.bfloat16), mask=xmask)


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
    @custom_fwd(device_type="cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits = bits, maxq, pack_bits
        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits = ctx.bits, ctx.maxq, ctx.pack_bits
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
