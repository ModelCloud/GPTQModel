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
from typing import List

import itertools
import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd

from ...utils.torch import HAS_XPU

def _sm() -> int:
    # returns 89 for sm_89, 120 for sm_120, etc.
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

def make_dequant_configs(block_sizes: List[int], num_warps: List[int], num_stages: List[int]):
    configs = []
    for bs, ws, ns in itertools.product(block_sizes, num_warps, num_stages):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws, num_stages=ns))
    return configs

def _arch_autotune_space():
    sm = _sm()
    if sm >= 120:  # Blackwell (e.g., 5090)
        return make_dequant_configs([4096, 2048, 1024, 512], [8, 4], [2])
    elif sm >= 89:  # Ada (e.g., 4090)
        return make_dequant_configs([2048, 1024, 512], [2, 4, 8], [1, 2])
    else: # A100/3090
        return make_dequant_configs([1024, 512, 256], [1, 2, 4], [1])

DEQUANT_CONFIGS = _arch_autotune_space()

@triton.autotune(DEQUANT_CONFIGS, key=["numels"])
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
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels

    row_idx = x_index // out_features
    col_idx = x_index % out_features

    # helpful vectorization hints
    tl.multiple_of(col_idx, 32)
    tl.multiple_of(x_index, 128)

    pack_scale: tl.constexpr = pack_bits // bits
    qzero_ncols: tl.constexpr = out_features // pack_scale

    # g_idx small & reused: default cache; others: stream via L2 (.cg)
    g_idx = tl.load(g_idx_ptr + row_idx, mask=xmask)
    groups = tl.where(g_idx < 0, g_idx + num_groups, g_idx)

    # scales contiguous across columns for a fixed row / group
    scales = tl.load(
        scales_ptr + (col_idx + out_features * groups),
        mask=xmask,
        eviction_policy="evict_last",
    )
    scales = tl.cast(scales, tl.float32)

    if bits == 3:
        # ---- branchless 3-bit zeros ----
        # bit position for each element
        zero_bit_pos = (groups * out_features + col_idx) * 3
        word_idx0 = tl.cast(zero_bit_pos >> 5, tl.int32)          # // 32
        bit_off   = tl.cast(zero_bit_pos & 31, tl.int32)          # % 32
        word_idx1 = word_idx0 + 1

        z0 = tl.load(qzeros_ptr + word_idx0, mask=xmask, eviction_policy="evict_last")
        z1 = tl.load(qzeros_ptr + word_idx1, mask=xmask & (bit_off != 0), eviction_policy="evict_last")
        # z0 = tl.cast(tl.load(qzeros_ptr + word_idx0, mask=xmask, eviction_policy="evict_last"),
        #              tl.uint32, bitcast=True)
        z0 = tl.cast(tl.load(qzeros_ptr + word_idx0, mask=xmask, eviction_policy="evict_last"),
                     tl.uint32, bitcast=True)

        zeros_u32 = (z0 >> bit_off) | (z1 << (32 - bit_off))
        zeros = tl.cast(zeros_u32 & 0x7, tl.int32)

        # ---- branchless 3-bit weights ----
        w_bit_pos = (row_idx * out_features + col_idx) * 3
        ww0 = tl.cast(w_bit_pos >> 5, tl.int32)
        woff = tl.cast(w_bit_pos & 31, tl.int32)
        ww1  = ww0 + 1

        w0 = tl.load(qweight_ptr + ww0, mask=xmask, eviction_policy="evict_last")
        w1 = tl.load(qweight_ptr + ww1, mask=xmask & (woff != 0), eviction_policy="evict_last")
        w0 = tl.cast(w0, tl.uint32, bitcast=True); w1 = tl.cast(w1, tl.uint32, bitcast=True)

        weights_u32 = (w0 >> woff) | (w1 << (32 - woff))
        weights = tl.cast(weights_u32 & 0x7, tl.int32)
    else:
        # zeros: many columns will hit same 32-bit lane; still load via .cg
        qzeros = tl.load(
            qzeros_ptr + (qzero_ncols * groups + col_idx // pack_scale),
            mask=xmask,
            eviction_policy="evict_last",
        )
        qzeros = tl.cast(qzeros, tl.uint32, bitcast=True)
        wf_zeros = (col_idx % pack_scale) * bits
        zeros = tl.cast((qzeros >> wf_zeros) & maxq, tl.int32)

        # weights: contiguous across columns for a fixed row-tile
        qweights = tl.load(
            qweight_ptr + (col_idx + out_features * (row_idx // pack_scale)),
            mask=xmask,
            eviction_policy="evict_last",
        )
        qweights = tl.cast(qweights, tl.uint32, bitcast=True)
        wf_weights = (row_idx % pack_scale) * bits
        weights = tl.cast((qweights >> wf_weights) & maxq, tl.int32)

    # dequant: fp32 math → cast on store
    w = (tl.cast(weights, tl.float32) - tl.cast(zeros, tl.float32)) * scales
    w = tl.cast(w, out_dtype)
    tl.store(out_ptr + x_index, w, mask=xmask, eviction_policy="evict_last")

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