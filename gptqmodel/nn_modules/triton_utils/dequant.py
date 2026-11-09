# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd

from ...utils.arch import normalize_arch, tuning_family
from ...utils.torch import HAS_XPU
from ..qlinear._numerics import matmul_fp32_reduce


class DequantLaunch(NamedTuple):
    """Resolved launch shape for :func:`dequant` on one architecture class."""

    x_block: int
    num_warps: int
    num_stages: int


# Defaults tested on A100 with [Llama 3.2 1B and Falcon 7B] bits:4, group_size:128.
# The dequant is memory-latency bound, not tile-shape bound: a full sweep on
# gfx950 (X_BLOCK 256..4096, warps 1..8, stages 1..2) found every candidate
# within 1-2% of this one. Keep it fixed rather than paying the per-call
# autotune dispatch cost for a single candidate.
_DEQUANT_LAUNCH_CONFIGS: dict[str, DequantLaunch] = {
    # Measured on gfx950 (MI350X/MI355X).
    "cdna4": DequantLaunch(x_block=1024, num_warps=1, num_stages=1),
    # RDNA4 (gfx1200/gfx1201/gfx1250): measured on an AMD Radeon AI PRO R9700
    # (``gfx1201``, HIP 7.16.26385).  The CDNA (1024, 1, 1) tile is *not* a
    # uniform choice on wave32: a sweep over X_BLOCK {256..4096}, warps {1,2,4}
    # and stages {1,2} across K,N in {(4096,4096), (4096,11008), (5120,5120),
    # (2048,1024,b8), (896,128), (17408,5120)} put (2048, 4, 1) at a 74.5 us
    # geomean against the CDNA tile's 81.7 us, and it was never slower than
    # 1.01x on any measured shape.  The 8-bit 2048x1024 case is the largest
    # single win (20.6 us vs 32.1 us).  The kernel is still memory-latency
    # bound, so the spread stays within a few percent, but this removes the
    # unmeasured CDNA inheritance.
    "rdna4": DequantLaunch(x_block=2048, num_warps=4, num_stages=1),
    # Measured on RTX 4090 (SM 8.9), torch 2.15.0+cu132 / Triton 3.9.0, 4-bit
    # gs=128 in FP16 and BF16, X_BLOCK in {256..4096} x warps {1,2,4} x stages
    # {1,2}: (512, 1, 2) has the best mean against the per-shape best (hot
    # 1.007x, L2-flushed 1.017x; max regret 1.018x hot / 1.048x flushed) versus
    # the A100 default (1024, 1, 1) (mean 1.024x / max 1.087x).  It is
    # 1.09x faster at 4096x4096 (35.3 vs 38.4 us hot).
    "ada": DequantLaunch(x_block=512, num_warps=1, num_stages=2),
}

# Used for CUDA/MPS/XPU and any unregistered ROCm target.
_DEFAULT_DEQUANT_LAUNCH = DequantLaunch(x_block=1024, num_warps=1, num_stages=1)

# Module-level defaults retained for compatibility with existing imports.
DEQUANT_X_BLOCK = _DEFAULT_DEQUANT_LAUNCH.x_block
DEQUANT_NUM_WARPS = _DEFAULT_DEQUANT_LAUNCH.num_warps
DEQUANT_NUM_STAGES = _DEFAULT_DEQUANT_LAUNCH.num_stages


def launch_config(arch=None) -> DequantLaunch:
    """Return the dequant launch shape for an architecture string or device.

    Routes through :func:`gptqmodel.utils.arch.tuning_family` so CUDA devices
    and ``sm_*`` strings reach the ``ada`` entry, while ROCm ``gfx*`` strings
    keep the ``cdna4``/``rdna4`` behaviour.  ``None`` keeps the A100-tested
    default.
    """

    if arch is None:
        return _DEFAULT_DEQUANT_LAUNCH
    if not isinstance(arch, str):
        return _DEQUANT_LAUNCH_CONFIGS.get(tuning_family(arch), _DEFAULT_DEQUANT_LAUNCH)

    name = normalize_arch(arch)
    if not name.startswith("gfx") and not name.startswith("sm_"):
        return _DEFAULT_DEQUANT_LAUNCH
    return _DEQUANT_LAUNCH_CONFIGS.get(tuning_family(name), _DEFAULT_DEQUANT_LAUNCH)


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
    # 1. block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels

    row_idx = x_index // out_features
    col_idx = x_index % out_features

    pack_scale: tl.constexpr = pack_bits // bits
    qzero_ncols: tl.constexpr = out_features // pack_scale  # only 2/4/8bit

    # 2. load groups / scales
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

    # 3. decode qzeros (zeros) —— GPTQ INT3 10-1-10-1-10 format
    if bits == 3:
        # qzeros shape (row-major):
        #   [num_groups, (out_features // 32) * 3]
        # For every 32 zeros → 3 × 32-bit words:
        #   word0, word1, word2 (10-1-10-1-10 pattern)
        tl.static_assert(out_features % 32 == 0, "out_features must be divisible by 32 for 3-bit zeros")

        BLOCKS_PER_ROW: tl.constexpr = out_features // 32

        block = col_idx // 32        # which 32-value group
        idx32 = col_idx % 32         # index inside a 32-value group (0..31)

        # each group produces one row; each row contains BLOCKS_PER_ROW * 3 words
        row_word_base = groups * (BLOCKS_PER_ROW * 3) + block * 3

        word0 = tl.load(
            qzeros_ptr + row_word_base + 0,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        word1 = tl.load(
            qzeros_ptr + row_word_base + 1,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        word2 = tl.load(
            qzeros_ptr + row_word_base + 2,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )

        # idx32: [0..31] → decode according to GPTQ INT3 10-1-10-1-10 packing
        idx = idx32

        zeros_i32 = tl.cast(idx, tl.int32) & 0  # generate zero-like int32 vector from idx

        # --- case 0: idx in [0..9] → from word0, shift = 3 * idx
        cond_0_9 = idx <= 9
        shift_0_9 = idx * 3
        val_0_9 = (word0 >> shift_0_9) & 0x7
        zeros_i32 = tl.where(cond_0_9, tl.cast(val_0_9, tl.int32), zeros_i32)

        # --- case 1: idx == 10 → word0 bits [30..31] + word1 bit[0]
        cond_10 = idx == 10
        base_10 = (word0 >> 30) & 0x3          # lower 2 bits
        extra_10 = ((word1 >> 0) << 2) & 0x4   # upper 1 bit
        val_10 = (base_10 | extra_10) & 0x7
        zeros_i32 = tl.where(cond_10, tl.cast(val_10, tl.int32), zeros_i32)

        # --- case 2: idx in [11..20] (shifts 1,4,7,...,28 in word1)
        cond_11_20 = (idx >= 11) & (idx <= 20)
        j_11_20 = idx - 11
        shift_11_20 = 1 + 3 * j_11_20
        val_11_20 = (word1 >> shift_11_20) & 0x7
        zeros_i32 = tl.where(cond_11_20, tl.cast(val_11_20, tl.int32), zeros_i32)

        # --- case 3: idx == 21 → word1 bit 31 + word2 bits [0..1]
        cond_21 = idx == 21
        base_21 = (word1 >> 31) & 0x1
        extra_21 = ((word2 >> 0) << 1) & 0x6
        val_21 = (base_21 | extra_21) & 0x7
        zeros_i32 = tl.where(cond_21, tl.cast(val_21, tl.int32), zeros_i32)

        # --- case 4: idx in [22..31] (shifts 2,5,8,...,29 in word2)
        cond_22_31 = idx >= 22
        j_22_31 = idx - 22
        shift_22_31 = 2 + 3 * j_22_31
        val_22_31 = (word2 >> shift_22_31) & 0x7
        zeros_i32 = tl.where(cond_22_31, tl.cast(val_22_31, tl.int32), zeros_i32)

        zeros = zeros_i32  # int32 3-bit value
    else:
        # original 2/4/8bit path unchanged
        qzeros = tl.load(
            qzeros_ptr + (qzero_ncols * groups + col_idx // pack_scale),
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        wf_zeros = (col_idx % pack_scale) * bits
        zeros = (qzeros >> wf_zeros) & maxq

    # 4. decode qweight (weights) — also 10-1-10-1-10 format
    if bits == 3:
        tl.static_assert(out_features % 32 == 0, "out_features must be divisible by 32 for 3-bit qweight")

        ROW_BLOCKS: tl.constexpr = 32  # every block is 32 rows
        rows_per_group: tl.constexpr = 3

        block_r = row_idx // ROW_BLOCKS
        idx32_r = row_idx % ROW_BLOCKS

        row_word_base = block_r * rows_per_group

        w0 = tl.load(
            qweight_ptr + (row_word_base + 0) * out_features + col_idx,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        w1 = tl.load(
            qweight_ptr + (row_word_base + 1) * out_features + col_idx,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        w2 = tl.load(
            qweight_ptr + (row_word_base + 2) * out_features + col_idx,
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )

        idxr = idx32_r
        weights_i32 = tl.cast(idxr, tl.int32) & 0

        # --- case 0: idxr in [0..9]
        cond_r_0_9 = idxr <= 9
        shift_r_0_9 = idxr * 3
        val_r_0_9 = (w0 >> shift_r_0_9) & 0x7
        weights_i32 = tl.where(cond_r_0_9, tl.cast(val_r_0_9, tl.int32), weights_i32)

        # --- case 1: idxr == 10
        cond_r_10 = idxr == 10
        base_r_10 = (w0 >> 30) & 0x3
        extra_r_10 = ((w1 >> 0) << 2) & 0x4
        val_r_10 = (base_r_10 | extra_r_10) & 0x7
        weights_i32 = tl.where(cond_r_10, tl.cast(val_r_10, tl.int32), weights_i32)

        # --- case 2: idxr in [11..20]
        cond_r_11_20 = (idxr >= 11) & (idxr <= 20)
        jr_11_20 = idxr - 11
        shift_r_11_20 = 1 + 3 * jr_11_20
        val_r_11_20 = (w1 >> shift_r_11_20) & 0x7
        weights_i32 = tl.where(cond_r_11_20, tl.cast(val_r_11_20, tl.int32), weights_i32)

        # --- case 3: idxr == 21
        cond_r_21 = idxr == 21
        base_r_21 = (w1 >> 31) & 0x1
        extra_r_21 = ((w2 >> 0) << 1) & 0x6
        val_r_21 = (base_r_21 | extra_r_21) & 0x7
        weights_i32 = tl.where(cond_r_21, tl.cast(val_r_21, tl.int32), weights_i32)

        # --- case 4: idxr in [22..31]
        cond_r_22_31 = idxr >= 22
        jr_22_31 = idxr - 22
        shift_r_22_31 = 2 + 3 * jr_22_31
        val_r_22_31 = (w2 >> shift_r_22_31) & 0x7
        weights_i32 = tl.where(cond_r_22_31, tl.cast(val_r_22_31, tl.int32), weights_i32)

        weights = weights_i32
    else:
        # original 2/4/8bit path unchanged
        qweights = tl.load(
            qweight_ptr + (col_idx + out_features * (row_idx // pack_scale)),
            mask=xmask,
            other=0,
            eviction_policy="evict_last",
        )
        wf_weights = (row_idx % pack_scale) * bits
        weights = (qweights >> wf_weights) & maxq

    # 5. Dequantize
    weights = (tl.cast(weights, tl.float32) - tl.cast(zeros, tl.float32)) * scales
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

    # The decoder uses dense offsets for every packed operand and group index.
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    g_idx = g_idx.contiguous()

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=dtype)
    out_dtype = dtype

    cfg = launch_config(qweight.device)
    numels = out.numel()
    grid = (triton.cdiv(numels, cfg.x_block),)

    with torch.xpu.device(qweight.device) if HAS_XPU else torch.cuda.device(qweight.device):
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
            X_BLOCK=cfg.x_block,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return out


def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False, bias=None):
    W = dequant(input.dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    weight = W.t() if transpose else W
    # Match every other low-precision inference kernel: keep the reduction in
    # FP32 and round once at the epilogue. A plain `input @ W` accumulates in
    # the activation dtype and loses ~1 ulp per output element versus the Torch
    # backend, which uses the same FP32-reduce helper.
    return matmul_fp32_reduce(input, weight, bias=bias)

class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="xpu" if HAS_XPU else "cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, bias=None):
        # Declared source leaves can be column views of a fused allocation.
        # The raw-pointer decoder requires compact operands for direct/fallback
        # calls. These packed temporaries live only for this computation.
        qweight, scales, qzeros = qweight.contiguous(), scales.contiguous(), qzeros.contiguous()
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, bias=bias)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits = bits, maxq, pack_bits
        ctx.bias_shape = None if bias is None else bias.shape
        ctx.bias_dtype = None if bias is None else bias.dtype
        return output

    @staticmethod
    @custom_bwd(device_type="xpu" if HAS_XPU else "cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits = ctx.bits, ctx.maxq, ctx.pack_bits
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True)
        grad_bias = None
        if len(ctx.needs_input_grad) > 8 and ctx.needs_input_grad[8]:
            grad_bias = grad_output.float().sum_to_size(ctx.bias_shape).to(ctx.bias_dtype)
        gradients = (grad_input, None, None, None, None, None, None, None, grad_bias)
        return gradients[:len(ctx.needs_input_grad)]
