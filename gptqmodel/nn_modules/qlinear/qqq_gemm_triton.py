# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fused W4A8 GEMM for the portable QQQ kernel on CUDA and ROCm.

``QQQTorchLinear`` used to materialise the whole INT8 weight plane for every
forward.  That plane is derived from the packed INT4 codes and the group
scales, so the dequantisation work and the global-memory round trip dominated
small-batch decode: the module was tens of times slower than dense FP16.

This kernel decodes the packed INT4 words inside the GEMM, applies the group
scale and the INT8 rounding in registers, and accumulates with the INT8
activation.  The packed weight is read once per output tile; no expanded
weight plane is written or retained.

The kernel consumes a *logical* packed layout: ``packed_weight[k // 8, n]``
holds the codes for ``k = 8 * (k // 8) + lane`` in nibble ``lane``.  The
module builds that layout once from the checkpoint words; it is the same
4-bits-per-weight storage, not an expanded copy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from gptqmodel.utils.triton_jit import jit as triton_jit


#: Block sizes measured on the two families this path ships for.  ``BLOCK_K``
#: is capped at the group size so one K tile never spans two quantisation
#: groups.
_BLOCK_M_SMALL = 16
_BLOCK_M_LARGE = 32
_BLOCK_N = 64
_BLOCK_K = 128


@triton_jit(do_not_specialize=["M"])
def _qqq_w4a8_gemm_kernel(
    a_ptr,
    packed_ptr,
    s_group_ptr,
    s_channel_ptr,
    s1_ptr,
    bias_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` output tile.

    ``GROUP_SIZE`` is the grouped QQQ group size, or ``0`` for the
    per-channel (``group_size == in_features``) representation.  Grouped
    codes are biased by eight; per-channel codes are signed two's-complement
    nibbles.
    """

    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row < M
    col_mask = col < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    k_offsets = tl.arange(0, BLOCK_K)
    word_offsets = tl.arange(0, BLOCK_K // 8)
    shifts = (tl.arange(0, 8) * 4)[None, :, None]

    for k_start in range(0, K, BLOCK_K):
        # [BLOCK_K / 8, BLOCK_N] packed words -> [BLOCK_K, BLOCK_N] codes.
        word_rows = k_start // 8 + word_offsets
        words = tl.load(
            packed_ptr + word_rows[:, None] * N + col[None, :],
            mask=(word_rows[:, None] < K // 8) & col_mask[None, :],
            other=0,
        )
        codes = tl.reshape(
            (words[:, None, :] >> shifts) & 0xF,
            (BLOCK_K, BLOCK_N),
        )

        if GROUP_SIZE > 0:
            # Round to the deployed INT8 weight grid exactly like the portable
            # Torch path: round first, then clamp, so an exact half-step above
            # 127 cannot wrap through the INT8 cast.
            group_scale = tl.load(
                s_group_ptr + (k_start // GROUP_SIZE) * N + col,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            weights = (codes - 8).to(tl.float32) * group_scale[None, :]
            weights = libdevice.nearbyint(weights)
            # A non-finite scale makes the checkpoint invalid.  The native and
            # CPU QQQ kernels fold those lanes to zero instead of leaving the
            # float-to-INT8 cast undefined; keep the same deterministic result.
            weights = tl.where(libdevice.isnan(weights), 0.0, weights)
            weights = tl.minimum(tl.maximum(weights, -128.0), 127.0)
            weights = weights.to(tl.int8)
        else:
            # Per-channel codes are signed nibbles scaled by 16.
            weights = (((codes ^ 8) - 8) * 16).to(tl.int8)

        activations = tl.load(
            a_ptr + row[:, None] * K + (k_start + k_offsets)[None, :],
            mask=row_mask[:, None] & ((k_start + k_offsets)[None, :] < K),
            other=0,
        )
        accumulator = tl.dot(activations, weights, accumulator, out_dtype=tl.int32)

    # The Torch forward applies the two scales through the historical
    # left-to-right product and only reverses the association for elements
    # whose first intermediate overflowed (#5608).  Keep that ordering and
    # disable FP fusion on the launch so the rounding matches.
    accum = accumulator.to(tl.float32)
    s1 = tl.load(s1_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
    s_channel = tl.load(s_channel_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
    scaled = (accum * s1[:, None]) * s_channel[None, :]
    alternate = (accum * s_channel[None, :]) * s1[:, None]
    result = tl.where(libdevice.finitef(scaled), scaled, alternate)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + col, mask=col_mask, other=0.0).to(tl.float32)
        result = result + bias[None, :]
    tl.store(
        out_ptr + row[:, None] * N + col[None, :],
        result,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _validate(
    quant_a: torch.Tensor,
    packed_weight: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    s1: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int]:
    if quant_a.device.type != "cuda":
        raise ValueError("QQQ fused GEMM requires a CUDA-family device.")
    if quant_a.dtype is not torch.int8 or quant_a.ndim != 2:
        raise ValueError("QQQ fused GEMM expects a 2D INT8 activation.")
    if not quant_a.is_contiguous():
        raise ValueError("QQQ fused GEMM expects a contiguous activation.")
    rows, in_features = quant_a.shape
    out_features = out.shape[1] if out.ndim == 2 else -1
    if out.ndim != 2 or out.shape[0] != rows or out_features <= 0:
        raise ValueError("QQQ fused GEMM output has the wrong shape.")
    if out.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QQQ fused GEMM writes FP16 or BF16 output.")
    if not out.is_contiguous():
        raise ValueError("QQQ fused GEMM expects a contiguous output.")
    if packed_weight.dtype is not torch.int32 or packed_weight.ndim != 2:
        raise ValueError("QQQ fused GEMM expects INT32 packed weight words.")
    if packed_weight.shape != (in_features // 8, out_features):
        raise ValueError("QQQ packed weight does not match the GEMM shape.")
    if not packed_weight.is_contiguous():
        raise ValueError("QQQ packed weight must be contiguous.")
    if s_channel.dtype is not torch.float32 or s_channel.shape != (out_features,):
        raise ValueError("QQQ channel scales must be one FP32 value per output.")
    if not s_channel.is_contiguous():
        raise ValueError("QQQ channel scales must be contiguous.")
    if s1.dtype is not torch.float32 or s1.shape not in ((rows, 1), (rows,)):
        raise ValueError("QQQ activation scales must be FP32 per row.")
    if not s1.is_contiguous():
        raise ValueError("QQQ activation scales must be contiguous.")
    if group_size:
        expected = (in_features // group_size, out_features)
        if group_size != _BLOCK_K:
            raise ValueError("QQQ fused GEMM requires the 128-element group size.")
        if s_group.shape != expected or not s_group.is_contiguous():
            raise ValueError("QQQ group scales do not match the GEMM shape.")
    elif s_group.numel() != 0:
        raise ValueError("Per-channel QQQ must not carry group scales.")
    if bias is not None:
        if bias.shape != (out_features,) or bias.device != quant_a.device:
            raise ValueError("QQQ bias must be one value per output on-device.")
        if bias.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("QQQ bias must be FP16, BF16 or FP32.")
        if not bias.is_contiguous():
            raise ValueError("QQQ bias must be contiguous.")
    if (
        packed_weight.device != quant_a.device
        or s_channel.device != quant_a.device
        or s1.device != quant_a.device
        or (group_size and s_group.device != quant_a.device)
    ):
        raise ValueError("QQQ GEMM operands must share one device.")
    if in_features % 8:
        raise ValueError("QQQ in_features must be divisible by 8.")
    return rows, in_features, out_features


def qqq_gemm(
    quant_a: torch.Tensor,
    packed_weight: torch.Tensor,
    s_group: torch.Tensor,
    s_channel: torch.Tensor,
    s1: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Launch the fused QQQ W4A8 GEMM and return ``out``.

    ``group_size`` is the QQQ group size, or ``0`` when the module uses the
    per-channel representation (``group_size == in_features``).  Only the
    supported QQQ group size (128) and the per-channel form reach this
    function; other group sizes keep the portable Torch path.
    """

    rows, in_features, out_features = _validate(
        quant_a,
        packed_weight,
        s_group,
        s_channel,
        s1,
        bias,
        out,
        group_size,
    )
    if rows == 0:
        return out

    block_m = _BLOCK_M_SMALL if rows <= _BLOCK_M_SMALL else _BLOCK_M_LARGE
    block_k = _BLOCK_K if group_size else min(_BLOCK_K, in_features)
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(out_features, _BLOCK_N),
    )
    with torch.cuda.device(quant_a.device):
        _qqq_w4a8_gemm_kernel[grid](
            quant_a,
            packed_weight,
            s_group,
            s_channel,
            s1,
            bias,
            out,
            rows,
            out_features,
            in_features,
            group_size,
            bias is not None,
            block_m,
            _BLOCK_N,
            block_k,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["qqq_gemm"]
