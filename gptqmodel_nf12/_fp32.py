# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""FP32 inference from NF12 weights for routers and score projections.

The upstream NF12 MMA operator accepts BF16 activations and returns BF16. A
router needs its checkpoint's BF16 weight but keeps FP32 activations, reduction,
and scores. This kernel decodes each weight fragment in registers and performs
FP32 multiply-accumulate without materializing a dense weight matrix.
"""

from __future__ import annotations

import threading
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .api import NF12Weight


_gate_views: weakref.WeakKeyDictionary[NF12Weight, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
    weakref.WeakKeyDictionary()
)
_gate_lock = threading.Lock()


def prepare_fp32_gate(weight: NF12Weight) -> None:
    """Prepare compressed K-contiguous planes for 128/384x5120 B1 routers.

    This is a load-time operation. The temporary BF16 decode dies before
    inference; the weakly owned cache retains only 12-bit packed planes and
    escaped BF16 groups. The ordinary tiled NF12 layout remains available for
    all other projection geometries.
    """
    n, k = weight.shape
    if n not in (128, 384) or k != 5120 or weight.device.type != "cuda":
        raise ValueError("prepare_fp32_gate requires a CUDA NF12 [128 or 384,5120] weight")
    with _gate_lock:
        if weight in _gate_views:
            return
        dense = weight.decode()
        bits = (dense.view(torch.int16).to(torch.int32) & 65535).view(n, 640, 8)
        exponent = (bits >> 7) & 255
        sign = bits >> 15
        offset = (exponent - 2 * weight.exp_base) >> 1
        meta_bytes = ((sign[:, :, 0::2] << 7) | offset[:, :, 0::2] |
                      (sign[:, :, 1::2] << 6) | (offset[:, :, 1::2] << 3))
        meta = (meta_bytes[:, :, 0] | (meta_bytes[:, :, 1] << 8) |
                (meta_bytes[:, :, 2] << 16) | (meta_bytes[:, :, 3] << 24)).contiguous()
        low = (bits & 255).to(torch.uint8).view(n, 5120).contiguous()
        valid = (((exponent >= 2 * weight.exp_base) &
                  (exponent <= 2 * weight.exp_base + 15)).all(-1) & (meta != 0))
        escaped_groups = ~valid
        escapes = bits[escaped_groups].to(torch.int16).contiguous()
        escaped_count = escapes.shape[0]
        if escaped_count:
            low_words = low.view(torch.int32).view(n, 640, 2)
            low_words[:, :, 0][escaped_groups] = torch.arange(
                escaped_count, dtype=torch.int32, device=weight.device)
            low_words[:, :, 1][escaped_groups] = 0
            meta[escaped_groups] = 0
        del dense, bits, exponent, sign, offset, meta_bytes, valid, escaped_groups
        _gate_views[weight] = (meta, low, escapes)


def _get_gate_view(weight: NF12Weight):
    with _gate_lock:
        return _gate_views.get(weight)


@triton.jit
def _linear_fp32_gate_planes(
    x_ptr, meta_ptr, low_ptr, escapes_ptr, bias_ptr, out_ptr,
    n: tl.constexpr, exp_base: tl.constexpr, has_bias: tl.constexpr, m: tl.constexpr,
    block_k: tl.constexpr,
):
    """One router row per CTA; share its packed decode across six inputs."""
    output_row = tl.program_id(0)
    cols = tl.arange(0, block_k)
    acc0 = tl.full((block_k,), 0, tl.float32)
    if m == 6:
        acc1 = tl.full((block_k,), 0, tl.float32)
        acc2 = tl.full((block_k,), 0, tl.float32)
        acc3 = tl.full((block_k,), 0, tl.float32)
        acc4 = tl.full((block_k,), 0, tl.float32)
        acc5 = tl.full((block_k,), 0, tl.float32)
    low_u32 = low_ptr.to(tl.pointer_type(tl.uint32))

    for block in range(tl.cdiv(5120, block_k)):
        col = block * block_k + cols
        valid = col < 5120
        meta = tl.load(meta_ptr + output_row * 640 + col // 8, valid, other=0).to(tl.uint32)
        low = tl.load(low_ptr + output_row * 5120 + col, valid, other=0).to(tl.uint32)
        slot = col % 8
        meta_byte = (meta >> (8 * (slot // 2))) & 255
        high_even = (meta_byte & 0x87) + exp_base
        high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 7)) + exp_base
        high = tl.where(slot % 2 == 0, high_even, high_odd)
        index = tl.load(low_u32 + (output_row * 5120 + (col // 8) * 8) // 4,
                        valid & (meta == 0), other=0)
        escaped = tl.load(escapes_ptr + index * 8 + slot, valid & (meta == 0),
                          other=0).to(tl.uint16, bitcast=True)
        bits = tl.where(meta == 0, escaped.to(tl.uint32), (high << 8) | low)
        w = (bits << 16).to(tl.float32, bitcast=True)
        x0 = tl.load(x_ptr + col, valid, other=0)
        acc0 = tl.fma(x0, w, acc0)
        if m == 6:
            x1 = tl.load(x_ptr + 5120 + col, valid, other=0)
            x2 = tl.load(x_ptr + 10240 + col, valid, other=0)
            x3 = tl.load(x_ptr + 15360 + col, valid, other=0)
            x4 = tl.load(x_ptr + 20480 + col, valid, other=0)
            x5 = tl.load(x_ptr + 25600 + col, valid, other=0)
            acc1 = tl.fma(x1, w, acc1)
            acc2 = tl.fma(x2, w, acc2)
            acc3 = tl.fma(x3, w, acc3)
            acc4 = tl.fma(x4, w, acc4)
            acc5 = tl.fma(x5, w, acc5)

    bias = tl.load(bias_ptr + output_row) if has_bias else 0.0
    tl.store(out_ptr + output_row, tl.sum(acc0, 0) + bias)
    if m == 6:
        tl.store(out_ptr + n + output_row, tl.sum(acc1, 0) + bias)
        tl.store(out_ptr + 2 * n + output_row, tl.sum(acc2, 0) + bias)
        tl.store(out_ptr + 3 * n + output_row, tl.sum(acc3, 0) + bias)
        tl.store(out_ptr + 4 * n + output_row, tl.sum(acc4, 0) + bias)
        tl.store(out_ptr + 5 * n + output_row, tl.sum(acc5, 0) + bias)


@triton.jit
def _linear_fp32(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, rows_ptr, out_ptr,
    n: tl.constexpr, output_n: tl.constexpr, k: tl.constexpr, m: tl.constexpr,
    exp_base: tl.constexpr, block_m: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr, indexed: tl.constexpr,
):
    output_row = tl.program_id(0)
    # Compare before narrowing: an out-of-range int64 ID can otherwise wrap
    # into a valid int32 row and read the wrong packed weight.
    row_id = tl.load(rows_ptr + output_row) if indexed else output_row
    valid_row = (row_id >= 0) & (row_id < n)
    row_n = row_id.to(tl.int32)
    row_m = tl.program_id(1) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    accum = tl.full((block_m, block_k), 0, tl.float32)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))

    for block in range(tl.cdiv(k, block_k)):
        col = block * block_k + cols
        valid = (col < k) & valid_row
        # One 16x64 super-atom has 1,536 bytes. Its three 512-byte planes
        # hold a meta word and two low-byte words per lane and 16-column atom.
        super_atom = ((row_n // 16) * (k // 64) + col // 64) * 1536
        atom = (col % 64) // 16
        lane = (row_n % 8) * 4 + ((col % 16) % 8) // 2
        slot = ((row_n % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
        meta_offset = super_atom + 16 * lane + 4 * atom
        low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                      + 16 * lane + 8 * (atom % 2))
        meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
        low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
        low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
        low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
        meta_byte = (meta >> (8 * (slot // 2))) & 255
        high_even = (meta_byte & 0x87) + exp_base
        high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
        high = tl.where(slot % 2 == 0, high_even, high_odd)
        ordinary = (high << 8) | low
        # A zero meta word is an escaped group; low0 is its table index.
        escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                          valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
        bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
        weight = (bits << 16).to(tl.float32, bitcast=True)
        activation = tl.load(x_ptr + row_m[:, None] * k + col[None, :],
                             (row_m[:, None] < m) & valid[None, :], other=0)
        accum = tl.fma(activation, weight[None, :], accum)

    result = tl.sum(accum, axis=1)
    if has_bias:
        result += tl.load(bias_ptr + row_n, valid_row, other=0)
    tl.store(out_ptr + row_m * output_n + output_row, result, row_m < m)


@triton.jit
def _linear_fp32_six_b1(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, out_ptr,
    n: tl.constexpr, k: tl.constexpr, exp_base: tl.constexpr,
    has_bias: tl.constexpr,
):
    """One weight decode and six independent ordinary-B1 reduction trees."""
    output_row = tl.program_id(0)
    row = tl.arange(0, 1)
    cols = tl.arange(0, 256)
    acc0 = tl.full((1, 256), 0, tl.float32)
    acc1 = tl.full((1, 256), 0, tl.float32)
    acc2 = tl.full((1, 256), 0, tl.float32)
    acc3 = tl.full((1, 256), 0, tl.float32)
    acc4 = tl.full((1, 256), 0, tl.float32)
    acc5 = tl.full((1, 256), 0, tl.float32)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))

    for block in range(tl.cdiv(k, 256)):
        col = block * 256 + cols
        valid = col < k
        super_atom = ((output_row // 16) * (k // 64) + col // 64) * 1536
        atom = (col % 64) // 16
        lane = (output_row % 8) * 4 + ((col % 16) % 8) // 2
        slot = ((output_row % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
        meta_offset = super_atom + 16 * lane + 4 * atom
        low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                      + 16 * lane + 8 * (atom % 2))
        meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
        low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
        low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
        low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
        meta_byte = (meta >> (8 * (slot // 2))) & 255
        high_even = (meta_byte & 0x87) + exp_base
        high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
        high = tl.where(slot % 2 == 0, high_even, high_odd)
        ordinary = (high << 8) | low
        escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                          valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
        bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
        weight = (bits << 16).to(tl.float32, bitcast=True)[None, :]
        x0 = tl.load(x_ptr + row[:, None] * k + col[None, :], valid[None, :], other=0)
        x1 = tl.load(x_ptr + (row[:, None] + 1) * k + col[None, :], valid[None, :], other=0)
        x2 = tl.load(x_ptr + (row[:, None] + 2) * k + col[None, :], valid[None, :], other=0)
        x3 = tl.load(x_ptr + (row[:, None] + 3) * k + col[None, :], valid[None, :], other=0)
        x4 = tl.load(x_ptr + (row[:, None] + 4) * k + col[None, :], valid[None, :], other=0)
        x5 = tl.load(x_ptr + (row[:, None] + 5) * k + col[None, :], valid[None, :], other=0)
        acc0 = tl.fma(x0, weight, acc0)
        acc1 = tl.fma(x1, weight, acc1)
        acc2 = tl.fma(x2, weight, acc2)
        acc3 = tl.fma(x3, weight, acc3)
        acc4 = tl.fma(x4, weight, acc4)
        acc5 = tl.fma(x5, weight, acc5)

    bias = tl.load(bias_ptr + output_row) if has_bias else 0.0
    tl.store(out_ptr + row * n + output_row, tl.sum(acc0, axis=1) + bias)
    tl.store(out_ptr + (row + 1) * n + output_row, tl.sum(acc1, axis=1) + bias)
    tl.store(out_ptr + (row + 2) * n + output_row, tl.sum(acc2, axis=1) + bias)
    tl.store(out_ptr + (row + 3) * n + output_row, tl.sum(acc3, axis=1) + bias)
    tl.store(out_ptr + (row + 4) * n + output_row, tl.sum(acc4, axis=1) + bias)
    tl.store(out_ptr + (row + 5) * n + output_row, tl.sum(acc5, axis=1) + bias)


@triton.jit
def _linear_fp32_short(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, rows_ptr, out_ptr,
    n: tl.constexpr, output_n: tl.constexpr, k: tl.constexpr,
    exp_base: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr, indexed: tl.constexpr,
):
    """One CTA decodes several short-K output rows and reuses one input row."""
    output_rows = tl.program_id(0) * block_n + tl.arange(0, block_n)
    input_row = tl.program_id(1)
    row_id = (tl.load(rows_ptr + output_rows, output_rows < output_n, other=-1)
              if indexed else output_rows)
    valid_row = (output_rows < output_n) & (row_id >= 0) & (row_id < n)
    row_n = row_id.to(tl.int32)[:, None]
    col = tl.arange(0, block_k)[None, :]
    valid = valid_row[:, None] & (col < k)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))
    super_atom = ((row_n // 16) * (k // 64) + col // 64) * 1536
    atom = (col % 64) // 16
    lane = (row_n % 8) * 4 + ((col % 16) % 8) // 2
    slot = ((row_n % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
    meta_offset = super_atom + 16 * lane + 4 * atom
    low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                  + 16 * lane + 8 * (atom % 2))
    meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
    low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
    low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
    low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
    meta_byte = (meta >> (8 * (slot // 2))) & 255
    high_even = (meta_byte & 0x87) + exp_base
    high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
    high = tl.where(slot % 2 == 0, high_even, high_odd)
    ordinary = (high << 8) | low
    escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                      valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
    bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
    weight = (bits << 16).to(tl.float32, bitcast=True)
    activation = tl.load(x_ptr + input_row * k + col, col < k, other=0)
    result = tl.sum(tl.fma(weight, activation, 0.), axis=1)
    if has_bias:
        result += tl.load(bias_ptr + row_id, valid_row, other=0)
    tl.store(out_ptr + input_row * output_n + output_rows, result, output_rows < output_n)


@triton.jit
def _linear_bf16_fp32_tc(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, out_ptr,
    n: tl.constexpr, k: tl.constexpr, m: tl.constexpr,
    exp_base: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Decode a short-K output tile once for an FP32-accumulating BF16 MMA."""
    output_rows = tl.program_id(0) * block_n + tl.arange(0, block_n)
    input_rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    row_n = output_rows[:, None]
    col = tl.arange(0, block_k)[None, :]
    valid = (row_n < n) & (col < k)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))
    super_atom = ((row_n // 16) * (k // 64) + col // 64) * 1536
    atom = (col % 64) // 16
    lane = (row_n % 8) * 4 + ((col % 16) % 8) // 2
    slot = ((row_n % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
    meta_offset = super_atom + 16 * lane + 4 * atom
    low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                  + 16 * lane + 8 * (atom % 2))
    meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
    low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
    low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
    low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
    meta_byte = (meta >> (8 * (slot // 2))) & 255
    high_even = (meta_byte & 0x87) + exp_base
    high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
    high = tl.where(slot % 2 == 0, high_even, high_odd)
    ordinary = (high << 8) | low
    escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                      valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
    bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
    weight = (bits << 16).to(tl.float32, bitcast=True).to(tl.bfloat16)
    activation = tl.load(x_ptr + input_rows[:, None] * k + col,
                         (input_rows[:, None] < m) & (col < k), other=0)
    result = tl.dot(activation, tl.trans(weight))
    if has_bias:
        result += tl.load(bias_ptr + output_rows, output_rows < n, other=0)[None, :]
    tl.store(out_ptr + input_rows[:, None] * n + output_rows[None, :], result,
             (input_rows[:, None] < m) & (output_rows[None, :] < n))


def linear_bf16_fp32(x: torch.Tensor, weight: NF12Weight,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    """Project BF16 activations from packed weights with FP32 output.

    Short-K full heads use BF16 Tensor Cores with FP32 accumulation. Other
    shapes use the general packed FP32 path without retaining a dense weight.
    """
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim < 1:
        raise ValueError("gptqmodel_nf12.linear_bf16_fp32 needs a CUDA BF16 activation")
    if x.shape[-1] != weight.shape[1] or x.device != weight.device:
        raise ValueError("activation shape/device does not match NF12 weight")
    n, k = weight.shape
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != (n,)
                             or bias.device != x.device):
        raise ValueError("bias must be FP32 [out_features] on the activation device")
    x2 = x.contiguous().view(-1, k)
    m = x2.shape[0]
    if m <= 1 or k > 512 or n < 1024 or m > 16:
        return linear_fp32(x.float(), weight, bias)
    out = torch.empty((*x.shape[:-1], n), dtype=torch.float32, device=x.device)
    if m:
        _linear_bf16_fp32_tc[(triton.cdiv(n, 16), triton.cdiv(m, 16))](
            x2, weight.packed, weight.escapes, bias if bias is not None else x2,
            out, n, k, m, weight.exp_base, 16, triton.next_power_of_2(k),
            bias is not None, num_warps=8,
        )
    return out


@triton.jit
def _linear_fp32_prefill_gate_tc(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    """Use Ampere Tensor Cores with FP32 inputs and exact BF16 weight values."""
    input_rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    output_rows = tl.program_id(1) * block_n + tl.arange(0, block_n)
    depth = tl.arange(0, block_k)
    acc = tl.full((block_m, block_n), 0, tl.float32)
    for block in range(tl.cdiv(k, block_k)):
        col = block * block_k + depth
        activation = tl.load(
            x_ptr + input_rows[:, None] * k + col[None, :],
            (input_rows[:, None] < m) & (col[None, :] < k), other=0,
        )
        weight = tl.load(
            weight_ptr + output_rows[:, None] * k + col[None, :],
            (output_rows[:, None] < n) & (col[None, :] < k), other=0,
        )
        acc = tl.dot(activation, tl.trans(weight).to(tl.float32), acc,
                     input_precision="tf32x3")
    if has_bias:
        acc += tl.load(bias_ptr + output_rows, output_rows < n, other=0)[None, :]
    tl.store(out_ptr + input_rows[:, None] * n + output_rows[None, :], acc,
             (input_rows[:, None] < m) & (output_rows[None, :] < n))


def _linear_fp32_prefill_gate(x: torch.Tensor, weight: NF12Weight,
                              bias: torch.Tensor | None) -> torch.Tensor:
    """Decode one BF16 gate only for this call, then run FP32 Tensor Core GEMM."""
    n, k = weight.shape
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), dtype=torch.float32, device=x.device)
    dense = weight.decode()
    try:
        _linear_fp32_prefill_gate_tc[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            x, dense, bias if bias is not None else x, out,
            m, n, k, bias is not None, 32, 64, 64,
            num_warps=4, num_stages=3,
        )
    finally:
        del dense
    return out


def linear_fp32(x: torch.Tensor, weight: NF12Weight,
                bias: torch.Tensor | None = None,
                rows: torch.Tensor | None = None, *, _staged_b1: bool = False) -> torch.Tensor:
    """FP32 ``x @ weight.T + bias`` directly from packed NF12 storage.

    This is intended for small-row inference projections such as MoE routers,
    compressor scores, and Markov correction. It returns FP32 and never holds
    a persistent BF16 or FP32 weight copy. The reduction is numerically close
    to dense FP32 GEMM but may use a different summation order.
    """
    if not x.is_cuda or x.dtype != torch.float32 or x.ndim < 1:
        raise ValueError("gptqmodel_nf12.linear_fp32 needs a CUDA FP32 activation")
    if x.shape[-1] != weight.shape[1] or x.device != weight.device:
        raise ValueError("activation shape/device does not match NF12 weight")
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != (weight.shape[0],)
                             or bias.device != x.device):
        raise ValueError("bias must be FP32 [out_features] on the activation device")
    if rows is not None and (rows.dtype not in (torch.int32, torch.int64) or rows.ndim != 1
                             or rows.device != x.device):
        raise ValueError("rows must be a 1-D CUDA int32/int64 tensor")
    x2 = x.contiguous().view(-1, weight.shape[1])
    m, k = x2.shape
    n = weight.shape[0]
    output_n = n if rows is None else rows.numel()
    if _staged_b1 and (rows is not None or not 1 <= m <= 8 or
                       (k <= 512 and output_n >= 1024)):
        raise ValueError("linear_fp32_staged_b1 requires 1..8 full rows on the ordinary long-K B1 path")
    # The per-output-row packed kernel wins at decode sizes. On Ampere, the
    # main router uses a call-local BF16 decode and accurate TF32x3 Tensor Core
    # GEMM for prefill. Other prefill shapes use a call-local FP32 cuBLAS GEMM.
    # Neither path retains a dense weight between calls.
    if not _staged_b1 and rows is None and m and (m >= 256 or (m >= 16 and n * k >= 1_000_000)):
        if (n, k) == (384, 5120) and m >= 384 and torch.version.hip is None and (
                torch.cuda.get_device_capability(x.device)[0] >= 8):
            return _linear_fp32_prefill_gate(x2, weight, bias).view(*x.shape[:-1], n)
        dense = weight.decode().float()
        result = F.linear(x, dense, bias)
        del dense
        return result
    out = torch.empty((*x.shape[:-1], output_n), dtype=torch.float32, device=x.device)
    if m and output_n:
        gate_view = (_get_gate_view(weight) if rows is None and n in (128, 384) and k == 5120
                     and (m == 1 or (m == 6 and _staged_b1)) else None)
        if gate_view is not None:
            meta, low, escapes = gate_view
            _linear_fp32_gate_planes[(n,)](
                x2, meta, low, escapes, bias if bias is not None else x2,
                out, n, weight.exp_base, bias is not None, m, 1024, num_warps=4,
            )
        elif k <= 512 and output_n >= 1024 and m == 1:
            # Full short-K heads benefit from eight output rows per CTA;
            # indexed shortlists retain four to limit irregular gathers.
            block_n = 4 if rows is not None else 8
            _linear_fp32_short[(triton.cdiv(output_n, block_n), m)](
                x2, weight.packed, weight.escapes, bias if bias is not None else x2,
                rows if rows is not None else x2, out,
                n, output_n, k, weight.exp_base, block_n, triton.next_power_of_2(k),
                bias is not None, rows is not None,
                num_warps=4,
            )
        else:
            if _staged_b1 and m == 6 and k > 512:
                _linear_fp32_six_b1[(n,)](
                    x2, weight.packed, weight.escapes, bias if bias is not None else x2,
                    out, n, k, weight.exp_base, bias is not None, num_warps=4,
                )
            else:
                block_m = 1 if _staged_b1 else min(triton.next_power_of_2(m), 8)
                _linear_fp32[(output_n, triton.cdiv(m, block_m))](
                    x2, weight.packed, weight.escapes, bias if bias is not None else x2,
                    rows if rows is not None else x2, out,
                    n, output_n, k, m, weight.exp_base, block_m, 256,
                    bias is not None, rows is not None,
                    num_warps=4,
                )
    return out


def linear_fp32_staged_b1(x: torch.Tensor, weight: NF12Weight,
                          bias: torch.Tensor | None = None) -> torch.Tensor:
    """Fuse FP32 staged rows while retaining each ordinary B1 reduction tree."""
    return linear_fp32(x, weight, bias, _staged_b1=True)
