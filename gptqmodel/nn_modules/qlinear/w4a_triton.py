# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""W4AFP8 input quantization and native FP8 group products.

The E4M3 MMA path needs Ada Lovelace or newer (SM 8.9+). It is shared by the
FP8 lane of every W4A policy, including the GB10 mixed recipes.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


def _launch_device(device: torch.device):
    """Bind a Triton launch to the device that owns the operands.

    Triton launches into the *current* CUDA context, but a checkpoint sharded
    over several devices hands each layer operands on that layer's device.
    Without this guard a layer on cuda:1 fails with "Pointer argument cannot
    be accessed from Triton (cpu tensor?)" while cuda:0 is the current device.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(device)
    return nullcontext()


@triton.jit
def _to_e4m3_rn(value):
    """Round f32 to E4M3 with round-to-nearest-even on every architecture.

    Triton lowers a plain ``.to(tl.float8e4nv)`` on SM 8.9 to a lossy
    fp32 ->(rz) fp16 ->(rn) e4m3 pair, which mis-rounds values just above an
    e4m3 midpoint; SM 9.0+ gets the direct conversion. Emit the direct
    ``cvt.rn.satfinite.e4m3x2.f32`` (available since SM 8.9) so Ada matches
    the Torch oracle bit for bit.
    """
    return tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,r,r",
        [value],
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _token_fp8_quant(X, Q, RowScale, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + row * K + k, k < K, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    scale = tl.where(amax > 0, amax / 448.0, 1.0)
    q = _to_e4m3_rn(tl.minimum(tl.maximum(x / scale, -448.0), 448.0))
    tl.store(Q + row * K + k, q, k < K)
    tl.store(RowScale + row, scale)


@gluon.jit
def _grouped_fp8_gemm(
    A, W, Scales, RowScale, Bias, Out,
    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    BM: gl.constexpr, BN: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]
    )
    rows = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, layout))
    cols = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    total = gl.full((BM, BN), 0.0, gl.float32, layout=layout)
    for group in range(K // 128):
        kk_a = group * 128 + ka
        kk_b = group * 128 + kb
        a = gl.load(A + rows[:, None] * K + kk_a[None, :], rows[:, None] < M)
        b = gl.load(W + kk_b[:, None] * N + cols[None, :], cols[None, :] < N)
        a_mma = gl.convert_layout(a, gl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=4))
        b_mma = gl.convert_layout(b, gl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=4))
        partial = gl.full((BM, BN), 0.0, gl.float32, layout=mma_layout)
        partial = mma_v2(a_mma, b_mma, partial)
        partial = gl.convert_layout(partial, layout)
        group_scale = gl.load(Scales + group * N + cols, cols < N, other=0).to(gl.float32)
        total += partial * group_scale[None, :]
    token_scale = gl.load(RowScale + rows, rows < M, other=1.0)
    total *= token_scale[:, None]
    if HAS_BIAS:
        bias = gl.load(Bias + cols, cols < N, other=0).to(gl.float32)
        total += bias[None, :]
    gl.store(Out + rows[:, None] * N + cols[None, :], total, (rows[:, None] < M) & (cols[None, :] < N))


def fp8_linear(x: torch.Tensor, weight_e4m3: torch.Tensor, scales: torch.Tensor,
               bias: torch.Tensor | None) -> torch.Tensor:
    """Run per-token E4M3 quantization and group-aware native FP8 MMA."""
    if x.device.type != "cuda" or weight_e4m3.device != x.device:
        raise ValueError("W4AFP8 needs activation and prepared weights on the same CUDA device.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W4AFP8 supports FP16 and BF16 inputs.")
    n, k = weight_e4m3.shape[1], weight_e4m3.shape[0]
    if x.shape[-1] != k or k % 128:
        raise ValueError("W4AFP8 requires K divisible by 128 and matching the weight cache.")
    shape = x.shape[:-1] + (n,)
    rows = x.numel() // k
    if rows == 0:
        return torch.empty(shape, device=x.device, dtype=x.dtype)
    x2 = x.contiguous().reshape(rows, k)
    q, row_scale = fp8_pack(x2)
    return fp8_linear_prepacked(q, row_scale, weight_e4m3, scales, bias, x.dtype).reshape(shape)


def fp8_pack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode CUDA [rows, K] with one E4M3 scale per row."""
    if x.ndim != 2 or x.device.type != "cuda":
        raise ValueError("FP8 packing expects a CUDA [rows, K] tensor.")
    rows, k = x.shape
    q = torch.empty((rows, k), device=x.device, dtype=torch.float8_e4m3fn)
    row_scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows:
        with _launch_device(x.device):
            _token_fp8_quant[(rows,)](x.contiguous(), q, row_scale, k, triton.next_power_of_2(k), num_warps=4)
    return q, row_scale


def fp8_linear_prepacked(q: torch.Tensor, row_scale: torch.Tensor,
                         weight_e4m3: torch.Tensor, scales: torch.Tensor,
                         bias: torch.Tensor | None, output_dtype: torch.dtype) -> torch.Tensor:
    """Consume encoded FP8 and its token scales without re-quantization."""
    if q.dtype != torch.float8_e4m3fn or q.ndim != 2 or q.shape[1] != weight_e4m3.shape[0]:
        raise ValueError("FP8 operand has the wrong dtype or K dimension.")
    if (q.device.type != "cuda" or q.device != weight_e4m3.device or
            q.device != row_scale.device or q.device != scales.device or
            (bias is not None and bias.device != q.device)):
        raise ValueError("FP8 codes, scales, weights, and bias must share one CUDA device.")
    rows, k = q.shape
    n = weight_e4m3.shape[1]
    if row_scale.shape != (rows,) or row_scale.dtype != torch.float32 or k % 128:
        raise ValueError("FP8 operand requires one FP32 scale per row and K divisible by 128.")
    output = torch.empty((rows, n), device=q.device, dtype=output_dtype)
    if not rows:
        return output
    tile_n = 32 if n % 64 else 64
    with _launch_device(q.device):
        _grouped_fp8_gemm[(triton.cdiv(rows, 16), triton.cdiv(n, tile_n))](
            q, weight_e4m3, scales, row_scale, bias, output,
            rows, n, k, bias is not None, 16, tile_n, num_warps=4,
        )
    return output
