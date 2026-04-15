# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache

import torch

from ..utils.env import env_flag


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised via availability gating.
    triton = None
    tl = None


# Restrict the FP8-output path to Ada/Hopper-or-newer CUDA devices where
# Triton's FP8 element type matches the fused W4A8 runtime expectations.
_MIN_FP8_CAPABILITY = (8, 9)
_MIN_SCALE_ONLY_CAPABILITY = (8, 0)
_PARTIAL_BLOCK_K = 256
_MIN_TOKEN_ROWS_FOR_TRITON = 64
_MIN_SCALE_ONLY_ROWS_FOR_TRITON = 2048
_MIN_SCALE_ONLY_COLS_FOR_TRITON = 3072
_QUANT_BLOCK_M = 8
_QUANT_BLOCK_K = 128


def triton_input_quant_available() -> bool:
    """Report whether the runtime can launch the Triton FP8 activation quantizer."""

    return (
        triton is not None
        and tl is not None
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(tl, "float8e4nv")
        and not env_flag("GPTQMODEL_DISABLE_TRITON_INPUT_QUANT", default=False)
    )


@lru_cache(maxsize=None)
def _cuda_device_capability(device_index: int | None) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _device_capability(device: torch.device) -> tuple[int, int]:
    return _cuda_device_capability(device.index)


def resolve_triton_fp8_input_quant_mode(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    dynamic: bool,
    strategy: str,
) -> str | None:
    """Return the supported Triton FP8 quantization mode for this tensor, if any."""

    if not triton_input_quant_available():
        return None
    if fp8_dtype is not torch.float8_e4m3fn:
        return None
    if not dynamic or strategy not in {"token", "tensor"}:
        return None
    if x.device.type != "cuda" or not x.is_floating_point():
        return None
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return None
    if x.numel() == 0 or x.ndim == 0 or x.shape[-1] == 0:
        return None
    if not x.is_contiguous() or x.stride(-1) != 1:
        return None

    major, minor = _device_capability(x.device)
    if major > _MIN_FP8_CAPABILITY[0] or (major == _MIN_FP8_CAPABILITY[0] and minor >= _MIN_FP8_CAPABILITY[1]):
        if strategy != "token" or _token_rows_are_large_enough_for_triton(x):
            return "full"
        return None

    if (major > _MIN_SCALE_ONLY_CAPABILITY[0] or (
        major == _MIN_SCALE_ONLY_CAPABILITY[0] and minor >= _MIN_SCALE_ONLY_CAPABILITY[1]
    )) and _scale_only_shape_is_large_enough_for_triton(x):
        return "scale_only"
    return None


def supports_triton_fp8_input_quant(x: torch.Tensor, fp8_dtype: torch.dtype, *, dynamic: bool, strategy: str) -> bool:
    """Gate the fast path to the exact runtime shape used by fused AWQ W4A8."""

    return resolve_triton_fp8_input_quant_mode(x, fp8_dtype, dynamic=dynamic, strategy=strategy) == "full"


@triton.jit
def _rowwise_token_quantize_fp8_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    num_rows,
    num_cols,
    x_stride_row,
    x_stride_col,
    out_stride_row,
    out_stride_col,
    fp8_min,
    fp8_max,
    min_scale,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    if row_idx >= num_rows:
        return

    row_x_ptr = x_ptr + row_idx * x_stride_row
    row_out_ptr = out_ptr + row_idx * out_stride_row

    abs_max = tl.full((), 0.0, dtype=tl.float32)
    for col_start in range(0, num_cols, BLOCK_K):
        col_offsets = col_start + tl.arange(0, BLOCK_K)
        mask = col_offsets < num_cols
        x = tl.load(row_x_ptr + col_offsets * x_stride_col, mask=mask, other=0.0).to(tl.float32)
        abs_max = tl.maximum(abs_max, tl.max(tl.abs(x), axis=0))

    scale = tl.maximum(abs_max / fp8_max, min_scale)
    tl.store(scale_ptr + row_idx, scale)

    for col_start in range(0, num_cols, BLOCK_K):
        col_offsets = col_start + tl.arange(0, BLOCK_K)
        mask = col_offsets < num_cols
        x = tl.load(row_x_ptr + col_offsets * x_stride_col, mask=mask, other=0.0).to(tl.float32)
        scaled = x / scale
        clipped = tl.maximum(tl.minimum(scaled, fp8_max), fp8_min)
        tl.store(row_out_ptr + col_offsets * out_stride_col, clipped.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def _partial_absmax_kernel(
    x_ptr,
    partial_ptr,
    num_rows,
    num_cols,
    x_stride_row,
    x_stride_col,
    partial_stride_row,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    col_offsets = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = col_offsets < num_cols

    x_ptrs = x_ptr + row_idx * x_stride_row + col_offsets * x_stride_col
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    block_absmax = tl.max(tl.abs(x), axis=0)

    tl.store(partial_ptr + row_idx * partial_stride_row + block_idx, block_absmax)


@triton.jit
def _quantize_fp8_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    num_rows,
    num_cols,
    x_stride_row,
    x_stride_col,
    out_stride_row,
    out_stride_col,
    fp8_min,
    fp8_max,
    PER_ROW_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offsets = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(axis=1) * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (row_offsets[:, None] < num_rows) & (col_offsets[None, :] < num_cols)
    x_ptrs = x_ptr + row_offsets[:, None] * x_stride_row + col_offsets[None, :] * x_stride_col
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    if PER_ROW_SCALE:
        scale = tl.load(scale_ptr + row_offsets, mask=row_offsets < num_rows, other=1.0).to(tl.float32)
        scaled = x / scale[:, None]
    else:
        scale = tl.load(scale_ptr).to(tl.float32)
        scaled = x / scale

    clipped = tl.maximum(tl.minimum(scaled, fp8_max), fp8_min)
    out_ptrs = out_ptr + row_offsets[:, None] * out_stride_row + col_offsets[None, :] * out_stride_col
    tl.store(out_ptrs, clipped.to(out_ptr.type.element_ty), mask=mask)


def _token_rows_are_large_enough_for_triton(x: torch.Tensor) -> bool:
    """Skip tiny decode-style launches where Triton overhead dominates the FP8 work."""

    return x.reshape(-1, x.shape[-1]).shape[0] >= _MIN_TOKEN_ROWS_FOR_TRITON


def _scale_only_shape_is_large_enough_for_triton(x: torch.Tensor) -> bool:
    """Use the reduction-only path only where it wins on local sm_80 A100-class GPUs."""

    rows, cols = x.reshape(-1, x.shape[-1]).shape
    return rows >= _MIN_SCALE_ONLY_ROWS_FOR_TRITON and cols >= _MIN_SCALE_ONLY_COLS_FOR_TRITON


def _compute_dynamic_fp8_scale_triton(
    x: torch.Tensor,
    *,
    strategy: str,
    fp8_max: float,
) -> torch.Tensor:
    """Compute the dynamic FP8 scale with Triton reductions and exact PyTorch post-processing."""

    x_2d = x.reshape(-1, x.shape[-1])
    rows, cols = x_2d.shape
    num_blocks = triton.cdiv(cols, _PARTIAL_BLOCK_K)
    partial_absmax = torch.empty((rows, num_blocks), device=x.device, dtype=torch.float32)
    _partial_absmax_kernel[(rows, num_blocks)](
        x_2d,
        partial_absmax,
        rows,
        cols,
        x_2d.stride(0),
        x_2d.stride(1),
        partial_absmax.stride(0),
        BLOCK_K=_PARTIAL_BLOCK_K,
    )

    if strategy == "token":
        abs_max = partial_absmax.amax(dim=1, keepdim=True).reshape(*x.shape[:-1], 1)
    elif strategy == "tensor":
        abs_max = partial_absmax.amax().reshape(())
    else:  # pragma: no cover - public dispatcher rejects unsupported strategies.
        raise ValueError(f"Unsupported activation strategy for Triton FP8 quantization: `{strategy}`.")

    min_scale = torch.full_like(abs_max, 1.0 / (fp8_max * 512.0))
    return torch.maximum(abs_max / fp8_max, min_scale)


def triton_quantize_input_dynamic_fp8(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a CUDA activation tensor to FP8 without materializing full-size FP32 staging buffers."""

    x_2d = x.reshape(-1, x.shape[-1])
    rows, cols = x_2d.shape
    fp8_info = torch.finfo(fp8_dtype)
    fp8_max = float(fp8_info.max)

    if strategy == "token" and _token_rows_are_large_enough_for_triton(x):
        scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
        out = torch.empty_like(x_2d, dtype=fp8_dtype)
        _rowwise_token_quantize_fp8_kernel[(rows,)](
            x_2d,
            scale.reshape(-1),
            out,
            rows,
            cols,
            x_2d.stride(0),
            x_2d.stride(1),
            out.stride(0),
            out.stride(1),
            float(fp8_info.min),
            fp8_max,
            1.0 / (fp8_max * 512.0),
            BLOCK_K=_PARTIAL_BLOCK_K,
        )
        return out.reshape(x.shape), scale.reshape(*x.shape[:-1], 1)

    num_blocks = triton.cdiv(cols, _PARTIAL_BLOCK_K)
    partial_absmax = torch.empty((rows, num_blocks), device=x.device, dtype=torch.float32)
    _partial_absmax_kernel[(rows, num_blocks)](
        x_2d,
        partial_absmax,
        rows,
        cols,
        x_2d.stride(0),
        x_2d.stride(1),
        partial_absmax.stride(0),
        BLOCK_K=_PARTIAL_BLOCK_K,
    )

    if strategy == "token":
        abs_max = partial_absmax.amax(dim=1, keepdim=True)
    elif strategy == "tensor":
        abs_max = partial_absmax.amax().reshape(())
    else:  # pragma: no cover - public dispatcher rejects unsupported strategies.
        raise ValueError(f"Unsupported activation strategy for Triton FP8 quantization: `{strategy}`.")

    min_scale = torch.full_like(abs_max, 1.0 / (fp8_max * 512.0))
    scale = torch.maximum(abs_max / fp8_max, min_scale)

    out = torch.empty_like(x_2d, dtype=fp8_dtype)
    scale_1d = scale.reshape(-1)
    _quantize_fp8_kernel[(triton.cdiv(rows, _QUANT_BLOCK_M), triton.cdiv(cols, _QUANT_BLOCK_K))](
        x_2d,
        scale_1d,
        out,
        rows,
        cols,
        x_2d.stride(0),
        x_2d.stride(1),
        out.stride(0),
        out.stride(1),
        float(fp8_info.min),
        float(fp8_info.max),
        PER_ROW_SCALE=(strategy == "token"),
        BLOCK_M=_QUANT_BLOCK_M,
        BLOCK_K=_QUANT_BLOCK_K,
    )

    out = out.reshape(x.shape)
    if strategy == "token":
        scale = scale.reshape(*x.shape[:-1], 1)
    else:
        scale = scale.reshape(())
    return out, scale


def triton_quantize_dequantize_input_dynamic_fp8_scale_only(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
    strategy: str,
) -> torch.Tensor:
    """Quantize-dequantize using Triton only for the FP8 scale reduction.

    This is the exact large-shape fallback used on Ampere-class GPUs where
    Triton cannot natively emit the same FP8 element type as PyTorch's
    `float8_e4m3fn`.
    """

    fp8_info = torch.finfo(fp8_dtype)
    scale = _compute_dynamic_fp8_scale_triton(x, strategy=strategy, fp8_max=float(fp8_info.max))
    x_q = torch.clamp(x.to(torch.float32) / scale, min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)
    return (x_q.to(torch.float32) * scale).to(x.dtype)


__all__ = [
    "resolve_triton_fp8_input_quant_mode",
    "supports_triton_fp8_input_quant",
    "triton_input_quant_available",
    "triton_quantize_input_dynamic_fp8",
    "triton_quantize_dequantize_input_dynamic_fp8_scale_only",
]
