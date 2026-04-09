# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utility helpers for dtype compatibility across accelerator generations."""

from __future__ import annotations

import os
from typing import Optional

import torch


try:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
except Exception:
    unpack_uint4 = None
    f4_unpacked_to_f32 = None

__all__ = [
    "available_float4_packed_dtype_names",
    "available_float4_packed_dtypes",
    "available_float8_dtype_names",
    "available_float8_dtypes",
    "device_supports_native_fp8",
    "dequantize_fp8",
    "dequantize_f8_e4m3",
    "dequantize_f4_e2m1",
    "is_fp4_packed_dtype",
]


# Keep the canonical floatx registries in one place so CPU dequant, config
# normalization, model conversion, and tests all follow the same torch surface.
_FLOAT8_CANDIDATE_NAMES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
)
_FLOAT8_DTYPE_NAMES = tuple(name for name in _FLOAT8_CANDIDATE_NAMES if hasattr(torch, name))
_FLOAT8_DTYPES = tuple(getattr(torch, name) for name in _FLOAT8_DTYPE_NAMES)

_FLOAT4_PACKED_CANDIDATE_NAMES = ("float4_e2m1fn_x2",)
_FLOAT4_PACKED_DTYPE_NAMES = tuple(name for name in _FLOAT4_PACKED_CANDIDATE_NAMES if hasattr(torch, name))
_FLOAT4_PACKED_DTYPES = tuple(getattr(torch, name) for name in _FLOAT4_PACKED_DTYPE_NAMES)
_FLOAT4_PACKED_DTYPE = _FLOAT4_PACKED_DTYPES[0] if _FLOAT4_PACKED_DTYPES else None

_TARGET_DTYPE_CODES = {
    torch.bfloat16: 0,
    torch.float16: 1,
}
_FP8_FORMAT_CODES = {
    getattr(torch, "float8_e4m3fn", None): 0,
    getattr(torch, "float8_e5m2", None): 1,
    getattr(torch, "float8_e4m3fnuz", None): 2,
    getattr(torch, "float8_e5m2fnuz", None): 3,
    getattr(torch, "float8_e8m0fnu", None): 4,
}


def available_float8_dtype_names() -> tuple[str, ...]:
    return _FLOAT8_DTYPE_NAMES


def available_float8_dtypes() -> tuple[torch.dtype, ...]:
    return _FLOAT8_DTYPES


def available_float4_packed_dtype_names() -> tuple[str, ...]:
    return _FLOAT4_PACKED_DTYPE_NAMES


def available_float4_packed_dtypes() -> tuple[torch.dtype, ...]:
    return _FLOAT4_PACKED_DTYPES


def is_fp4_packed_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FLOAT4_PACKED_DTYPES


def _cpu_floatx_threads(numel: Optional[int] = None, *, enable_large_threads: bool = False) -> int:
    raw = os.environ.get("GPTQMODEL_FLOATX_CPU_THREADS", "").strip()
    default_value = 32 if enable_large_threads and numel is not None and numel >= 64 * 1024 * 1024 else 8
    try:
        value = int(raw) if raw else default_value
    except ValueError:
        value = default_value
    return max(1, min(value, 32, os.cpu_count() or 1))


def _can_use_fast_path(
    tensor: torch.Tensor,
    scale_tensor: Optional[torch.Tensor],
    *,
    target_dtype: torch.dtype,
    allow_float4_storage: bool = False,
) -> bool:
    if target_dtype not in _TARGET_DTYPE_CODES:
        return False
    if tensor.device.type != "cpu":
        return False
    if tensor.ndim not in (1, 2):
        return False
    if tensor.dtype not in _FLOAT8_DTYPES:
        if allow_float4_storage:
            if tensor.dtype != torch.uint8 and tensor.dtype not in _FLOAT4_PACKED_DTYPES:
                return False
        else:
            return False
    if scale_tensor is None:
        return True
    if scale_tensor.device.type != "cpu" or scale_tensor.ndim > 2:
        return False
    return True


def _prefer_reference_fp8_cpu(
    tensor: torch.Tensor,
    scale_tensor: Optional[torch.Tensor],
    *,
    target_dtype: torch.dtype,
) -> bool:
    if scale_tensor is None:
        return False
    if tensor.device.type != "cpu":
        return False
    if target_dtype not in _TARGET_DTYPE_CODES:
        return False
    if os.environ.get("GPTQMODEL_FLOATX_CPU_FORCE_NATIVE_FP8", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False

    standard_fp8_dtypes = tuple(
        dtype for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        ) if dtype is not None
    )
    return tensor.dtype in standard_fp8_dtypes


def _load_floatx_cpu_ops():
    try:
        from ..utils.cpp import load_floatx_cpu_extension
    except Exception:
        return None

    ext = load_floatx_cpu_extension()
    if not ext:
        return None

    namespace = getattr(torch.ops, "gptqmodel_floatx", None)
    if namespace is None:
        return None
    if not hasattr(namespace, "dequantize_fp8_cpu") or not hasattr(namespace, "dequantize_fp4_cpu"):
        return None
    return namespace


def _fast_scale_arg(
    *,
    scale: Optional[torch.Tensor],
    scale_inv: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], int]:
    if scale is not None:
        return scale.to(device="cpu", dtype=torch.float32).contiguous(), 1
    if scale_inv is None:
        return None, 0

    scale_tensor = scale_inv.to(device="cpu", dtype=torch.float32).contiguous()
    max_abs = float(torch.max(torch.abs(scale_tensor)).item()) if scale_tensor.numel() else 0.0
    return scale_tensor, 1 if max_abs <= 1.0 else 2


def _expand_scale(
    scale_tensor: torch.Tensor,
    result: torch.Tensor,
    *,
    axis_hint: Optional[int],
) -> torch.Tensor:
    if scale_tensor.ndim == 0:
        return scale_tensor

    target_shape = result.shape
    if scale_tensor.shape == target_shape:
        return scale_tensor

    if scale_tensor.ndim == 2 and len(target_shape) == 2:
        blocks_r, blocks_c = scale_tensor.shape
        rows, cols = target_shape
        if rows % blocks_r == 0 and cols % blocks_c == 0:
            repeat_r = rows // blocks_r
            repeat_c = cols // blocks_c
            expanded = scale_tensor.repeat_interleave(repeat_r, dim=0)
            expanded = expanded.repeat_interleave(repeat_c, dim=1)
            return expanded

    if scale_tensor.ndim == 1 and len(target_shape) == 2:
        rows, cols = target_shape
        count = scale_tensor.shape[0]
        axis = axis_hint if axis_hint is not None else 0
        axis = axis if axis >= 0 else axis + len(target_shape)
        if axis == 0 and rows % count == 0:
            repeat = rows // count
            expanded = scale_tensor.repeat_interleave(repeat, dim=0).view(rows, 1)
            return expanded.expand(rows, cols)
        if axis == 1 and cols % count == 0:
            repeat = cols // count
            expanded = scale_tensor.repeat_interleave(repeat, dim=0).view(1, cols)
            return expanded.expand(rows, cols)

    if scale_tensor.ndim == result.ndim:
        expanded = scale_tensor
        for dim, (target_size, current_size) in enumerate(zip(result.shape, expanded.shape)):
            if target_size == current_size:
                continue
            if current_size == 1:
                expanded = expanded.expand(*[
                    target_size if i == dim else expanded.shape[i]
                    for i in range(expanded.ndim)
                ])
                continue
            if target_size % current_size != 0:
                raise ValueError(
                    f"Cannot broadcast scale dimension {current_size} to target {target_size}"
                )
            repeat = target_size // current_size
            expanded = expanded.repeat_interleave(repeat, dim=dim)
        return expanded

    reshaped = _reshape_for_axis(scale_tensor, axis_hint, result.ndim)
    return reshaped.expand(result.shape)


def _dequantize_f8_reference(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if not _FLOAT8_DTYPES:
        raise RuntimeError("Current PyTorch build does not provide FP8 tensors")

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")

    result = tensor.to(target_dtype)
    if scale is not None:
        scale_tensor = _expand_scale(scale.to(result.dtype), result, axis_hint=axis)
        result = result * scale_tensor
    elif scale_inv is not None:
        scale_tensor = _expand_scale(scale_inv.to(result.dtype), result, axis_hint=axis)
        if torch.max(torch.abs(scale_tensor)) <= 1:
            result = result * scale_tensor
        else:
            result = result / scale_tensor
    return result


def _dequantize_f4_reference(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if unpack_uint4 is None or f4_unpacked_to_f32 is None:
        raise RuntimeError("torchao with nvfp4 support is required for FP4 dequantization")

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")

    if is_fp4_packed_dtype(tensor.dtype):
        tensor = tensor.view(torch.uint8)
    elif tensor.dtype is not torch.uint8:
        raise ValueError("FP4 packed tensors must use torch.uint8 storage")

    orig_shape = list(tensor.shape)
    if not orig_shape:
        raise ValueError("Tensor must have at least one dimension")

    unpacked = unpack_uint4(tensor.reshape(-1))
    expanded_shape = orig_shape[:-1] + [orig_shape[-1] * 2]
    unpacked = unpacked.view(*expanded_shape)
    result = f4_unpacked_to_f32(unpacked).to(target_dtype)

    if scale is not None:
        scale_tensor = _expand_scale(scale.to(result.dtype), result, axis_hint=axis)
        result = result * scale_tensor
    elif scale_inv is not None:
        scale_tensor = _expand_scale(scale_inv.to(result.dtype), result, axis_hint=axis)
        if torch.max(torch.abs(scale_tensor)) <= 1:
            result = result * scale_tensor
        else:
            result = result / scale_tensor
    return result


def device_supports_native_fp8(device: Optional[torch.device] = None) -> bool:
    """Return ``True`` when the target CUDA device supports native FP8 (E4M3).

    Hopper-class GPUs (SM >= 9.0) expose hardware accelerated FP8 kernels while
    earlier generations such as the A100 (SM 8.x) do not.  When CUDA is
    unavailable this helper always returns ``False``.
    """

    if not torch.cuda.is_available():
        return False

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    if device.type != "cuda":
        return False

    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9


def _reshape_for_axis(tensor: torch.Tensor, axis: Optional[int], target_ndim: int) -> torch.Tensor:
    """Expand ``tensor`` with singleton dimensions so it broadcasts on ``axis``."""

    if tensor.ndim == 0:
        return tensor

    if axis is None:
        if tensor.ndim == target_ndim:
            return tensor
        return tensor.view(*tensor.shape, *([1] * (target_ndim - tensor.ndim)))

    axis = axis if axis >= 0 else axis + target_ndim
    if axis < 0 or axis >= target_ndim:
        raise ValueError(f"axis {axis} out of range for target ndim {target_ndim}")

    if tensor.ndim == 1:
        view_shape = [1] * target_ndim
        view_shape[axis] = tensor.shape[0]
        return tensor.view(*view_shape)

    if tensor.ndim == target_ndim:
        return tensor

    if tensor.ndim < target_ndim:
        return tensor.view(*tensor.shape, *([1] * (target_ndim - tensor.ndim)))

    raise ValueError(
        "scale tensor has higher rank than target; explicit broadcasting required"
    )


def dequantize_f8_e4m3(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fully dequantize FP8 (E4M3) values to ``target_dtype``.

    ``scale`` or ``scale_inv`` mirrors the metadata stored alongside FP8 weights
    (e.g. ``weight_scale`` / ``weight_scale_inv``).  Both are optional; when
    omitted the helper falls back to a plain dtype conversion.
    """

    if not _FLOAT8_DTYPES:
        raise RuntimeError("Current PyTorch build does not provide FP8 tensors")

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")
    if tensor.dtype in _FLOAT8_DTYPES and _can_use_fast_path(
        tensor,
        scale if scale is not None else scale_inv,
        target_dtype=target_dtype,
    ):
        if scale is None and scale_inv is None:
            return tensor.to(target_dtype)
        if _prefer_reference_fp8_cpu(
            tensor,
            scale if scale is not None else scale_inv,
            target_dtype=target_dtype,
        ):
            return _dequantize_f8_reference(
                tensor,
                scale=scale,
                scale_inv=scale_inv,
                axis=axis,
                target_dtype=target_dtype,
            )

        ops = _load_floatx_cpu_ops()
        if ops is not None:
            fast_scale, scale_mode = _fast_scale_arg(scale=scale, scale_inv=scale_inv)
            format_code = _FP8_FORMAT_CODES.get(tensor.dtype)
            if format_code is not None:
                enable_large_threads = (
                    target_dtype is torch.bfloat16 and
                    hasattr(torch, "float8_e4m3fn") and
                    tensor.dtype is torch.float8_e4m3fn
                )
                source = tensor.contiguous().view(torch.uint8)
                return ops.dequantize_fp8_cpu(
                    source,
                    fast_scale,
                    scale_mode,
                    0 if axis is None else int(axis),
                    axis is None,
                    _TARGET_DTYPE_CODES[target_dtype],
                    int(format_code),
                    _cpu_floatx_threads(tensor.numel(), enable_large_threads=enable_large_threads),
                )

    return _dequantize_f8_reference(
        tensor,
        scale=scale,
        scale_inv=scale_inv,
        axis=axis,
        target_dtype=target_dtype,
    )


def dequantize_fp8(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return dequantize_f8_e4m3(
        tensor,
        scale=scale,
        scale_inv=scale_inv,
        axis=axis,
        target_dtype=target_dtype,
    )


def dequantize_f4_e2m1(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP4 (E2M1) values packed as two nibbles per byte."""

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")
    if _can_use_fast_path(
        tensor,
        scale if scale is not None else scale_inv,
        target_dtype=target_dtype,
        allow_float4_storage=True,
    ):
        ops = _load_floatx_cpu_ops()
        if ops is not None:
            fast_scale, scale_mode = _fast_scale_arg(scale=scale, scale_inv=scale_inv)
            source = tensor.contiguous()
            if source.dtype is not torch.uint8:
                source = source.view(torch.uint8)
            return ops.dequantize_fp4_cpu(
                source,
                fast_scale,
                scale_mode,
                0 if axis is None else int(axis),
                axis is None,
                _TARGET_DTYPE_CODES[target_dtype],
                _cpu_floatx_threads(source.numel() * 2),
            )

    return _dequantize_f4_reference(
        tensor,
        scale=scale,
        scale_inv=scale_inv,
        axis=axis,
        target_dtype=target_dtype,
    )
