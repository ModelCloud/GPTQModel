# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utility helpers for dtype compatibility across accelerator generations."""

from __future__ import annotations

from typing import Optional

import torch


try:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
except Exception:
    unpack_uint4 = None
    f4_unpacked_to_f32 = None

__all__ = [
    "device_supports_native_fp8",
    "dequantize_f8_e4m3",
    "dequantize_f4_e2m1",
]


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

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current PyTorch build does not provide float8_e4m3fn tensors")

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")

    if tensor.dtype is not torch.float8_e4m3fn:
        result = tensor.to(target_dtype)
    else:
        result = tensor.to(target_dtype)

    def _expand_scale(scale_tensor: torch.Tensor, *, axis_hint: Optional[int]) -> torch.Tensor:
        if scale_tensor.ndim == 0:
            return scale_tensor

        target_shape = result.shape

        if scale_tensor.shape == target_shape:
            return scale_tensor

        # Block-wise expansion (e.g. [num_row_blocks, num_col_blocks])
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

    if scale is not None:
        scale_tensor = _expand_scale(scale.to(result.dtype), axis_hint=axis)
        result = result * scale_tensor
    elif scale_inv is not None:
        scale_tensor = _expand_scale(scale_inv.to(result.dtype), axis_hint=axis)
        if torch.max(torch.abs(scale_tensor)) <= 1:
            result = result * scale_tensor
        else:
            result = result / scale_tensor

    return result


def dequantize_f4_e2m1(
    tensor: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    scale_inv: Optional[torch.Tensor] = None,
    axis: Optional[int] = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP4 (E2M1) values packed as two nibbles per byte."""

    if unpack_uint4 is None or f4_unpacked_to_f32 is None:
        raise RuntimeError("torchao with nvfp4 support is required for FP4 dequantization")

    if scale is not None and scale_inv is not None:
        raise ValueError("Provide either scale or scale_inv, not both")

    if tensor.dtype is not torch.uint8:
        raise ValueError("FP4 packed tensors must use torch.uint8 storage")

    orig_shape = list(tensor.shape)
    if not orig_shape:
        raise ValueError("Tensor must have at least one dimension")

    unpacked = unpack_uint4(tensor.reshape(-1))
    expanded_shape = orig_shape[:-1] + [orig_shape[-1] * 2]
    unpacked = unpacked.view(*expanded_shape)

    result = f4_unpacked_to_f32(unpacked).to(target_dtype)

    def _expand_scale_fp4(scale_tensor: torch.Tensor, *, axis_hint: Optional[int]) -> torch.Tensor:
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

    if scale is not None:
        scale_tensor = _expand_scale_fp4(scale.to(result.dtype), axis_hint=axis)
        result = result * scale_tensor
    elif scale_inv is not None:
        scale_tensor = _expand_scale_fp4(scale_inv.to(result.dtype), axis_hint=axis)
        if torch.max(torch.abs(scale_tensor)) <= 1:
            result = result * scale_tensor
        else:
            result = result / scale_tensor

    return result
