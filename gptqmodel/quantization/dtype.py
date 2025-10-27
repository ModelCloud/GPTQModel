# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utility helpers for dtype compatibility across accelerator generations."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "device_supports_native_fp8",
    "dequantize_f8_e4m3",
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

    if scale is not None:
        scale_tensor = scale.to(result.dtype)
        if axis is not None or scale_tensor.ndim < result.ndim:
            scale_tensor = _reshape_for_axis(scale_tensor, axis, result.ndim)
        result = result * scale_tensor
    elif scale_inv is not None:
        scale_tensor = scale_inv.to(result.dtype)
        if axis is not None or scale_tensor.ndim < result.ndim:
            scale_tensor = _reshape_for_axis(scale_tensor, axis, result.ndim)
        result = result / scale_tensor

    return result
