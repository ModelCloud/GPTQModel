# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Utility helpers for dtype compatibility across accelerator generations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


try:
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
except Exception:
    unpack_uint4 = None
    f4_unpacked_to_f32 = None

__all__ = [
    "DeviceDTypeSupport",
    "get_device_dtype_support",
    "device_supports_dtype",
    "device_supports_native_fp8",
    "dequantize_fp8",
    "dequantize_f8_e4m3",
    "dequantize_f4_e2m1",
]


_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2")
    if hasattr(torch, name)
)

_DTYPE_SUPPORT_CACHE: dict[tuple[str, Optional[int], bool], "DeviceDTypeSupport"] = {}


@dataclass(frozen=True)
class DeviceDTypeSupport:
    """Describe which execution dtypes a device advertises and validates.

    ``advertised_linear_dtypes`` is architecture-based. It answers what the
    device family is expected to support in native matmul / linear kernels.
    ``validated_linear_dtypes`` is runtime-probed and therefore stricter.
    """

    device: torch.device
    capability: Optional[tuple[int, int]]
    advertised_linear_dtypes: frozenset[torch.dtype]
    validated_linear_dtypes: frozenset[torch.dtype]

    def supports(self, dtype: torch.dtype, *, require_validation: bool = False) -> bool:
        """Return whether ``dtype`` is supported for linear-style execution."""

        supported = (
            self.validated_linear_dtypes
            if require_validation
            else self.advertised_linear_dtypes
        )
        return dtype in supported


def _normalize_device(device: Optional[torch.device]) -> torch.device:
    """Resolve ``device`` into a concrete torch device."""

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return torch.device(device)


def _cuda_capability(device: torch.device) -> Optional[tuple[int, int]]:
    """Return CUDA compute capability for a concrete device."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return tuple(int(v) for v in torch.cuda.get_device_capability(device))


def _advertised_cuda_linear_dtypes(capability: tuple[int, int]) -> frozenset[torch.dtype]:
    """Map CUDA architecture families to expected fast linear dtypes."""

    major, minor = capability
    supported = {
        torch.float16,
        torch.float32,
    }

    if major >= 8:
        supported.add(torch.bfloat16)

    if _FLOAT8_DTYPES and (major >= 9 or (major, minor) == (8, 9)):
        supported.update(_FLOAT8_DTYPES)

    return frozenset(supported)


def _advertised_linear_dtypes_for_device(device: torch.device) -> tuple[Optional[tuple[int, int]], frozenset[torch.dtype]]:
    """Return the architecture-level linear dtype support set for ``device``."""

    if device.type == "cuda" and torch.cuda.is_available():
        capability = _cuda_capability(device)
        assert capability is not None
        return capability, _advertised_cuda_linear_dtypes(capability)

    # CPU / other devices fall back to portable dtypes only. This helper is
    # used for accelerator routing decisions, so keep the default conservative.
    return None, frozenset({torch.float16, torch.float32, torch.bfloat16})


def _validate_linear_dtype_support(device: torch.device, dtype: torch.dtype) -> bool:
    """Runtime-probe whether one small linear/matmul path works for ``dtype``."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return dtype in {torch.float16, torch.float32, torch.bfloat16}

    try:
        if dtype in _FLOAT8_DTYPES:
            if not hasattr(torch, "_scaled_mm"):
                return False
            compute_dtype = torch.bfloat16 if device_supports_dtype(device, torch.bfloat16) else torch.float16
            a = torch.randn(16, 16, device=device, dtype=compute_dtype).to(dtype)
            b = torch.randn(16, 16, device=device, dtype=compute_dtype).to(dtype)
            result = torch._scaled_mm(
                a,
                b,
                scale_a=torch.tensor(1.0, device=device),
                scale_b=torch.tensor(1.0, device=device),
                out_dtype=compute_dtype,
            )
            if isinstance(result, tuple):
                result = result[0]
            return isinstance(result, torch.Tensor)

        a = torch.randn(16, 16, device=device, dtype=dtype)
        b = torch.randn(16, 16, device=device, dtype=dtype)
        result = torch.matmul(a, b)
        return isinstance(result, torch.Tensor)
    except Exception:
        return False


def get_device_dtype_support(
    device: Optional[torch.device] = None,
    *,
    validate: bool = False,
) -> DeviceDTypeSupport:
    """Return linear-dtype support metadata for ``device``.

    ``validate=False`` is architecture-based and cheap.
    ``validate=True`` probes one small kernel per advertised dtype and caches
    the results for subsequent callers.
    """

    resolved_device = _normalize_device(device)
    cache_key = (resolved_device.type, resolved_device.index, bool(validate))
    cached = _DTYPE_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    capability, advertised = _advertised_linear_dtypes_for_device(resolved_device)
    validated = advertised
    if validate:
        validated = frozenset(
            dtype
            for dtype in advertised
            if _validate_linear_dtype_support(resolved_device, dtype)
        )

    support = DeviceDTypeSupport(
        device=resolved_device,
        capability=capability,
        advertised_linear_dtypes=advertised,
        validated_linear_dtypes=validated,
    )
    _DTYPE_SUPPORT_CACHE[cache_key] = support
    return support


def device_supports_dtype(
    device: Optional[torch.device],
    dtype: torch.dtype,
    *,
    require_validation: bool = False,
) -> bool:
    """Return whether ``device`` supports ``dtype`` for linear-style kernels."""

    support = get_device_dtype_support(device, validate=require_validation)
    return support.supports(dtype, require_validation=require_validation)


def device_supports_native_fp8(
    device: Optional[torch.device] = None,
    *,
    require_validation: bool = False,
) -> bool:
    """Return ``True`` when the target CUDA device supports native FP8 (E4M3).

    This compatibility wrapper now forwards to the generic device/dtype support
    map. By default it returns architecture-advertised support; callers that
    need a stricter answer may pass ``require_validation=True``.
    """

    if not _FLOAT8_DTYPES:
        return False
    return device_supports_dtype(
        device,
        torch.float8_e4m3fn,
        require_validation=require_validation,
    )


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

    if tensor.dtype not in _FLOAT8_DTYPES:
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
