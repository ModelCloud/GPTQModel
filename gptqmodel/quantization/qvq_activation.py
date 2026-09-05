# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference FP8 activation quantization shared by QVQ calibration and inference."""

from __future__ import annotations

import math
from typing import Any

import torch

QVQ_FP8_ACTIVATION_FORMAT = "float8_e4m3fn"
QVQ_FP8_ACTIVATION_SCALE_METHOD = "dynamic_per_token"

_QVQ_FP8_FORMAT_ALIASES = {
    "e4m3": QVQ_FP8_ACTIVATION_FORMAT,
    "e4m3fn": QVQ_FP8_ACTIVATION_FORMAT,
    QVQ_FP8_ACTIVATION_FORMAT: QVQ_FP8_ACTIVATION_FORMAT,
}
_QVQ_FP8_SCALE_METHOD_ALIASES = {
    "dynamic_per_token": QVQ_FP8_ACTIVATION_SCALE_METHOD,
    "per_token": QVQ_FP8_ACTIVATION_SCALE_METHOD,
    "token": QVQ_FP8_ACTIVATION_SCALE_METHOD,
}


def normalize_qvq_fp8_activation_format(value: Any) -> str:
    """Normalize the NVIDIA-oriented QVQ A8 storage dtype."""

    normalized = (
        QVQ_FP8_ACTIVATION_FORMAT if value is None else str(value).strip().lower()
    )
    resolved = _QVQ_FP8_FORMAT_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(_QVQ_FP8_FORMAT_ALIASES))
        raise ValueError(
            f"QVQ A8 `format` must be one of {{{supported}}}, got `{value}`."
        )
    if not hasattr(torch, resolved):
        raise ValueError(f"QVQ A8 requires a PyTorch build with `{resolved}` support.")
    return resolved


def normalize_qvq_fp8_activation_scale_method(value: Any) -> str:
    """Normalize the row-local scaling contract used for decode and prefill."""

    normalized = (
        QVQ_FP8_ACTIVATION_SCALE_METHOD if value is None else str(value).strip().lower()
    )
    resolved = _QVQ_FP8_SCALE_METHOD_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(_QVQ_FP8_SCALE_METHOD_ALIASES))
        raise ValueError(
            f"QVQ A8 `scale_method` must be one of {{{supported}}}, got `{value}`."
        )
    return resolved


def quantize_qvq_fp8_activation(
    activation: torch.Tensor,
    *,
    format: str = QVQ_FP8_ACTIVATION_FORMAT,
    scale_method: str = QVQ_FP8_ACTIVATION_SCALE_METHOD,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to FP8 and return one FP32 scale per row.

    The scale maps an FP8 value back to the activation domain. Zero rows use a
    scale of one, avoiding a zero divisor without changing their exact value.
    """

    if not isinstance(activation, torch.Tensor) or not activation.is_floating_point():
        raise TypeError(
            "QVQ A8 activation quantization requires a floating-point tensor."
        )
    if activation.ndim < 1 or activation.shape[-1] < 1:
        raise ValueError(
            "QVQ A8 activation quantization requires a nonempty last dimension."
        )
    format = normalize_qvq_fp8_activation_format(format)
    normalize_qvq_fp8_activation_scale_method(scale_method)
    if not isinstance(validate, bool):
        raise TypeError("QVQ A8 `validate` must be boolean.")
    if validate and not bool(torch.isfinite(activation).all()):
        raise ValueError("QVQ A8 activation quantization requires finite values.")

    if (
        activation.device.type == "cuda"
        and activation.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and activation.is_contiguous()
        and torch.cuda.get_device_capability(activation.device) >= (8, 9)
    ):
        # One row-local CUDA kernel replaces the reference chain of FP32 cast,
        # abs, reduction, where/divide, clamp, and E4M3 conversion.  The scale
        # and saturation contract is bit-exact with the operations below.
        from ..utils.qvq_cuda import qvq_cuda_quantize_fp8_per_row

        return qvq_cuda_quantize_fp8_per_row(activation)

    fp8_dtype = getattr(torch, format)
    fp8_max = float(torch.finfo(fp8_dtype).max)
    working = activation.to(torch.float32)
    peak = working.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        peak > 0,
        peak / fp8_max,
        torch.ones_like(peak),
    )
    scaled = torch.clamp(working / scale, min=-fp8_max, max=fp8_max)
    quantized = scaled.to(fp8_dtype)
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def dequantize_qvq_fp8_activation(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize the portable reference for a QVQ FP8 activation payload."""

    if quantized.dtype != getattr(torch, QVQ_FP8_ACTIVATION_FORMAT):
        raise TypeError(
            f"QVQ A8 payload must use {QVQ_FP8_ACTIVATION_FORMAT}, got {quantized.dtype}."
        )
    if scale.dtype != torch.float32 or scale.shape != (*quantized.shape[:-1], 1):
        raise ValueError(
            "QVQ A8 scale must be contiguous FP32 with one value per activation row."
        )
    if (
        scale.device != quantized.device
        or not quantized.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError(
            "QVQ A8 payload and scale must be contiguous on the same device."
        )
    return (quantized.to(torch.float32) * scale).to(dtype)


def fake_quantize_qvq_fp8_activation(
    activation: torch.Tensor,
    *,
    format: str = QVQ_FP8_ACTIVATION_FORMAT,
    scale_method: str = QVQ_FP8_ACTIVATION_SCALE_METHOD,
    straight_through: bool = False,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the quantized payload, scale, and dequantized activation.

    ``straight_through=True`` preserves the source gradient while the forward
    value remains the exact FP8 quantize/dequantize result. YAQA uses this to
    collect full-model Fisher factors under the same A8 forward contract.
    """

    quantized, scale = quantize_qvq_fp8_activation(
        activation,
        format=format,
        scale_method=scale_method,
        validate=validate,
    )
    dequantized = dequantize_qvq_fp8_activation(
        quantized, scale, dtype=activation.dtype
    )
    if straight_through and activation.requires_grad:
        dequantized = activation + (dequantized - activation).detach()
    return quantized, scale, dequantized


def qvq_fp8_activation_error(
    source: torch.Tensor, dequantized: torch.Tensor
) -> dict[str, float | int]:
    """Summarize bounded calibration error for one activation tensor."""

    if source.shape != dequantized.shape:
        raise ValueError("QVQ A8 error inputs must have identical shapes.")
    source_fp32 = source.detach().to(torch.float32)
    error = dequantized.detach().to(torch.float32) - source_fp32
    source_square = float(source_fp32.square().sum().item())
    error_square = float(error.square().sum().item())
    elements = source.numel()
    rmse = math.sqrt(error_square / max(1, elements))
    relative_rmse = math.sqrt(
        error_square / max(source_square, torch.finfo(torch.float32).tiny)
    )
    return {
        "elements": elements,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "maximum_absolute_error": float(error.abs().amax().item()) if elements else 0.0,
    }


__all__ = [
    "QVQ_FP8_ACTIVATION_FORMAT",
    "QVQ_FP8_ACTIVATION_SCALE_METHOD",
    "dequantize_qvq_fp8_activation",
    "fake_quantize_qvq_fp8_activation",
    "normalize_qvq_fp8_activation_format",
    "normalize_qvq_fp8_activation_scale_method",
    "quantize_qvq_fp8_activation",
    "qvq_fp8_activation_error",
]
