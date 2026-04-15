# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import InputActivationQuantConfig, _normalize_input_activations


def normalize_input_activations(
    payload: Optional[InputActivationQuantConfig | dict[str, Any]],
) -> Optional[InputActivationQuantConfig]:
    """Reuse the shared config normalizer so helpers accept the same activation schema as QuantizeConfig."""

    return _normalize_input_activations(payload)


def calibrate_input_scale_inv(
    x: torch.Tensor,
    payload: Optional[InputActivationQuantConfig | dict[str, Any]],
) -> Optional[torch.Tensor]:
    """Calibrate the static FP8 input scale used by activation-aware W4A8 paths."""

    config = normalize_input_activations(payload)
    if config is None or config.dynamic:
        return None
    if config.strategy != "tensor":
        raise NotImplementedError("Static activation calibration currently only supports `strategy='tensor'`.")
    if not x.is_floating_point():
        raise TypeError(f"Activation calibration expects floating-point inputs, got `{x.dtype}`.")

    fp8_dtype = getattr(torch, config.format)
    fp8_max = torch.finfo(fp8_dtype).max
    x_work = x.to(torch.float32)
    abs_max = x_work.abs().amax()
    eps = torch.finfo(torch.float32).tiny
    return torch.where(
        abs_max > 0,
        torch.full_like(abs_max, float(fp8_max)) / abs_max.clamp_min(eps),
        torch.ones_like(abs_max),
    )


def quantize_dequantize_input(
    x: torch.Tensor,
    payload: Optional[InputActivationQuantConfig | dict[str, Any]],
    *,
    scale_inv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference activation quantize-then-dequantize path used by current W4A8 runtimes."""

    config = normalize_input_activations(payload)
    if config is None:
        return x
    if not x.is_floating_point():
        raise TypeError(f"Activation quantization expects floating-point inputs, got `{x.dtype}`.")

    fp8_dtype = getattr(torch, config.format)
    fp8_max = torch.finfo(fp8_dtype).max
    x_work = x.to(torch.float32)

    if config.dynamic:
        if config.strategy == "tensor":
            abs_max = x_work.abs().amax()
        else:
            abs_max = x_work.abs().amax(dim=-1, keepdim=True)
        eps = torch.finfo(torch.float32).tiny
        scale_inv = torch.where(
            abs_max > 0,
            torch.full_like(abs_max, float(fp8_max)) / abs_max.clamp_min(eps),
            torch.ones_like(abs_max),
        )
    else:
        if scale_inv is None:
            raise ValueError("Static activation quantization requires a calibrated `scale_inv` tensor.")
        scale_inv = scale_inv.to(device=x_work.device, dtype=torch.float32)

    x_q = torch.clamp(x_work * scale_inv, min=-fp8_max, max=fp8_max).to(fp8_dtype)
    x_dq = x_q.to(torch.float32) / scale_inv
    return x_dq.to(dtype=x.dtype)


__all__ = ["normalize_input_activations", "calibrate_input_scale_inv", "quantize_dequantize_input"]
