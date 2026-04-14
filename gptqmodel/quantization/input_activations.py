# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import InputActivationQuantConfig, _normalize_input_activations


def normalize_input_activations(
    payload: Optional[InputActivationQuantConfig | dict[str, Any]],
) -> Optional[InputActivationQuantConfig]:
    """Reuse the shared config normalizer so helpers accept the same activation payloads as QuantizeConfig."""

    return _normalize_input_activations(payload)


def quantize_dequantize_input(
    x: torch.Tensor,
    payload: Optional[InputActivationQuantConfig | dict[str, Any]],
) -> torch.Tensor:
    config = normalize_input_activations(payload)
    if config is None:
        return x
    if not x.is_floating_point():
        raise TypeError(f"Activation quantization expects floating-point inputs, got `{x.dtype}`.")

    fp8_dtype = getattr(torch, config.format)
    fp8_max = torch.finfo(fp8_dtype).max
    x_work = x.to(torch.float32)

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

    x_q = torch.clamp(x_work * scale_inv, min=-fp8_max, max=fp8_max).to(fp8_dtype)
    x_dq = x_q.to(torch.float32) / scale_inv
    return x_dq.to(dtype=x.dtype)


__all__ = ["normalize_input_activations", "quantize_dequantize_input"]
