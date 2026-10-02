# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Persistent calibration policy at the producer of an NVFP4 carrier."""

from __future__ import annotations

import math

import torch

from .w4a_activation import pack_activation


def llama_nvfp4_boundaries(layers, selected):
    """Yield activation producers in execution order, including stream entries."""
    for index, (layer, enabled) in enumerate(zip(layers, selected, strict=True)):
        if not enabled:
            continue
        prefix = f"model.layers.{index}"
        if index == 0 or not selected[index - 1]:
            yield layer, "input", f"{prefix}.input"
        yield layer.self_attn, "output", f"{prefix}.self_attn.o_proj.input"
        yield layer, "attention_residual", f"{prefix}.post_attention_residual"
        yield layer.mlp, "product", f"{prefix}.mlp.down_proj.input"
        yield layer, "output", f"{prefix}.output"


def validate_producer_scales(boundaries, scales):
    if scales is None:
        return
    expected = {key for _, _, key in boundaries}
    if set(scales) != expected:
        raise ValueError(f"Incomplete NVFP4 producer scales: missing={sorted(expected - set(scales))}, "
                         f"extra={sorted(set(scales) - expected)}")


class NVFP4BoundaryQuantizer(torch.nn.Module):
    """Use a calibrated FP32 global scale and dynamic E4M3 block scales.

    The authoritative values live in the activation config. Runtime copies are
    nonpersistent INT32 bit buffers so model dtype conversions cannot round
    them. An observer is used only by the explicit calibration pass.
    """

    def __init__(self, key: str, device, scale: float | None = None):
        super().__init__()
        self.key = key
        self.observer = None
        self.calibrated = False
        self.register_buffer("scale_bits", torch.ones((), dtype=torch.float32, device=device).view(torch.int32),
                             persistent=False)
        if scale is not None:
            self.set_scale(scale)

    @property
    def global_scale(self):
        return self.scale_bits.view(torch.float32)

    def set_scale(self, scale: float) -> None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Boundary {self.key} needs a finite positive FP32 scale")
        value = torch.tensor(scale, dtype=torch.float32, device=self.scale_bits.device)
        if not bool(torch.isfinite(value)) or not bool(value > 0):
            raise ValueError(f"Boundary {self.key} scale is outside the positive FP32 range")
        self.scale_bits.copy_(value.view(torch.int32))
        self.calibrated = True

    def forward(self, x: torch.Tensor, mode: str, **kwargs):
        if mode != "w4a_nvfp4":
            raise ValueError("An NVFP4 boundary quantizer requires an NVFP4 carrier")
        if self.observer is not None:
            self.observer(x)
        if self.calibrated:
            kwargs["global_scale"] = self.global_scale
        return pack_activation(x, mode, **kwargs)


def pack_boundary(owner, boundary: str, x: torch.Tensor, mode: str, **kwargs):
    quantizer = getattr(owner, f"_w4a_{boundary}_quantizer", None)
    return (pack_activation(x, mode, **kwargs) if quantizer is None
            else quantizer(x, mode, **kwargs))
