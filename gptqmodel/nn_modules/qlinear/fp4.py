# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...quantization.dtype import dequantize_f4_e2m1


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:
    NVFP4Tensor = None


class TorchFP4Linear(nn.Module):
    """Execute one linear layer directly from NVFP4-packed checkpoint weights."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_block_size: int,
        orig_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_block_size = int(weight_block_size)
        self.orig_dtype = orig_dtype
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale)
        if isinstance(bias, torch.Tensor):
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _native_weight(self) -> "NVFP4Tensor":
        """Wrap the stored packed weight buffers into the torchao NVFP4 tensor view."""

        if NVFP4Tensor is None:
            raise RuntimeError("TorchFP4Linear requires torchao NVFP4Tensor support.")
        return NVFP4Tensor(
            self.weight,
            self.weight_scale,
            block_size=self.weight_block_size,
            orig_dtype=self.orig_dtype,
        )

    def _cpu_forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Decode live FP4 buffers on CPU without retaining a dense weight."""

        weight = dequantize_f4_e2m1(
            self.weight,
            scale=self.weight_scale,
            axis=None,
            target_dtype=x.dtype,
        )
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch one dense linear projection through the NVFP4 tensor wrapper."""

        bias = self.bias
        if isinstance(bias, torch.Tensor) and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        if x.device.type == "cpu" and self.weight.device.type == "cpu":
            return self._cpu_forward(x, bias)
        return F.linear(x, self._native_weight(), bias)
