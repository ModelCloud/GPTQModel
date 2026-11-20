# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU correctness kernel for row-packed MXFP4, without a dense weight cache."""

import torch
from torch import nn
from torch.nn import functional as F


class MXFP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=None):
        super().__init__()
        if in_features <= 0 or in_features % 32 or out_features <= 0:
            raise ValueError("MXFP4 requires positive dimensions and 32-element input blocks")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.full((out_features, in_features // 32), 127, dtype=torch.uint8))
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_solution(cls, solution, bias=None):
        result = cls(solution.grid.shape[1], solution.grid.shape[0], bias)
        result.qweight = solution.payload.clone()
        result.scales = solution.grid.scales.clone()
        return result

    @torch.no_grad()
    def forward(self, inputs):
        if inputs.device.type != "cpu" or self.qweight.device.type != "cpu":
            raise ValueError("MXFP4Linear is currently a CPU correctness kernel")
        if self.qweight.dtype != torch.uint8 or self.scales.dtype != torch.uint8 or (self.scales == 255).any():
            raise ValueError("Invalid MXFP4 payload or E8M0 scales")
        if not inputs.is_floating_point() or inputs.shape[-1] != self.in_features:
            raise ValueError("MXFP4Linear requires floating inputs with the declared input width")
        dtype = torch.float64 if inputs.dtype == torch.float64 else torch.float32
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=dtype)
        source = inputs.to(dtype)
        output = torch.empty((*inputs.shape[:-1], self.out_features), dtype=inputs.dtype)
        with torch.autocast("cpu", enabled=False):
            for start in range(0, self.out_features, 128):
                end = min(start + 128, self.out_features)
                payload = self.qweight[start:end]
                codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(end - start, self.in_features)
                weight = values[(codes & 7).long()]
                weight = torch.where((codes & 8) != 0, -weight, weight)
                exponent = self.scales[start:end].to(torch.int32).sub(127).repeat_interleave(32, dim=1)
                weight = torch.ldexp(weight, exponent)
                bias = None if self.bias is None else self.bias[start:end].to(dtype)
                output[..., start:end] = F.linear(source, weight, bias).to(inputs.dtype)
                del weight, codes, exponent
        return output
