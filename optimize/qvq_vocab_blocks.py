# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounded output-channel partitions for offline QVQ head calibration.

The wrapper keeps the full vocabulary distribution for the real-Fisher loss.
Each child linear exposes one principal output block to the YAQA collector.
This is a preparation operator; serving and checkpoint formats are separate.
"""

from __future__ import annotations

import torch
from torch import nn


class VocabBlockLinear(nn.Module):
    """Present a dense vocabulary head as contiguous independent row blocks."""

    def __init__(self, source: nn.Linear, block_rows: int = 2048):
        super().__init__()
        if not isinstance(source, nn.Linear) or source.bias is not None:
            raise TypeError("QVQ vocabulary blocking requires a bias-free nn.Linear head")
        if block_rows < 16 or block_rows % 16:
            raise ValueError("QVQ vocabulary block rows must be a positive multiple of 16")
        if source.out_features % 16 or source.in_features % 16:
            raise ValueError("QVQ vocabulary dimensions must be multiples of 16")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.block_rows = block_rows
        self.blocks = nn.ModuleList()
        for start in range(0, source.out_features, block_rows):
            stop = min(start + block_rows, source.out_features)
            block = nn.Linear(source.in_features, stop - start, bias=False,
                              device=source.weight.device, dtype=source.weight.dtype)
            with torch.no_grad():
                block.weight.copy_(source.weight[start:stop])
            self.blocks.append(block)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(hidden) for block in self.blocks], dim=-1)

    def yaqa_targets(self, prefix: str = "lm_head") -> dict[str, nn.Linear]:
        return {f"{prefix}.blocks.{index}": block for index, block in enumerate(self.blocks)}


def factored_head_fisher_loss(
    errors: list[torch.Tensor],
    output_factors: list[torch.Tensor],
    input_hessian: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate the complete two-sided head Fisher without an output Gram.

    For output Fisher ``S @ S.T`` and concatenated weight error ``E``, the
    objective is ``tr((E @ S).T @ H @ (E @ S))``. Keeping ``E @ S`` small
    retains the interactions between different vocabulary blocks.
    """
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("Fisher oracle dtype must be FP32 or FP64")
    if not errors or len(errors) != len(output_factors):
        raise ValueError("Fisher oracle requires matching nonempty error/factor blocks")
    width = input_hessian.shape[0]
    rank = output_factors[0].shape[1]
    if input_hessian.shape != (width, width):
        raise ValueError("Fisher input Hessian must be square")
    if any(error.shape != (width, factor.shape[0]) or factor.shape[1] != rank
           for error, factor in zip(errors, output_factors)):
        raise ValueError("Fisher blocks have mismatched input, output, or factor dimensions")
    if any(error.device != input_hessian.device or factor.device != input_hessian.device
           for error, factor in zip(errors, output_factors)):
        raise ValueError("Fisher blocks and input Hessian must share a device")
    projected = sum(error.to(dtype) @ factor.to(dtype) for error, factor in zip(errors, output_factors))
    return torch.einsum("ir,ij,jr->", projected, input_hessian.to(dtype), projected)
