# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Routed-expert containers for Inkling 2 Mini.

Two storage layouts are supported:

``Inkling2LoopExperts``
    Dense ``gate_up_proj`` / ``down_proj`` 3-D parameters, the layout Defuser
    expands into per-expert ``Linear`` modules before quantization.

``Inkling2LoopPackedExperts``
    The released checkpoint's NVFP4 (``uint8`` codes + E4M3 block scales + FP32
    global scale) and MXFP8 (``float8_e4m3fn`` values + E8M0 block scales)
    expert weights. They are decoded inside the GEMM by
    :mod:`gptqmodel.utils.nvfp4_triton`, so a 4-bit expert keeps its 4-bit
    footprint instead of being materialized as bf16.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations import use_experts_implementation

from ...utils.nvfp4_triton import (
    dequant_mxfp8,
    dequant_nvfp4,
    mxfp8_w8a16_linear,
    nvfp4_w4a16_linear,
    scaled_wxa16_available,
)


class Inkling2LoopSoftcap(nn.Module):
    """``softcap * tanh(x / softcap)``: bounds a projection without changing its sign."""

    def __init__(self, softcap: float):
        super().__init__()
        self.softcap = softcap

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.softcap) * self.softcap


class Inkling2LoopSoftcappedSwiGLU(nn.Module):
    """SwiGLU whose gate sigmoid sees the raw gate while both factors stay bounded."""

    def __init__(self, gate_softcap: float, up_softcap: float):
        super().__init__()
        self.gate_softcap = Inkling2LoopSoftcap(gate_softcap)
        self.up_softcap = Inkling2LoopSoftcap(up_softcap)

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        return (self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)).to(dtype)


def _expert_activation(config) -> Inkling2LoopSoftcappedSwiGLU:
    return Inkling2LoopSoftcappedSwiGLU(config.mlp_gate_softcap, config.mlp_up_softcap)


def _route(
    hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int
):
    """Yield ``(expert, token rows, routing slots)`` for every expert that received tokens."""

    expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        slots, tokens = torch.where(expert_mask[expert_idx])
        if tokens.numel():
            yield expert_idx, tokens, slots


@use_experts_implementation
class Inkling2LoopExperts(nn.Module):
    """Dense routed experts: ``gate_up_proj`` of shape ``(E, 2 * I, H)``."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = _expert_activation(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx, tokens, slots in _route(
            hidden_states, top_k_index, top_k_weights, self.num_experts
        ):
            current = hidden_states[tokens]
            current = F.linear(current, self.gate_up_proj[expert_idx])
            current = self._apply_gate(current)
            current = F.linear(current, self.down_proj[expert_idx])
            current = current * top_k_weights[tokens, slots, None]
            final_hidden_states.index_add_(0, tokens, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate, up)


class Inkling2LoopPackedExperts(nn.Module):
    """Routed experts kept in the checkpoint's block-scaled storage.

    ``gate_up_weight`` / ``down_weight`` are the packed payload, ``*_scales`` the
    per-block scales, and for NVFP4 ``*_amax`` / ``*_outer_scale`` the per-expert
    and per-tensor factors that give the FP32 global scale. The activation is
    never quantized: it stays in the model dtype and the dot accumulates in FP32.
    """

    def __init__(self, config, fmt: str):
        super().__init__()
        if fmt not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported expert format {fmt!r}")
        self.config = config
        self.fmt = fmt
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = _expert_activation(config)

        out13, in13 = 2 * self.intermediate_dim, self.hidden_dim
        out2, in2 = self.hidden_dim, self.intermediate_dim
        if fmt == "nvfp4":
            payload13 = (self.num_experts, out13, in13 // 2)
            scales13 = (self.num_experts, out13, in13 // 16)
            payload2 = (self.num_experts, out2, in2 // 2)
            scales2 = (self.num_experts, out2, in2 // 16)
            scale_dtype = torch.float8_e4m3fn
        else:
            payload13 = (self.num_experts, out13, in13)
            scales13 = (self.num_experts, out13, in13 // 32)
            payload2 = (self.num_experts, out2, in2)
            scales2 = (self.num_experts, out2, in2 // 32)
            scale_dtype = torch.uint8

        self.gate_up_weight = nn.Parameter(
            torch.empty(payload13, dtype=torch.uint8 if fmt == "nvfp4" else torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.gate_up_scales = nn.Parameter(
            torch.empty(scales13, dtype=scale_dtype), requires_grad=False
        )
        self.down_weight = nn.Parameter(
            torch.empty(payload2, dtype=torch.uint8 if fmt == "nvfp4" else torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.down_scales = nn.Parameter(
            torch.empty(scales2, dtype=scale_dtype), requires_grad=False
        )
        if fmt == "nvfp4":
            self.gate_up_amax = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32), requires_grad=False
            )
            self.gate_up_outer_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
            self.down_amax = nn.Parameter(
                torch.empty(self.num_experts, dtype=torch.float32), requires_grad=False
            )
            self.down_outer_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )

    def _global_scale(self, amax: torch.Tensor, outer: torch.Tensor, expert_idx: int) -> torch.Tensor:
        return amax[expert_idx].float() / outer.float().reshape(())

    def _matmul(self, x: torch.Tensor, weight, scales, global_scale=None) -> torch.Tensor:
        if scaled_wxa16_available(x.device):
            if self.fmt == "nvfp4":
                return nvfp4_w4a16_linear(x, weight, scales, global_scale)
            return mxfp8_w8a16_linear(x, weight, scales)
        # Reference decode for CPU / pre-Ampere: same math, dense tile.
        if self.fmt == "nvfp4":
            dense = dequant_nvfp4(weight, scales, global_scale)
        else:
            dense = dequant_mxfp8(weight, scales)
        return F.linear(x, dense.to(x.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx, tokens, slots in _route(
            hidden_states, top_k_index, top_k_weights, self.num_experts
        ):
            current = hidden_states[tokens]
            if self.fmt == "nvfp4":
                gate_up = self._matmul(
                    current,
                    self.gate_up_weight[expert_idx],
                    self.gate_up_scales[expert_idx],
                    self._global_scale(self.gate_up_amax, self.gate_up_outer_scale, expert_idx),
                )
            else:
                gate_up = self._matmul(
                    current, self.gate_up_weight[expert_idx], self.gate_up_scales[expert_idx]
                )
            gate, up = gate_up.chunk(2, dim=-1)
            current = self.act_fn(gate, up)
            if self.fmt == "nvfp4":
                current = self._matmul(
                    current,
                    self.down_weight[expert_idx],
                    self.down_scales[expert_idx],
                    self._global_scale(self.down_amax, self.down_outer_scale, expert_idx),
                )
            else:
                current = self._matmul(
                    current, self.down_weight[expert_idx], self.down_scales[expert_idx]
                )
            current = current * top_k_weights[tokens, slots, None]
            final_hidden_states.index_add_(0, tokens, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    @torch.no_grad()
    def dequantize(self) -> "Inkling2LoopExperts":
        """Materialize a dense container (used for export and as a numerical oracle)."""

        gate_up, down = [], []
        for expert_idx in range(self.num_experts):
            if self.fmt == "nvfp4":
                gs = self._global_scale(self.gate_up_amax, self.gate_up_outer_scale, expert_idx)
                gate_up.append(
                    dequant_nvfp4(
                        self.gate_up_weight[expert_idx], self.gate_up_scales[expert_idx], gs
                    )
                )
                gs = self._global_scale(self.down_amax, self.down_outer_scale, expert_idx)
                down.append(
                    dequant_nvfp4(self.down_weight[expert_idx], self.down_scales[expert_idx], gs)
                )
            else:
                gate_up.append(
                    dequant_mxfp8(self.gate_up_weight[expert_idx], self.gate_up_scales[expert_idx])
                )
                down.append(
                    dequant_mxfp8(self.down_weight[expert_idx], self.down_scales[expert_idx])
                )
        dense = Inkling2LoopExperts(self.config)
        dense.gate_up_proj = nn.Parameter(torch.stack(gate_up).to(torch.bfloat16))
        dense.down_proj = nn.Parameter(torch.stack(down).to(torch.bfloat16))
        return dense


def build_experts(config, layer_idx: int) -> nn.Module:
    """Return the container the checkpoint stores for ``layer_idx``."""

    fmt: Optional[str] = config.experts_format(layer_idx)
    if fmt is None:
        return Inkling2LoopExperts(config)
    return Inkling2LoopPackedExperts(config, fmt)


__all__ = [
    "Inkling2LoopExperts",
    "Inkling2LoopPackedExperts",
    "Inkling2LoopSoftcap",
    "Inkling2LoopSoftcappedSwiGLU",
    "build_experts",
]
