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

    :meth:`Inkling2LoopPackedExperts.defuse` splits the fused 3-D payload into
    one :class:`Inkling2LoopPackedExpertLinear` per expert and projection. That
    is the layout a quantizer consumes: each leaf is a real ``nn.Linear`` for
    planning and hooks, it keeps the packed payload for the calibration forward
    (JIT decode to bf16 activations, no dense weight), and only materializes the
    dense bf16 weight when the quantization processor asks for it.
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
from ...utils.refusion import RefusionContractError, assert_refusable


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

    def forward(
        self, gate: torch.Tensor, up: torch.Tensor, routing_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = gate.dtype
        gate, up = gate.float(), up.float()
        activated = self.gate_softcap(gate) * torch.sigmoid(gate) * self.up_softcap(up)
        if routing_weight is not None:
            activated = activated * routing_weight.float()
        return activated.to(dtype)


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
        return self._forward_routed(hidden_states, top_k_index, top_k_weights)

    def _forward_routed(
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
            routing_weight = top_k_weights[tokens, slots, None]
            expert = self._modules.get(str(expert_idx))
            if expert is None:
                current = F.linear(current, self.gate_up_proj[expert_idx])
                current = self._apply_gate(current, routing_weight)
                current = F.linear(current, self.down_proj[expert_idx])
            else:
                current = self.act_fn(expert.gate_proj(current), expert.up_proj(current), routing_weight)
                current = expert.down_proj(current)
            final_hidden_states.index_add_(0, tokens, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    def _apply_gate(
        self, gate_up: torch.Tensor, routing_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate, up, routing_weight)


class Inkling2LoopPackedExpertLinear(nn.Linear):
    """One expert projection that keeps the checkpoint's packed storage.

    The class is a real ``nn.Linear`` so module planning, hook installation and
    the quantizer's ``weight`` contract work unchanged, but it does not own a
    dense weight by default:

    - ``forward`` decodes inside the GEMM (:func:`nvfp4_w4a16_linear` /
      :func:`mxfp8_w8a16_linear`) and returns bf16 activations, so calibration
      and Hessian collection never materialize a dense weight.
    - ``weight`` is materialized on first access. A quantization processor reads
      it when it starts packing, which hands it the fully dequantized bf16
      source instead of the packed payload.
    """

    #: Marker consumed by ``replace_module_with_hooked_legacy``: the module keeps
    #: its kernel forward instead of becoming a dense ``HookedLinear``.
    _packed_source_linear = True

    #: Payload and decode factors are kernel storage, not activation operands.
    CHECKPOINT_EXACT_DTYPE_PARAMETERS = (
        "packed_weight",
        "packed_scales",
        "packed_global_scale",
    )

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        fmt: str,
        weight: torch.Tensor,
        scales: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
    ) -> None:
        # ``nn.Linear.__init__`` would allocate the dense weight this class
        # exists to avoid; only the module bookkeeping is needed.
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fmt = fmt
        self.packed_weight = nn.Parameter(weight, requires_grad=False)
        self.packed_scales = nn.Parameter(scales, requires_grad=False)
        if global_scale is None:
            self.packed_global_scale = None
        else:
            self.packed_global_scale = nn.Parameter(
                global_scale.detach().reshape(()).float(), requires_grad=False
            )
        self.bias = None

    def __getattr__(self, name: str):
        """Materialize ``weight`` on first access instead of at construction."""

        if name == "weight":
            parameters = self.__dict__.get("_parameters")
            if parameters is not None and "weight" in parameters:
                return parameters["weight"]
            return self._materialize_dense_weight()
        return super().__getattr__(name)

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        """Decode the packed payload into the dense bf16 quantization source."""

        if self.fmt == "nvfp4":
            dense = dequant_nvfp4(
                self.packed_weight, self.packed_scales, self.packed_global_scale
            )
        else:
            dense = dequant_mxfp8(self.packed_weight, self.packed_scales)
        return dense.to(torch.bfloat16)

    @torch.no_grad()
    def _materialize_dense_weight(self) -> nn.Parameter:
        cached = self._parameters.get("_dense_weight")
        if cached is not None:
            return cached
        weight = nn.Parameter(self.dequantize_weight(), requires_grad=False)
        self._dense_weight = weight
        return weight

    def dense_weight(self) -> torch.Tensor:
        """Return the fully dequantized bf16 weight (the quantization source)."""

        return self._materialize_dense_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if scaled_wxa16_available(x.device):
            if self.fmt == "nvfp4":
                return nvfp4_w4a16_linear(
                    x,
                    self.packed_weight,
                    self.packed_scales,
                    self.packed_global_scale,
                )
            return mxfp8_w8a16_linear(x, self.packed_weight, self.packed_scales)
        # No fused decode kernel on this device: fall back to the reference
        # dense bf16 product (requirement: dequantize when no kernel exists).
        dense = self.dequantize_weight().to(x.dtype)
        return F.linear(x, dense)


class Inkling2LoopExpertProjections(nn.Module):
    """Per-expert ``gate_proj`` / ``up_proj`` / ``down_proj`` leaves."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj


class Inkling2LoopPackedExperts(nn.Module):
    """Routed experts kept in the checkpoint's block-scaled storage.

    ``gate_up_weight`` / ``down_weight`` are the packed payload, ``*_scales`` the
    per-block scales, and for NVFP4 ``*_amax`` / ``*_outer_scale`` the per-expert
    and per-tensor factors that give the FP32 global scale. The activation is
    never quantized: it stays in the model dtype and the dot accumulates in FP32.

    :meth:`defuse` rewrites the container into the per-expert layout a quantizer
    needs (``experts.<i>.gate_proj`` / ``up_proj`` / ``down_proj``) without
    decoding anything: each leaf keeps a view of the packed payload.
    """

    # The payload and every factor that decodes it are kernel storage, not
    # activation operands. A model-dtype cast would narrow the FP32 global
    # scales and reinterpret the E4M3/E8M0 block scales, so the loader keeps
    # them at exactly the checkpoint dtype.
    CHECKPOINT_EXACT_DTYPE_PARAMETERS = (
        "gate_up_weight",
        "down_weight",
        "gate_up_scales",
        "down_scales",
        "gate_up_amax",
        "gate_up_outer_scale",
        "down_amax",
        "down_outer_scale",
    )

    def __init__(self, config, fmt: str):
        super().__init__()
        if fmt not in ("nvfp4", "mxfp8"):
            raise ValueError(f"unsupported expert format {fmt!r}")
        self.config = config
        self.fmt = fmt
        self._defused = False
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

    @property
    def defused(self) -> bool:
        """Whether the fused payload was already split into per-expert leaves."""

        return self._defused

    @torch.no_grad()
    def defuse(self) -> "Inkling2LoopPackedExperts":
        """Split the fused 3-D payload into per-expert projections, in place.

        The checkpoint alternates gate and up rows for each intermediate
        channel. Every leaf keeps a view of that storage (no copy, no decode),
        so the calibration forward still runs the packed W4A16/W8A16 kernels.
        The fused parameters are dropped afterwards; the views keep their
        storage alive.
        """

        if self._defused:
            return self

        gate_amax = getattr(self, "gate_up_amax", None)
        gate_outer = getattr(self, "gate_up_outer_scale", None)
        down_amax = getattr(self, "down_amax", None)
        down_outer = getattr(self, "down_outer_scale", None)
        # Scalars only: keep the original per-expert/per-tensor factors so
        # `refuse()` can put them back without recomputing the product.
        self._fused_factors = {
            name: getattr(self, name, None)
            for name in (
                "gate_up_amax",
                "gate_up_outer_scale",
                "down_amax",
                "down_outer_scale",
            )
        }
        for expert_idx in range(self.num_experts):
            gate_up_weight = self.gate_up_weight[expert_idx]
            gate_up_scales = self.gate_up_scales[expert_idx]
            gate = Inkling2LoopPackedExpertLinear(
                in_features=self.hidden_dim,
                out_features=self.intermediate_dim,
                fmt=self.fmt,
                weight=gate_up_weight[0::2],
                scales=gate_up_scales[0::2],
                global_scale=None
                if gate_amax is None
                else gate_amax[expert_idx] / gate_outer.reshape(()),
            )
            up = Inkling2LoopPackedExpertLinear(
                in_features=self.hidden_dim,
                out_features=self.intermediate_dim,
                fmt=self.fmt,
                weight=gate_up_weight[1::2],
                scales=gate_up_scales[1::2],
                global_scale=None
                if gate_amax is None
                else gate_amax[expert_idx] / gate_outer.reshape(()),
            )
            down = Inkling2LoopPackedExpertLinear(
                in_features=self.intermediate_dim,
                out_features=self.hidden_dim,
                fmt=self.fmt,
                weight=self.down_weight[expert_idx],
                scales=self.down_scales[expert_idx],
                global_scale=None
                if down_amax is None
                else down_amax[expert_idx] / down_outer.reshape(()),
            )
            self.add_module(
                str(expert_idx), Inkling2LoopExpertProjections(gate, up, down)
            )

        for name in (
            "gate_up_weight",
            "gate_up_scales",
            "down_weight",
            "down_scales",
            "gate_up_amax",
            "gate_up_outer_scale",
            "down_amax",
            "down_outer_scale",
        ):
            if name in self._parameters:
                delattr(self, name)
        self._defused = True
        return self

    @torch.no_grad()
    def refuse(self, target: Optional[str] = None) -> "Inkling2LoopPackedExperts":
        """Reassemble the fused packed storage from untouched per-expert leaves.

        A layer whose experts the quantization plan skipped must keep the
        checkpoint's packed declaration, so its leaves are re-fused before the
        save instead of being written in a layout the packed declaration cannot
        describe. Re-fusion is only sound while every leaf still holds this
        container's own packed payload with one identical quantization: one
        format, bit width, block size and dtype. That contract
        (``contracts/model/refusion.md``) is enforced before any tensor is
        created, so a rejected layer is left exactly as it was.
        """

        if not self._defused:
            return self

        leaves = []
        labelled = []
        for expert_idx in range(self.num_experts):
            expert = getattr(self, str(expert_idx))
            projections = (expert.gate_proj, expert.up_proj, expert.down_proj)
            leaves.append(projections)
            labelled.extend(
                (f"expert {expert_idx}.{name}", projection)
                for name, projection in zip(("gate_proj", "up_proj", "down_proj"), projections)
            )

        owner = target or f"Inkling2Loop packed experts ({self.num_experts} experts)"
        signature = assert_refusable(owner, labelled)
        if signature.format != self.fmt:
            # The container's declaration is part of the checkpoint: a layer
            # cannot claim one format and store another.
            raise RefusionContractError(
                f"{owner} cannot be re-fused: its leaves are {signature.format}, but the "
                f"container declares {self.fmt} (contracts/model/refusion.md R1)."
            )

        self.gate_up_weight = nn.Parameter(
            torch.stack(
                [
                    torch.stack([gate.packed_weight, up.packed_weight], dim=1).flatten(0, 1)
                    for gate, up, _ in leaves
                ]
            ),
            requires_grad=False,
        )
        self.gate_up_scales = nn.Parameter(
            torch.stack(
                [
                    torch.stack([gate.packed_scales, up.packed_scales], dim=1).flatten(0, 1)
                    for gate, up, _ in leaves
                ]
            ),
            requires_grad=False,
        )
        self.down_weight = nn.Parameter(
            torch.stack([down.packed_weight for _, _, down in leaves]),
            requires_grad=False,
        )
        self.down_scales = nn.Parameter(
            torch.stack([down.packed_scales for _, _, down in leaves]),
            requires_grad=False,
        )
        for name, value in getattr(self, "_fused_factors", {}).items():
            if value is not None:
                setattr(self, name, nn.Parameter(value.detach(), requires_grad=False))

        for expert_idx in range(self.num_experts):
            delattr(self, str(expert_idx))
        self._defused = False
        return self

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
            routing_weight = top_k_weights[tokens, slots, None]
            if self._defused:
                expert = getattr(self, str(expert_idx))
                current = self.act_fn(expert.gate_proj(current), expert.up_proj(current), routing_weight)
                current = expert.down_proj(current)
                final_hidden_states.index_add_(
                    0, tokens, current.to(final_hidden_states.dtype)
                )
                continue
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
            gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
            current = self.act_fn(gate, up, routing_weight)
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
            final_hidden_states.index_add_(0, tokens, current.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)

    @torch.no_grad()
    def dequantize(self) -> "Inkling2LoopExperts":
        """Materialize a dense container (used for export and as a numerical oracle)."""

        gate_up, down = [], []
        for expert_idx in range(self.num_experts):
            if self.fmt == "nvfp4":
                gs = self._global_scale(self.gate_up_amax, self.gate_up_outer_scale, expert_idx)
                interleaved = dequant_nvfp4(
                    self.gate_up_weight[expert_idx], self.gate_up_scales[expert_idx], gs
                )
                gs = self._global_scale(self.down_amax, self.down_outer_scale, expert_idx)
                down.append(
                    dequant_nvfp4(self.down_weight[expert_idx], self.down_scales[expert_idx], gs)
                )
            else:
                interleaved = dequant_mxfp8(self.gate_up_weight[expert_idx], self.gate_up_scales[expert_idx])
                down.append(
                    dequant_mxfp8(self.down_weight[expert_idx], self.down_scales[expert_idx])
                )
            # Dense experts use separate contiguous halves for Defuser.
            gate_up.append(torch.cat((interleaved[0::2], interleaved[1::2]), dim=0))
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


def defuse_packed_experts(module: nn.Module) -> int:
    """Split every packed expert container under ``module`` into per-expert leaves.

    Called once a layer is materialized and before its modules are resolved, so
    the quantizer sees ``experts.<i>.gate_proj`` / ``up_proj`` / ``down_proj``
    for a pre-quantized checkpoint exactly as it does for a dense one.
    """

    converted = 0
    for _, child in list(module.named_children()):
        if isinstance(child, Inkling2LoopPackedExperts):
            if not child.defused:
                child.defuse()
                converted += 1
            continue
        converted += defuse_packed_experts(child)
    return converted


__all__ = [
    "Inkling2LoopExpertProjections",
    "Inkling2LoopExperts",
    "Inkling2LoopPackedExpertLinear",
    "Inkling2LoopPackedExperts",
    "Inkling2LoopSoftcap",
    "Inkling2LoopSoftcappedSwiGLU",
    "build_experts",
    "defuse_packed_experts",
]
