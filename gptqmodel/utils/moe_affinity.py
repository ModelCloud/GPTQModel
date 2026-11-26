# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Routing-affinity capture for MoEQuant Affinity-Guided Quantization (AGQ).

Clean-room implementation of the affinity-weighted loss (Eqs. 18-19) in:

    Zhixuan Chen, Xing Hu, Dawei Yang, Zukang Xu, Xu Chen, Zhihang Yuan,
    Sifan Zhou, and Jiangyong Yu.  "MoEQuant: Enhancing Quantization for
    Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and
    Affinity Guidance."  ICML 2025, PMLR 267:8245-8260.
    https://arxiv.org/abs/2505.03804

Each routed token's quantization error is weighted by the router affinity
that the experts container receives. The first input projection identifies the
dispatch order for all projections in that expert during the same forward.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class _RoutingEntry:
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor
    rows_by_expert: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def _argument(args: Tuple[Any, ...], kwargs: Dict[str, Any], index: int, *names: str) -> Any:
    for name in names:
        value = kwargs.get(name)
        if value is not None:
            return value
    if index < len(args):
        return args[index]
    return None


def _flatten_routes(tensor: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Put one token on each row, with one column per selected expert."""
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    if tensor.ndim == 3:
        if hidden_states.ndim == 3 and tensor.shape[:2] == hidden_states.shape[:2]:
            return tensor.flatten(0, 1)
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
        if tensor.shape[1] == 1:
            return tensor.squeeze(1)
    return tensor


class MoEAffinityCapture:
    """Thread-local routing and row order for each experts container forward."""

    def __init__(self) -> None:
        self._tls = threading.local()
        self._experts: Dict[str, Tuple[str, int]] = {}

    def has_expert(self, name: str) -> bool:
        return name in self._experts

    def register_expert(self, name: str, container: str, expert_index: int) -> None:
        # Subset preprocessing registers these names before the workers replay.
        self._experts[name] = (container, expert_index)

    def _routing(self) -> Dict[str, _RoutingEntry]:
        routing = getattr(self._tls, "routing", None)
        if routing is None:
            routing = {}
            self._tls.routing = routing
        return routing

    def _routing_entry(self, container_name: str) -> _RoutingEntry:
        entry = self._routing().get(container_name)
        if entry is None:
            raise RuntimeError(
                f"MoEQuant AGQ: routing capture for `{container_name}` is missing; "
                "the experts container must run through its capture hook."
            )
        return entry

    def make_pre_hook(self, container_name: str):
        def hook(module, args, kwargs):
            self.record(container_name, module, args, kwargs)
        return hook

    def make_post_hook(self, container_name: str):
        def hook(module, args, kwargs, output):
            self.clear(container_name)
        return hook

    def make_expert_input_hook(self, container_name: str, expert_index: int):
        def hook(module, args, kwargs):
            self.observe_expert_input(
                container_name, expert_index, _argument(args, kwargs, 0, "input", "hidden_states")
            )
        return hook

    def record(
        self,
        container_name: str,
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        hidden_states = _argument(args, kwargs, 0, "hidden_states")
        top_k_index = _argument(args, kwargs, 1, "top_k_index", "selected_experts")
        top_k_weights = _argument(args, kwargs, 2, "top_k_weights", "routing_weights")
        missing = [
            name
            for name, value in (
                ("hidden_states", hidden_states),
                ("top_k_index", top_k_index),
                ("top_k_weights", top_k_weights),
            )
            if not torch.is_tensor(value)
        ]
        if missing:
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{type(module).__name__}` "
                f"did not expose {missing}. Affinity weighting requires native routes."
            )
        if hidden_states.ndim not in (2, 3):
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{container_name}` received "
                f"{tuple(hidden_states.shape)} hidden states; expected 2-D or 3-D."
            )

        top_k_index = _flatten_routes(top_k_index, hidden_states)
        top_k_weights = _flatten_routes(top_k_weights, hidden_states)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.flatten(0, 1)
        if (
            top_k_index.ndim != 2
            or top_k_index.shape != top_k_weights.shape
            or top_k_index.shape[0] != hidden_states.shape[0]
        ):
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{container_name}` received routing "
                f"shapes {tuple(top_k_index.shape)} and {tuple(top_k_weights.shape)} "
                f"for {hidden_states.shape[0]} tokens; expected matching [tokens, top_k] rows."
            )
        if top_k_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise RuntimeError("MoEQuant AGQ: routing indices must be integers.")
        if (top_k_index < 0).any():
            raise RuntimeError("MoEQuant AGQ: router produced negative expert indices.")
        self._routing()[container_name] = _RoutingEntry(
            hidden_states=hidden_states.detach(),
            top_k_index=top_k_index.detach(),
            top_k_weights=top_k_weights.detach().to(device=top_k_index.device),
        )

    def clear(self, container_name: str) -> None:
        self._routing().pop(container_name, None)

    def observe_expert_input(
        self, container_name: str, expert_index: int, inp: torch.Tensor
    ) -> None:
        entry = self._routing_entry(container_name)
        if expert_index in entry.rows_by_expert:
            return
        if not torch.is_tensor(inp) or inp.ndim != 2:
            raise RuntimeError("MoEQuant AGQ: expert input must be a 2-D routed tensor.")

        # The eager routed loop is slot-major and Defuser's linear loop is
        # token-major. Validate against the actual input before weighting a Hessian.
        selected = entry.top_k_index == expert_index
        slots, tokens = torch.where(selected.T)
        expected = entry.hidden_states[tokens.to(entry.hidden_states.device)].to(
            device=inp.device, dtype=inp.dtype
        )
        if not torch.equal(expected, inp):
            tokens, slots = torch.where(selected)
            expected = entry.hidden_states[tokens.to(entry.hidden_states.device)].to(
                device=inp.device, dtype=inp.dtype
            )
            if not torch.equal(expected, inp):
                raise RuntimeError(
                    f"MoEQuant AGQ: routed rows for `{container_name}.{expert_index}` "
                    "do not match the captured routing order."
                )

        weights = entry.top_k_weights[tokens, slots].to(device=inp.device, dtype=torch.float32)
        if not torch.isfinite(weights).all():
            raise RuntimeError(
                f"MoEQuant AGQ: expert `{container_name}.{expert_index}` received non-finite affinity weights."
            )
        if (weights < 0).any():
            raise RuntimeError(
                f"MoEQuant AGQ: expert `{container_name}.{expert_index}` received negative affinity weights."
            )
        entry.rows_by_expert[expert_index] = (tokens, weights)

    def calibration_rows(
        self, name: str, inp: torch.Tensor, keep_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the expert's calibration inputs and affinities in matching order."""
        container_name, expert_index = self._experts[name]
        entry = self._routing_entry(container_name)
        routed = entry.rows_by_expert.get(expert_index)
        if routed is None:
            raise RuntimeError(
                f"MoEQuant AGQ: `{name}` ran before a routed input projection established its row order."
            )
        tokens, weights = routed
        if not torch.is_tensor(inp) or inp.ndim != 2 or inp.shape[0] != tokens.numel():
            raise RuntimeError(
                f"MoEQuant AGQ: `{name}` received input "
                f"{tuple(inp.shape) if torch.is_tensor(inp) else type(inp).__name__} "
                f"for {tokens.numel()} routed rows."
            )
        weights = weights.to(device=inp.device)
        if keep_mask is None:
            return inp, weights
        if keep_mask.numel() != entry.hidden_states.shape[0]:
            raise RuntimeError(
                f"MoEQuant AGQ: calibration mask for `{name}` does not match the experts container's token rows."
            )
        keep_rows = keep_mask.reshape(-1).to(device=tokens.device, dtype=torch.bool)[tokens].to(inp.device)
        return inp[keep_rows].contiguous(), weights[keep_rows]
