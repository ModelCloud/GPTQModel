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

The paper weights each routed token's quantization error by the router weight
``c_i`` that the MoE block applies to that expert's output.  GPTQ consumes only
``H = X^T X``, so AGQ is exactly a per-row scale ``sqrt(c_i)`` on the expert's
calibration inputs.

This module captures the routing weights that the model's own experts container
receives.  It deliberately does not re-derive softmax/top-k normalisation from
router logits: ports differ in normalisation, and a wrong re-derivation would
silently weight the Hessian with the wrong coefficients.  If the container does
not expose ``top_k_index`` and ``top_k_weights``, capture fails closed.

Row order is the order the Defuser/native expert loop uses: experts are visited
in ascending index and each expert's rows are ordered by top-k slot and then by
token (``torch.where`` over the ``[top_k, tokens]`` mask).  For the first
projection the captured rows are compared with the container's hidden states so
a different layout raises instead of corrupting the Hessian.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .logger import setup_logger


log = setup_logger()


@dataclass
class _RoutingEntry:
    hidden_states: torch.Tensor
    top_k_index: torch.Tensor
    top_k_weights: torch.Tensor
    num_experts: Optional[int]


def _argument(args: Tuple[Any, ...], kwargs: Dict[str, Any], index: int, *names: str) -> Any:
    if isinstance(kwargs, dict):
        for name in names:
            value = kwargs.get(name)
            if value is not None:
                return value
    if len(args) > index:
        return args[index]
    return None


class MoEAffinityCapture:
    """Thread-local capture of one MoE container's routing for one forward."""

    def __init__(self) -> None:
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._experts: Dict[str, Tuple[str, int]] = {}

    @property
    def experts(self) -> Dict[str, Tuple[str, int]]:
        return self._experts

    def has_expert(self, name: str) -> bool:
        return name in self._experts

    def register_expert(self, name: str, container: str, expert_index: int) -> None:
        if not isinstance(container, str) or not container:
            raise ValueError("MoEQuant AGQ: expert container name must be a non-empty string.")
        if isinstance(expert_index, bool) or not isinstance(expert_index, int) or expert_index < 0:
            raise ValueError("MoEQuant AGQ: expert index must be a non-negative integer.")
        with self._lock:
            self._experts[name] = (container, expert_index)

    def _routing(self) -> Dict[str, _RoutingEntry]:
        routing = getattr(self._tls, "routing", None)
        if routing is None:
            routing = {}
            self._tls.routing = routing
        return routing

    def make_pre_hook(self, container_name: str):
        def hook(module, args, kwargs):
            self.record(container_name, module, args, kwargs)
        return hook

    def make_post_hook(self, container_name: str):
        def hook(module, args, kwargs, output):
            self.clear(container_name)
            return None
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
            signature = ", ".join(sorted(kwargs)) if isinstance(kwargs, dict) else ""
            raise RuntimeError(
                "MoEQuant AGQ: experts container "
                f"`{type(module).__name__}` did not expose {missing} "
                f"(positional args={len(args)}, kwargs=[{signature}]). "
                "Affinity weighting cannot be applied safely."
            )
        self._routing()[container_name] = _RoutingEntry(
            hidden_states=hidden_states.detach(),
            top_k_index=top_k_index.detach(),
            top_k_weights=top_k_weights.detach(),
            num_experts=getattr(module, "num_experts", None),
        )

    def clear(self, container_name: str) -> None:
        self._routing().pop(container_name, None)

    def weights_for(self, name: str, inp: torch.Tensor) -> torch.Tensor:
        """Return the per-row affinity weights aligned with ``inp`` rows."""

        registered = self._experts.get(name)
        if registered is None:
            raise RuntimeError(
                f"MoEQuant AGQ: `{name}` is not registered as a routed expert."
            )
        container_name, expert_index = registered
        entry = self._routing().get(container_name)
        if entry is None:
            raise RuntimeError(
                f"MoEQuant AGQ: routing capture for `{container_name}` is missing; "
                "the experts container was not called through the captured hook."
            )
        if not torch.is_tensor(inp) or inp.dim() != 2:
            raise RuntimeError(
                f"MoEQuant AGQ: routed expert `{name}` received a "
                f"{tuple(inp.shape) if torch.is_tensor(inp) else type(inp).__name__} "
                "input; only 2-D routed rows are supported."
            )

        hidden_states = entry.hidden_states
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        elif hidden_states.dim() != 2:
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{container_name}` received "
                f"{tuple(hidden_states.shape)} hidden states; expected 2-D or 3-D."
            )

        top_k_index = entry.top_k_index
        if top_k_index.dim() == 1:
            top_k_index = top_k_index.reshape(-1, 1)
        elif top_k_index.dim() == 2:
            pass
        elif top_k_index.dim() == 3 and top_k_index.shape[-1] == 1:
            top_k_index = top_k_index.squeeze(-1)
        elif top_k_index.dim() == 3 and top_k_index.shape[1] == 1:
            top_k_index = top_k_index.squeeze(1)
        else:
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{container_name}` exposed routing "
                f"indices with shape {tuple(top_k_index.shape)}; expected "
                "[tokens, top_k] or a singleton expert axis."
            )

        top_k_weights = entry.top_k_weights
        if top_k_weights.dim() == 1:
            top_k_weights = top_k_weights.reshape(-1, 1)
        elif top_k_weights.dim() == 3 and top_k_weights.shape[-1] == 1:
            top_k_weights = top_k_weights.squeeze(-1)
        elif top_k_weights.dim() == 3 and top_k_weights.shape[1] == 1:
            top_k_weights = top_k_weights.squeeze(1)
        elif top_k_weights.dim() != 2:
            raise RuntimeError(
                f"MoEQuant AGQ: experts container `{container_name}` exposed routing "
                f"weights with shape {tuple(top_k_weights.shape)}; expected "
                "[tokens, top_k] or a singleton expert axis."
            )
        if top_k_weights.shape[0] != top_k_index.shape[0]:
            raise RuntimeError(
                f"MoEQuant AGQ: routing indices and weights disagree on token count "
                f"({top_k_index.shape[0]} vs {top_k_weights.shape[0]})."
            )
        if top_k_weights.shape[1] < top_k_index.shape[1]:
            raise RuntimeError(
                f"MoEQuant AGQ: routing weights have {top_k_weights.shape[1]} slots "
                f"but indices select {top_k_index.shape[1]} experts."
            )
        if top_k_weights.device != top_k_index.device:
            top_k_weights = top_k_weights.to(top_k_index.device)
        if top_k_index.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=inp.device)

        index = top_k_index.to(dtype=torch.long)
        if (index < 0).any():
            raise RuntimeError("MoEQuant AGQ: router produced negative expert indices.")
        max_index = int(index.max().item())
        num_experts = entry.num_experts if isinstance(entry.num_experts, int) else 0
        num_classes = max(num_experts, max_index + 1, expert_index + 1)

        expert_mask = F.one_hot(index, num_classes=num_classes).permute(2, 1, 0)
        top_k_pos, token_idx = torch.where(expert_mask[expert_index])
        row_weights = top_k_weights[token_idx, top_k_pos].to(
            device=inp.device, dtype=torch.float32
        )
        token_major_token, token_major_pos = torch.where(index == expert_index)
        token_major_weights = top_k_weights[token_major_token, token_major_pos].to(
            device=inp.device, dtype=torch.float32
        )

        if row_weights.numel() != inp.shape[0]:
            if token_major_weights.numel() == inp.shape[0]:
                row_weights = token_major_weights
                token_idx = token_major_token
            else:
                raise RuntimeError(
                    f"MoEQuant AGQ: expert `{name}` captured {row_weights.numel()} "
                    f"affinity weights for {inp.shape[0]} routed rows."
                )

        # The first expert projection consumes the MoE block's hidden states;
        # its rows identify the routing order exactly.  `down_proj` consumes the
        # expert's intermediate activation, so only its row count is checked.
        projection = name.rsplit(".", 1)[-1]
        if projection != "down_proj" and inp.shape[-1] == hidden_states.shape[-1]:
            expected = hidden_states[token_idx.to(hidden_states.device)].to(
                device=inp.device, dtype=inp.dtype
            )
            if expected.shape != inp.shape or not torch.equal(expected, inp):
                # Some ports use token-major order.  Try it before failing.
                token_major_expected = hidden_states[
                    token_major_token.to(hidden_states.device)
                ].to(device=inp.device, dtype=inp.dtype)
                if (
                    token_major_weights.numel() == inp.shape[0]
                    and token_major_expected.shape == inp.shape
                    and torch.equal(token_major_expected, inp)
                ):
                    row_weights = token_major_weights
                else:
                    raise RuntimeError(
                        f"MoEQuant AGQ: routed rows for expert `{name}` do not match "
                        "the captured routing order; refusing to weight the Hessian."
                    )

        if not torch.isfinite(row_weights).all():
            raise RuntimeError(
                f"MoEQuant AGQ: expert `{name}` received non-finite affinity weights."
            )
        if bool((row_weights < 0).any()):
            raise RuntimeError(
                f"MoEQuant AGQ: expert `{name}` received negative affinity weights."
            )
        return row_weights
