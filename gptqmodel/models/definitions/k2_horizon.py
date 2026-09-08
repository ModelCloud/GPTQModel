# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

import torch

from ...nn_modules.hooked_linear import StopForward
from ...utils.device import get_device
from ...utils.model import move_to
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks, _get_module_by_relative_path


class K2HorizonMoELifecycleHooks(GateUpDownMoELifecycleHooks):
    """Quantization replay hooks for K2's MLP MoE and MoVA value experts."""

    def get_moe_block_for_subset(
        self,
        layer_module: torch.nn.Module,
        model_class: type,
        current_subset: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.nn.Module]:
        # ``module_tree`` keeps ``mlp`` as the canonical :moe root for generic
        # MoE handling.  MoVA value experts are a second family in attention;
        # select that block only when the active subset contains its leaves.
        if current_subset and any(
            name.startswith("self_attn.v_experts.") for name in current_subset
        ):
            return getattr(layer_module, "self_attn", None)
        return super().get_moe_block_for_subset(
            layer_module,
            model_class,
            current_subset=current_subset,
        )

    def _extract_moe_block_prefix(
        self,
        subset: Dict[str, Any],
        moe_block: torch.nn.Module,
    ) -> Optional[str]:
        if any(name.startswith("self_attn.v_experts.") for name in subset or ()):
            return "self_attn"
        return super()._extract_moe_block_prefix(subset, moe_block)

    def forward_to_all_experts(
        self,
        moe_block: torch.nn.Module,
        hidden_states: torch.Tensor,
        processor: Any,
        subset: Dict[str, Any],
        ordered_module_names: Optional[list[str]],
        original_forward: callable,
        model_class: type,
        module_looper: Any,
        moe_block_prefix: Optional[str] = None,
        replica_module: Optional[torch.nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        v_experts = getattr(moe_block, "v_experts", None)
        if v_experts is None:
            return super().forward_to_all_experts(
                moe_block=moe_block,
                hidden_states=hidden_states,
                processor=processor,
                subset=subset,
                ordered_module_names=ordered_module_names,
                original_forward=original_forward,
                model_class=model_class,
                module_looper=module_looper,
                moe_block_prefix=moe_block_prefix,
                replica_module=replica_module,
                **kwargs,
            )

        if hidden_states.dim() == 3:
            expert_input = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            expert_input = hidden_states

        prefix = moe_block_prefix or "self_attn"
        expert_count = 0
        stop_forward_raised = False

        for expert_idx, expert in enumerate(v_experts):
            expert_name = f"{prefix}.v_experts.{expert_idx}"
            if expert_name not in subset:
                continue

            runtime_expert = expert
            if replica_module is not None:
                runtime_expert = _get_module_by_relative_path(
                    replica_module, expert_name
                )
            if runtime_expert is None:
                runtime_expert = subset[expert_name]

            try:
                runtime_device = get_device(runtime_expert)
                runtime_expert(move_to(expert_input, runtime_device))
                expert_count += 1
            except StopForward:
                stop_forward_raised = True

        if stop_forward_raised:
            raise StopForward()

        # The native MoVA forward routes through v_experts again.  Pause the
        # processor hooks for that final call so each selected leaf is counted
        # exactly once while preserving the model's true attention output.
        if expert_count > 0:
            module_looper._set_processor_hooks_paused(processor, True)
            try:
                return original_forward(hidden_states, **kwargs)
            finally:
                module_looper._set_processor_hooks_paused(processor, False)
        return original_forward(hidden_states, **kwargs)


class K2HorizonQModel(BaseQModel):
    """GPT-QModel definition for K2 Horizon dense and MoVA checkpoints."""

    require_trust_remote_code = True
    layer_modules_strict = False
    dynamic_expert_index = "num_experts"
    dynamic_expert_indices = {
        "mlp.experts": "num_experts",
        "self_attn.v_experts": "mova_num_experts",
    }

    pre_lm_head_norm_module = "model.norm"
    moe_expert_module_name_prefixes = [".expert", ".v_experts."]
    moe_lifecycle_hooks = K2HorizonMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": {
                "q_norm": ("q_norm:!",),
                "k_norm": ("k_norm:!",),
                "q_proj": ("q_proj:0",),
                "k_proj": ("k_proj:0",),
                # Dense layers expose v_proj; MoVA layers expose routed value
                # experts instead.  Strict=False makes this one tree valid for
                # both layer variants.
                "v_proj": ("v_proj:0",),
                "gate_proj": ("gate_proj:0",),
                "v_router": ("v_router:!",),
                "v_experts": {
                    # ('#',) denotes the ModuleList element itself, whose
                    # runtime module is a Linear rather than an MLP wrapper.
                    "#": ("#",),
                },
                "o_proj": ("o_proj:1",),
            },
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Dense fallback used by K2-Horizon's first layers and the
                # standalone 0.9B dense checkpoint.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


# A descriptive alias is useful to callers that use GPTQ-oriented names.
K2HorizonGPTQ = K2HorizonQModel

__all__ = ["K2HorizonGPTQ", "K2HorizonMoELifecycleHooks", "K2HorizonQModel"]
