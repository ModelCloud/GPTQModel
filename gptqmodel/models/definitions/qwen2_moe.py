# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Qwen2MoeQModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"qwen2_moe"})
    """Qwen2 MoE definition aligned to the upstream HF forward execution order."""

    shared_input_verified_model_types = frozenset({"qwen2_moe"})

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # The module tree is the forward-order ground truth for subset replay and
    # early-stop. Qwen2 executes the shared expert path before routed experts
    # in upstream HF modeling code, and Defuser should preserve that same
    # execution order when it unfuses the experts.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0:shared": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
                "experts:0:routed:expert_activation=expert.act_fn": {
                    "#": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
                },
            },
        }
    ]
