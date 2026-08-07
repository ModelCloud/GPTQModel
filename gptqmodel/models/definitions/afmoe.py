# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel


class AfMoeQModel(BaseQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    require_trust_remote_code = True
    layer_modules_strict = False

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    moe_lifecycle_hooks.expert_block_names = ['experts']
    moe_lifecycle_hooks.shared_expert_block_names = ['shared_expert']

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert:0:shared": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "experts:0:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
            },
        }
    ]
