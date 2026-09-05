# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class LongCatFlashQModel(BaseQModel):
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": {
                "0": ("q_a_proj:0:q", "q_b_proj:0:q:in=q_a", "kv_a_proj_with_mqa:0:k:v", "kv_b_proj:0:k:v:in=kv_a", "o_proj:1"),
                "1": ("q_a_proj:0:q", "q_b_proj:0:q:in=q_a", "kv_a_proj_with_mqa:0:k:v", "kv_b_proj:0:k:v:in=kv_a", "o_proj:1")
            },
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlps": {
                "0": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "1": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down")
            },
            "mlp:moe": {
                "experts:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down")
                }
            }
        }
    ]
