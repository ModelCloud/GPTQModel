# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel


class Dots1QModel(BaseQModel):
    # allow missing expert modules on dense layers and support MoE expansion via config
    layer_modules_strict = False

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "k_norm:!", "q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "experts:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts:shared": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            },
        }
    ]
