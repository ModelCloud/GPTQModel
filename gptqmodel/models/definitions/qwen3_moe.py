# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...quantization import METHOD
from ..base import BaseQModel


class Qwen3MoeQModel(BaseQModel):
    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0",  "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate": ("gate:!",), # <-- 0.5MB per layer. Not worth quantizing
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]

    module_tree_overrides = {
        METHOD.AWQ: [
            {
                "mlp": {
                    "gate": ("gate",),
                }
            }
        ]
    }
