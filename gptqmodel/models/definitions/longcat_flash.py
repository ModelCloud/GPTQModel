# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class LongCatFlashQModel(BaseQModel):
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": {
                "0": ("q_a_proj:0", "q_b_proj:0", "kv_a_proj_with_mqa:0", "kv_b_proj:0", "o_proj:1"),
                "1": ("q_a_proj:0", "q_b_proj:0", "kv_a_proj_with_mqa:0", "kv_b_proj:0", "o_proj:1")
            },
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlps": {
                "0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "1": ("gate_proj:0", "up_proj:0", "down_proj:1")
            },
            "mlp": {
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1")
                }
            }
        }
    ]
