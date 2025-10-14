# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class LFM2MoeQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"
    layer_modules_strict = False

    module_tree= [
        "model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            "conv": ("in_proj", "out_proj"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward": {
                "gate": ("gate:!",),
                "": ("w1:0", "w3:0", "w2:1"),
                "experts": {
                    "#": ("w1:0", "w3:0", "w2:1"),
                },
            },
        }
    ]

