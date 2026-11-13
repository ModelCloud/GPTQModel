# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class DbrxConvertedQModel(BaseQModel):
    # dbrx_converted requires custom model code
    require_trust_remote_code = True

    # DBRX stores its expert cardinality under the nested ``ffn_config``; the
    # base ``_expert_count_config`` resolver follows that nesting.
    dynamic_expert_index = "moe_num_experts"

    pre_lm_head_norm_module = "transformer.norm_f"

    module_tree= [
        "transformer",
        "blocks",
        "#",
        {
            "norm_attn_norm": {
                "norm_1": ("norm_1:!",),
                "attn": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
                "norm_2": ("norm_2:!",),
            },
            "ffn:moe": {
                "router": {
                    "layer": ("layer:!",),
                },
                "experts": {
                    "mlp": {
                        "#": ("w1:0", "v1:0", "w2:1"),
                    },
                },
            },
        }
    ]
