# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class DbrxConvertedQModel(BaseQModel):
    # dbrx_converted requires custom model code
    require_trust_remote_code = True

    pre_lm_head_norm_module = "transformer.norm_f"

    module_tree= [
        "transformer",
        "blocks",
        "#",
        {
            "norm_attn_norm": {
                "attn": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
            },
            "ffn": {
                "experts": {
                    "mlp": {
                        "#": ("w1:0", "v1:0", "w2:1"),
                    },
                },
            },
        }
    ]
