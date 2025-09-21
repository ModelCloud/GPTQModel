# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GptNeoQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.ln_f"
    lm_head = "lm_head"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "attn": {
                "attention": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1")
            },
            "ln_2": ("ln_2:!",),
            "mlp": ("c_fc:0", "c_proj:1"),
        }
    ]
