# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class CodeGenQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "attn": ("qkv_proj:0", "out_proj:1"),
            "mlp": ("fc_in:0", "fc_out:1"),
        }
    ]
