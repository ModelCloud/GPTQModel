# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class MptQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.norm_f"

    module_tree = [
        "transformer",
        "blocks",
        "#",
        {
            "attn": ("Wqkv:0", "out_proj:1"),
            "ffn": ("up_proj:0", "down_proj:1"),
        }
    ]
