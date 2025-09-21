# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class InternLM2QModel(BaseQModel):

    require_pkgs_version = ["transformers<=4.44.2"]
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "attention_norm": ("attention_norm:!",),
            "attention": ("wqkv:0", "wo:0"),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward": ("w1:0", "w3:0", "w2:1"),
        }
    ]
