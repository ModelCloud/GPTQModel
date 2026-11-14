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
            # MPT normalizes before attention and before the FFN; the tree used
            # to omit both norms, so AWQ could not build a single group. The FFN
            # output projection follows the GELU.
            "norm_1": ("norm_1:!",),
            "attn": ("Wqkv:0", "out_proj:1"),
            "norm_2": ("norm_2:!",),
            "ffn": ("up_proj:0:nosrc", "down_proj:1"),
        }
    ]
