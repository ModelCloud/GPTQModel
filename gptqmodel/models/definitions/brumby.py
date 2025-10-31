# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class BrumbyQModel(BaseQModel):
    require_trust_remote_code = True
    require_pkgs_version = ["retention>=1.0.7"]

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:0:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "g_proj:0:!",
                "o_proj:1",
                "q_norm:0:!",
                "k_norm:0:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:1:!",),
            "mlp": (
                "gate_proj:0",
                "up_proj:0",
                "down_proj:1",
            ),
        },
    ]
