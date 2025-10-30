# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GraniteMoeHybridQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "mamba": ("in_proj:0", "out_proj:1"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "shared_mlp": ("input_linear:0", "output_linear:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
        }
    ]
