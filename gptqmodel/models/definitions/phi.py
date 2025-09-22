# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class PhiQModel(BaseQModel):
    pre_lm_head_norm_module = "model.final_layernorm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:1", "v_proj:2", "dense:3"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("fc1:0", "fc2:1"),
        }
    ]
