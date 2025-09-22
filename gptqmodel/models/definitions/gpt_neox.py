# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class GPTNeoXQModel(BaseQModel):
    pre_lm_head_norm_module = "gpt_neox.final_layer_norm"
    lm_head = "embed_out"

    module_tree= [
        "gpt_neox",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("dense_h_to_4h:0", "dense_4h_to_h:1"),
        }
    ]
