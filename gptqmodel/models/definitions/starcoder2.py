# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import LlamaFusableConfig


class Starcoder2QModel(BaseQModel):
    fusable = LlamaFusableConfig().derive("starcoder2", parent_types={
        "qkv": (
            "transformers.models.starcoder2.modeling_starcoder2.Starcoder2Attention",
        ),
    })

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("c_fc:0", "c_proj:1"),
        }
    ]
