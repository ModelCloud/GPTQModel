# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import LlamaFusableConfig


class FalconH1QModel(BaseQModel):
    fusable = LlamaFusableConfig().derive("falcon_h1", parent_types={
        "qkv": (
            "transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Attention",
        ),
        "gate_up": (
            "transformers.models.falcon_h1.modeling_falcon_h1.FalconH1MLP",
        ),
    })

    layers_node = "model.layers"

    module_tree= [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mamba": ("in_proj:0", "out_proj:1"),
            "feed_forward": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
