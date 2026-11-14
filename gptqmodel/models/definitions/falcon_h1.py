# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class FalconH1QModel(BaseQModel):
    fusable = FusableConfig("falcon_h1").merge(LlamaFusableConfig())

    layers_node = "model.layers"

    module_tree= [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Attention and the Mamba mixer are parallel branches of one norm.
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            # The SSM between in_proj and out_proj is not homogeneous.
            "mamba": ("in_proj:0:in=x:nosrc", "out_proj:1"),
            "pre_ff_layernorm": ("pre_ff_layernorm:!",),
            "feed_forward": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
