# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class Starcoder2QModel(BaseQModel):
    fusable = FusableConfig("starcoder2").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # ``c_proj`` consumes the GELU of ``c_fc``, so folding a scale into
            # ``c_fc`` would not preserve the MLP output.
            "mlp": ("c_fc:0:nosrc", "c_proj:1"),
        }
    ]
