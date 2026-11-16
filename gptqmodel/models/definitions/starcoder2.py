# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class Starcoder2QModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"starcoder2"})

    fusable = FusableConfig("starcoder2").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # ``c_proj`` consumes the GELU of ``c_fc``, so folding a scale into
            # ``c_fc`` would not preserve the MLP output.
            "mlp": ("c_fc:0:nosrc", "c_proj:1"),
        }
    ]
