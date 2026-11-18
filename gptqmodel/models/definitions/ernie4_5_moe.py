# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class Ernie4_5_MoeQModel(BaseQModel):
    fusable = FusableConfig("ernie4_5_moe").merge(LlamaFusableConfig(), profiles=("qkv",))

    support_batch_quantize = False
    layer_modules_strict = False

    dynamic_expert_index = "moe_num_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate_proj": ("gate_proj",),
                "up_proj": ("up_proj",),
                "down_proj": ("down_proj",),
                "gate": ("gate:!", "moe_statics:!"),
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]

