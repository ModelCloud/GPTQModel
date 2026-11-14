# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class LFM2MoeQModel(BaseQModel):
    fusable = FusableConfig("lfm2_moe").merge(LlamaFusableConfig(), profiles=("qkv",))

    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"
    layer_modules_strict = False

    # Transformers 5.22 splits the fused expert tensors into per-expert
    # gate_proj/up_proj/down_proj linears (the dense MLP still uses w1/w3/w2).
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree= [
        "model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            "conv": ("in_proj", "out_proj"),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "out_proj:1"),
            "ffn_norm": ("ffn_norm:!",),
            "feed_forward:moe": {
                "gate": ("gate:!",),
                "": ("w1:0", "w3:0", "w2:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]
