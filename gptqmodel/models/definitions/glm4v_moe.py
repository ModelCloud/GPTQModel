# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .glm4v import Glm4vGPTQ
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Glm4vMoeQModel(Glm4vGPTQ):
    dynamic_expert_index = "n_routed_experts"

    layer_modules_strict = False

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                    "down_proj": ("down_proj:1",),
                },
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


__all__ = ["Glm4vMoeQModel"]
