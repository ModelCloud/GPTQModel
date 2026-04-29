# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class LagunaQModel(BaseQModel):
    # Laguna has a dense first MLP layer and sparse MoE layers after that.
    layer_modules_strict = False
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # Defused Laguna experts follow the standard gate/up/down projection layout.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "g_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # Dense fallback used by Laguna's first decoder block.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


__all__ = ["LagunaQModel"]
