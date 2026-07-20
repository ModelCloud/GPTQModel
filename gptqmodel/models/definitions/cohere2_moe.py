# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class Cohere2MoeQModel(BaseQModel):
    # Cohere2-MoE uses dense prefix layers followed by routed MoE layers, both
    # under the same `mlp` attribute.
    layer_modules_strict = False
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # Cohere2-MoE uses a parallel residual block:
    #   norm -> attention
    #        -> dense MLP or routed MoE
    #   residual + attention + mlp
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                # Dense fallback used by Cohere2-MoE prefix decoder blocks.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


__all__ = ["Cohere2MoeQModel"]
