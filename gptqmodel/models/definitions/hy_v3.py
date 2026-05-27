# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class HYV3QModel(BaseQModel):
    # HYV3 uses a dense first MLP layer and sparse MoE layers after it.
    layer_modules_strict = False
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

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
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


__all__ = ["HYV3QModel"]
