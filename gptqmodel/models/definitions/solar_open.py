# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class SolarOpenQModel(BaseQModel):
    """GPTQ definition for Solar Open's GQA MoE decoder."""

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # Solar Open uses GQA, so o_proj does not match v_proj's shape.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Router weights and the score-correction bias remain dense.
                "gate": ("gate:!",),
                # Match SolarOpenMoE.forward(): routed experts execute before
                # the shared expert contribution is added.
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]


__all__ = ["SolarOpenQModel"]
