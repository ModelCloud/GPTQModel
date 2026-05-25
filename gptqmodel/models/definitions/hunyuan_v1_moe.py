# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel


class HunYuanMoEV1QModel(BaseQModel):
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    # Hunyuan MoE uses GQA, so AWQ should not force o_proj scaling shape to
    # match v_proj.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()
    moe_lifecycle_hooks.shared_expert_block_names = ["shared_mlp"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "query_layernorm:!",
                "key_layernorm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                # Router weights are tiny and are not useful weight-only targets.
                "gate": ("gate:!",),
                # The original forward runs shared_mlp before routed experts.
                "shared_mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]
