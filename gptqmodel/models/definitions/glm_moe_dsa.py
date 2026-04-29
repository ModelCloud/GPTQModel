# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class GlmMoeDsaQModel(BaseQModel):
    # GLM-5 and GLM-5.1 currently share the same modeling config and both resolve
    # to transformers model_type `glm_moe_dsa`.
    # The first three decoder blocks are dense MLPs, with later blocks switching
    # to routed experts plus a shared-expert branch.
    layer_modules_strict = False

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # GLM-5 / GLM-5.1 use MLA attention plus a DSA indexer. `q_proj`
                # is an optional fallback path; current public configs use q_a/q_b.
                "q_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "indexer.wk:0",
                "q_b_proj:1",
                "kv_b_proj:1",
                "indexer.wq_b:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Dense fallback for the first `mlp_layer_types == "dense"` blocks.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

__all__ = ["GlmMoeDsaQModel"]
