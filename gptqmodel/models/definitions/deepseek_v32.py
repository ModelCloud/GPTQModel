# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class DeepSeekV32QModel(BaseQModel):
    """DeepSeek-V3.2 MLA/DSA model with dense and routed/shared-MoE layers."""

    require_trust_remote_code = False

    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"

    # The first three decoder layers are dense and the remaining layers are MoE.
    layer_modules_strict = False

    # Transformers intentionally ignores the checkpoint's auxiliary MTP layer.
    # Preserve those tensors when saving a quantized checkpoint.
    out_of_model_tensors = {"prefixes": ["model.layers.61"]}

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_proj:0:q",
                "kv_a_proj_with_mqa:0:k:v",
                "indexer.wk:0",
                # DSA accumulates this projection's scores in float32; leave its weights unquantized.
                "indexer.weights_proj:0:!",
                "q_b_proj:1:q",
                "kv_b_proj:1:k:v",
                "indexer.wq_b:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Dense fallback used by the first three decoder layers.
                "": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                "gate": ("gate:!", "e_score_correction_bias:!"),
                "experts:routed:expert_activation=experts.act_fn": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts:shared": (
                    "gate_proj:0:gate",
                    "up_proj:0:up",
                    "down_proj:1:down",
                ),
            },
        },
    ]


__all__ = ["DeepSeekV32QModel"]
