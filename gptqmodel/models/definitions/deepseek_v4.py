# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .deepseek_v3 import DeepSeekV3QModel


class DeepSeekV4QModel(DeepSeekV3QModel):
    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_norm:!",
                "q_a_proj:0:q",
                "q_b_norm:!",
                "q_b_proj:0:q",
                "o_a_proj:!",
                "o_b_proj:1",
                "kv_norm:!",
                "kv_proj:2:k:v",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                # DeepSeek V4's exact expert gate path includes the model's
                # configured activation and swiglu_limit clamps. Lifecycle
                # replay must call this declared method instead of rebuilding
                # the intermediate from an assumed activation.
                "experts:routed:expert_gate=experts._apply_gate": {
                    "#": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
                },
                "shared_experts:shared": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
            },
        },
    ]



__all__ = ["DeepSeekV4QModel"]
