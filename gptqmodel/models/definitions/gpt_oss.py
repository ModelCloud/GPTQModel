# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..base import BaseQModel


class GPTOSSGPTQ(BaseQModel):
    shared_input_verified_model_types = frozenset({"gpt_oss"})
    dynamic_expert_index = "num_local_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "experts:routed": {
                    "#": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
                },
            }
        }
    ]
