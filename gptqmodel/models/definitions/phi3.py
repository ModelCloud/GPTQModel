# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..base import BaseQModel


class Phi3QModel(BaseQModel):
    shared_input_verified_model_types = frozenset({"phi3"})

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("qkv_proj:0:k:q:v", "o_proj:1"),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        }
    ]

class PhiMoEGPTQForCausalLM(BaseQModel):
    dynamic_expert_index = "num_local_experts"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp|block_sparse_moe:moe:?": {
                "router": ("router:!",),  # PhimoeTopKRouter.forward() returns two values, skipping its quantization.
                "experts:routed": {
                    "#": ("gate_proj|w1:0:gate", "up_proj|w3:0:up", "down_proj|w2:1:down"),
                },
            },
        }
    ]

__all__ = ["Phi3QModel", "PhiMoEGPTQForCausalLM"]
