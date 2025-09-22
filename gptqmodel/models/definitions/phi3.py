# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..base import BaseQModel


class Phi3QModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("qkv_proj:0", "o_proj:1"),
            "mlp": ("gate_up_proj:0", "down_proj:1"),
        }
    ]

class PhiMoEGPTQForCausalLM(BaseQModel):
    require_pkgs_version = ["transformers<=4.44.2"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe": {
                "experts": {
                    "#": ("w1:0", "w2:1"),
                },
            },
        }
    ]

__all__ = ["Phi3QModel", "PhiMoEGPTQForCausalLM"]
