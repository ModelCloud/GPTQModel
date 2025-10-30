# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class KimiLinearQModel(BaseQModel):
    require_trust_remote_code = True

    layer_modules_strict = False

    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "q_conv1d:0:!",
                "kv_a_proj_with_mqa:0",
                "k_proj:0",
                "k_conv1d:0:!",
                "v_proj:0",
                "v_conv1d:0:!",
                "kv_b_proj:1",
                "f_a_proj:1:!",
                "f_b_proj:1:!",
                "b_proj:1:!",
                "g_a_proj:1:!",
                "g_b_proj:1:!",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "block_sparse_moe": {
                "experts": {
                    "#": ("w1:0", "w3:0", "w2:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

__all__ = ["KimiLinearQModel"]
