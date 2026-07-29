# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class KimiK3QModel(BaseQModel):
    """Kimi-K3 text+vision model. Quantize the language_model decoder only."""

    require_trust_remote_code = True
    require_load_processor = False

    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"

    layer_modules_strict = False
    dynamic_expert_index = "num_experts"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "f_a_proj:0",
                "b_proj:0",
                "g_proj:0",
                "q_b_proj:1",
                "kv_b_proj:1",
                "f_b_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "block_sparse_moe": {
                "routed_expert_down_proj": ("routed_expert_down_proj:0",),
                "experts": {
                    "#": ("w1:0", "w2:1", "w3:0"),
                },
                "routed_expert_up_proj": ("routed_expert_up_proj:1",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
            "self_attention_res_proj": ("self_attention_res_proj:0",),
            "mlp_res_proj": ("mlp_res_proj:0",),
        },
    ]
