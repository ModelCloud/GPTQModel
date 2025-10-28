# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class MiniMaxM2GPTQ(BaseQModel):
    """
    GPTQ config for MiniMax-M2 (HF: MiniMaxM2*), featuring:
      - Per-layer RMSNorm before/after attention
      - Standard attention with q/k normalization parameters
      - Sparse MoE block (MiniMaxM2SparseMoeBlock) with up to 256 experts
    """

    pre_lm_head_norm_module = "model.norm"

    layer_modules_strict = False

    dynamic_expert_index = "num_local_experts"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "q_norm:0:!",
                "k_proj:0",
                "k_norm:0:!",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "block_sparse_moe": {
                "gate": ("gate:!",),
                "e_score_correction_bias": ("e_score_correction_bias:!",),
                "experts": {
                    "#": ("w1:0", "w3:0", "w2:1"),
                },
            },
        },
    ]

