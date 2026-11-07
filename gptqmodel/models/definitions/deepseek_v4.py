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
            "attn_hc:!": ("input_norm:!",),
            "input_layernorm": ("input_layernorm:!",),
            # `sinks` is a direct parameter on the attention container.
            "self_attn:!": (
                "q_a_norm:!",
                "q_a_proj:0:q",
                "q_b_norm:!",
                "q_b_proj:0:q:in=q_a",
                "o_a_proj:!",
                "o_b_proj:1:o",
                "kv_norm:!",
                "kv_proj:2:k:v",
                "compressor:!",
                "compressor.kv_norm:!",
                "compressor.kv_proj:!",
                "compressor.gate_proj:!",
                "compressor.indexer:!",
                "compressor.indexer.kv_norm:!",
                "compressor.indexer.kv_proj:!",
                "compressor.indexer.q_b_proj:!",
                "compressor.indexer.gate_proj:!",
                "compressor.indexer.scorer.weights_proj:!",
            ),
            "ffn_hc:!": ("input_norm:!",),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!", "e_score_correction_bias:!", "tid2eid:!"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]



__all__ = ["DeepSeekV4QModel"]
