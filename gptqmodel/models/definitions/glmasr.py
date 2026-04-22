# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModel
from ..base import BaseQModel


class GlmASRGPTQ(BaseQModel):
    loader = AutoModel

    require_load_processor = True

    # GLM-ASR keeps the speech encoder and projector at the top level while the
    # text decoder stays under `language_model.model`.
    lm_head = "language_model.lm_head"
    pre_lm_head_norm_module = "language_model.model.norm"
    rotary_embedding = "language_model.model.rotary_emb"

    module_tree = [
        "language_model",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]

__all__ = ["GlmASRGPTQ"]
