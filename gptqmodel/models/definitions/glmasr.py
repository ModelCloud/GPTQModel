# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForMultimodalLM

from ..base import BaseQModel


class GlmASRGPTQ(BaseQModel):
    # `AutoModel` builds `GlmAsrModel`, which has no `lm_head` and nests the
    # text decoder directly under `language_model`. The checkpoint is a
    # `GlmAsrForConditionalGeneration`, so load the complete wrapper and
    # quantize the decoder through `model.language_model`.
    loader = AutoModelForMultimodalLM

    require_load_processor = True

    # GLM-ASR keeps the speech encoder and projector at the top level while the
    # text decoder stays under `model.language_model`.
    lm_head = "lm_head"
    pre_lm_head_norm_module = ["model.language_model.norm", "language_model.norm"]
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
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
