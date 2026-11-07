# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..fusable import FusableConfig, LlamaFusableConfig
from .base_qwen3_vl import BaseQwen3VLGPTQ


class GlmOCRGPTQ(BaseQwen3VLGPTQ):
    # GLM-OCR computes multimodal RoPE from `mm_token_type_ids`, so the
    # processor-produced field must reach the model on the calibration path.
    fusable = FusableConfig("glm_ocr").merge(LlamaFusableConfig(), profiles=("qkv",))

    require_mm_token_type_ids = True

    require_load_processor = True

    pre_lm_head_norm_module = "model.language_model.norm"
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


__all__ = ["GlmOCRGPTQ"]
