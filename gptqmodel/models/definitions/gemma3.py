# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

from ..base import BaseQModel
from . import LlamaQModel


class Gemma3QModel(LlamaQModel):
    pass


class Gemma3ForConditionalGenerationGPTQ(BaseQModel):
    support_batch_quantize = False
    pre_lm_head_norm_module = "model.language_model.norm"
    HF_CONVERSION_MAP_REVERSED = (
        # Gemma 3 shells expose `model.vision_tower.*`, but checkpoint tensors
        # live under `vision_tower.vision_model.*`, so restore that missing hop first.
        SimpleNamespace(
            source_patterns=[r"^model\.vision_tower\.(?!vision_model\.)(.+)$"],
            target_patterns=[r"^vision_tower.vision_model.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.language_model"],
            target_patterns=[r"^language_model.model"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"lm_head"],
            target_patterns=[r"^language_model.lm_head"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.vision_tower"],
            target_patterns=[r"^vision_tower"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"model\.multi_modal_projector"],
            target_patterns=[r"^multi_modal_projector"],
            operations=[],
        ),
    )

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
        }
    ]

    lm_head_module = "model.lm_head"
