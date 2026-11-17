# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.model import MODALITY
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..ports.llava_qwen2.multimodal import IMAGE_TOKEN_INDEX


class LlavaQwen2QModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.llava_qwen2.configuration_llava_qwen2.LlavaConfig",
        model_class="gptqmodel.models.ports.llava_qwen2.modeling_llava_qwen2.LlavaQwen2ForCausalLM",
        processor_class="gptqmodel.models.ports.llava_qwen2.processing_llava_qwen2.LlavaQwen2Processor",
    )
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    pre_lm_head_norm_module = "model.norm"
    rotary_embedding = "model.rotary_emb"
    modules_with_direct_meta_tensors = ["model"]

    def _sanitize_input_ids_for_embeddings(self, input_ids):
        image_mask = input_ids == IMAGE_TOKEN_INDEX
        sanitized = super()._sanitize_input_ids_for_embeddings(input_ids.masked_fill(image_mask, 0))
        return torch.where(image_mask, IMAGE_TOKEN_INDEX, sanitized)

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # The multimodal replay/folding objective is not qualified.
        return []

    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        from ...utils.calibration import batched_conversations
        from ._vlm_calibration import (
            finalize_conversation_batches,
            pop_conversation_calibration_options,
            split_calibration_dataset,
        )
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(pretokenized, batch_size=batch_size, **kwargs)
        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        batches = [
            self.processor.apply_chat_template(
                batch, add_generation_prompt=True, padding=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            for batch in batched_conversations(conversations, batch_size)
        ]
        return finalize_conversation_batches(batches, sort=sort, min_length=min_length, label=label)

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]
