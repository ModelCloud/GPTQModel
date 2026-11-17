# Copyright 2026 Alibaba Group Holding Limited
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned released-root Qwen-Drive VLM (not the separate planner).

The upstream wrapper instantiates exactly Qwen3.5 under ``vlm`` and delegates
VQA to it. No planner weights are in the released root safetensors.
"""

from transformers import GenerationMixin, PreTrainedModel, Qwen3_5ForConditionalGeneration

from .configuration_qwen_drive import QwenDriveConfig


class QwenDriveForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = QwenDriveConfig
    base_model_prefix = "vlm"
    _supports_sdpa = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]
    _tied_weights_keys = {"vlm.lm_head.weight": "vlm.model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.vlm = Qwen3_5ForConditionalGeneration(config.vlm_config)
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return self.vlm(input_ids=input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs.setdefault("generation_config", self.generation_config)
        return self.vlm.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.vlm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.vlm.set_output_embeddings(new_embeddings)
