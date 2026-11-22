# Copyright 2023 Haotian Liu
# SPDX-License-Identifier: Apache-2.0
# Transformers 5 adaptation by ModelCloud.ai, 2026; see NOTICE.

import re

import torch
from torch import nn
from transformers import initialization as init

from ..native.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from .configuration_llava_qwen2 import LlavaConfig
from .multimodal import LlavaMultimodalMixin
from .vision import build_vision_tower


def build_vision_projector(config):
    kind = config.mm_projector_type
    if kind == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if kind == "identity":
        if config.mm_hidden_size != config.hidden_size:
            raise ValueError("Identity projector requires equal vision/text dimensions")
        return nn.Identity()
    match = re.fullmatch(r"mlp([1-9]\d*)x_gelu", kind)
    if match is None:
        raise ValueError(f"Unknown projector type: {kind}")
    modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, int(match[1])):
        modules.extend((nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)))
    return nn.Sequential(*modules)


class LlavaQwen2Model(Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        if config.mm_vision_tower is not None:
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_vision_projector(config)
            if "unpad" in config.mm_patch_merge_type:
                self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * config.hidden_size ** -0.5)

    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)


class LlavaQwen2ForCausalLM(LlavaMultimodalMixin, Qwen2ForCausalLM):
    config_class = LlavaConfig
    _can_compile_fullgraph = False

    def __init__(self, config):
        # Keep the original flat model.layers and root lm_head ownership.
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LlavaQwen2Model) and hasattr(module, "image_newline"):
            init.normal_(module.image_newline, std=self.config.hidden_size ** -0.5)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, images=None, image_sizes=None, **kwargs):
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask,
                                                         past_key_values, labels, images, image_sizes))
        elif images is not None:
            raise ValueError("Pass images with input_ids, not with inputs_embeds")
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, **kwargs):
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("Use input_ids rather than inputs_embeds for LLaVA generation")
        if images is None:
            return super().generate(inputs=inputs, **kwargs)
        mask, positions = kwargs.pop("attention_mask", None), kwargs.pop("position_ids", None)
        _, positions, mask, _, embeddings, _ = self.prepare_inputs_labels_for_multimodal(
            inputs, positions, mask, None, None, images, image_sizes)
        # Upstream's image generate API returns generated IDs, not negative placeholder IDs.
        return super().generate(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions, **kwargs)
