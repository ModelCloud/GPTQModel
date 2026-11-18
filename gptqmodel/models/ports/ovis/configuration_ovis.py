# Copyright 2024 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
"""Ovis 1.0/1.5/1.6 configuration; see NOTICE for pinned sources."""

import copy

from transformers import CLIPVisionConfig, Gemma2Config, LlamaConfig, PretrainedConfig, Qwen2Config, SiglipVisionConfig


IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]
IMAGE_TOKEN = "<image>"
TEXT_CONFIGS = {"llama": LlamaConfig, "gemma2": Gemma2Config, "qwen2": Qwen2Config}


def native_config(value, classes, default):
    if value is None:
        return default()
    payload = copy.deepcopy(value.to_dict() if isinstance(value, PretrainedConfig) else value)
    model_type = payload.pop("model_type", None)
    if model_type not in classes:
        raise ValueError(f"Unqualified Ovis nested model_type: {model_type!r}")
    return classes[model_type](**payload)


class BaseVisualTokenizerConfig(PretrainedConfig):
    sub_configs = {"backbone_config": SiglipVisionConfig}

    def __init__(self, vocab_size=16384, tokenize_function="softmax", tau=1.0, depths=None,
                 drop_cls_token=False, backbone_config=None, hidden_stride=1, use_indicators=False,
                 hd_booster=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        self.depths = [int(x) for x in depths.split("|")] if isinstance(depths, str) else depths
        self.hidden_stride = hidden_stride
        self.use_indicators = use_indicators
        self.hd_booster = hd_booster
        self.drop_cls_token = drop_cls_token if self.model_type == "clip_visual_tokenizer" else False
        cls = CLIPVisionConfig if self.model_type == "clip_visual_tokenizer" else SiglipVisionConfig
        self.backbone_config = native_config(backbone_config, {cls.model_type: cls}, cls)
        if self.depths:
            if len(self.depths) != 1:
                raise ValueError("Ovis visual depths must contain exactly one depth")
            self.backbone_config.num_hidden_layers = self.depths[0]
        if hidden_stride < 1 or vocab_size <= 5 or tau <= 0:
            raise ValueError("Invalid Ovis visual stride, vocabulary or temperature")
        if hd_booster not in (None, "s2wrapper", "s2wrapper-adaptive"):
            raise ValueError(f"Unqualified Ovis hd_booster: {hd_booster!r}")
        if tokenize_function not in ("softmax", "gumbel_argmax", "st_argmax"):
            raise ValueError(f"Unknown Ovis tokenize_function: {tokenize_function!r}")


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "siglip_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ClipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "clip_visual_tokenizer"
    sub_configs = {"backbone_config": CLIPVisionConfig}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


VISUAL_CONFIGS = {"siglip_visual_tokenizer": SiglipVisualTokenizerConfig,
                  "clip_visual_tokenizer": ClipVisualTokenizerConfig}


class OvisConfig(PretrainedConfig):
    model_type = "ovis"
    sub_configs = {"llm_config": PretrainedConfig, "visual_tokenizer_config": BaseVisualTokenizerConfig}

    def __init__(self, llm_config=None, visual_tokenizer_config=None, multimodal_max_length=8192,
                 hidden_size=None, conversation_formatter_class=None, llm_attn_implementation=None,
                 disable_tie_weight=False, ovis_version=None, **kwargs):
        raw_visual = visual_tokenizer_config.to_dict() if isinstance(visual_tokenizer_config, PretrainedConfig) else visual_tokenizer_config
        if ovis_version is None:
            if raw_visual and "hd_booster" in raw_visual:
                ovis_version = "1.5"
            elif raw_visual and raw_visual.get("model_type") == "clip_visual_tokenizer":
                ovis_version = "1.0"
            else:
                ovis_version = "1.6"
        if ovis_version not in ("1.0", "1.5", "1.6"):
            raise ValueError(f"Unqualified Ovis version: {ovis_version!r}")
        self.llm_config = native_config(llm_config, TEXT_CONFIGS, LlamaConfig)
        self.visual_tokenizer_config = native_config(visual_tokenizer_config, VISUAL_CONFIGS, SiglipVisualTokenizerConfig)
        self.ovis_version = ovis_version
        visual = self.visual_tokenizer_config
        if ovis_version == "1.6" and (visual.model_type != "siglip_visual_tokenizer" or visual.hd_booster is not None):
            raise ValueError("Ovis1.6 requires its partitioned SigLIP visual tokenizer")
        if ovis_version == "1.0" and (visual.model_type != "clip_visual_tokenizer" or visual.hd_booster is not None):
            raise ValueError("Ovis1.0 requires its CLIP visual tokenizer")
        if visual.model_type == "clip_visual_tokenizer" and visual.hidden_stride != 1:
            raise ValueError("Source CLIP Ovis does not support hidden_stride != 1")
        self.hidden_size = hidden_size if hidden_size is not None else self.llm_config.hidden_size
        if self.hidden_size != self.llm_config.hidden_size or multimodal_max_length < 1:
            raise ValueError("Invalid Ovis text hidden size or multimodal length")
        self.multimodal_max_length = multimodal_max_length
        self.conversation_formatter_class = conversation_formatter_class or {
            "llama": "Llama3ConversationFormatter", "qwen2": "QwenConversationFormatter",
            "gemma2": "GemmaConversationFormatter"}[self.llm_config.model_type]
        if self.conversation_formatter_class not in ("Llama3ConversationFormatter", "QwenConversationFormatter", "GemmaConversationFormatter"):
            raise ValueError("Unknown package-owned Ovis conversation formatter")
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
        kwargs.setdefault("tie_word_embeddings", self.llm_config.tie_word_embeddings and not disable_tie_weight)
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False, encoder=False):
        return self.llm_config
