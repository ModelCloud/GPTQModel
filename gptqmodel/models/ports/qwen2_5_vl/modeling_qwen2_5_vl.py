# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
"""Use the native decoder unchanged; supply only the missing text-task head."""

from torch import nn
from transformers import PreTrainedConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel, Qwen2_5_VLTextModel
from transformers.utils.generic import can_return_tuple


class Qwen2_5_VLForCausalLM(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    """Text extraction schema: ``model.<native decoder key>`` and ``lm_head``.

    This is an explicit text checkpoint, not automatic extraction from a VL
    checkpoint. Native mRoPE, attention, masks, cache and decoder layers are
    reused without parameter renaming or a second modeling implementation.
    """

    config_class = Qwen2_5_VLTextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    accepts_loss_kwargs = False
    input_modalities = ("text",)

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        media = {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
            "second_per_grid_ts",
            "mm_encoder_outputs",
        }
        if any(kwargs.get(key) is not None for key in media):
            raise ValueError("qwen2_5_vl_text accepts text only; use qwen2_5_vl for image/video inputs")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, selection, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2_5_VLConfigLoader:
    """Read only declarative metadata; never dispatch through checkpoint Python."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        payload, _ = PreTrainedConfig.get_config_dict(path, **kwargs)
        types = {"qwen2_5_vl": Qwen2_5_VLConfig, "qwen2_5_vl_text": Qwen2_5_VLTextConfig}
        model_type = payload.get("model_type")
        config_type = types.get(model_type) if isinstance(model_type, str) else None
        if config_type is None:
            raise ValueError(f"Unsupported Qwen2.5-VL model_type: {payload.get('model_type')!r}")
        return config_type.from_pretrained(path, **kwargs)


class Qwen2_5_VLModelLoader:
    """Keep the composite model exactly native, even with hostile auto_map."""

    @staticmethod
    def _model_class(config):
        if type(config) is Qwen2_5_VLConfig:
            return Qwen2_5_VLForConditionalGeneration
        if type(config) is Qwen2_5_VLTextConfig:
            return Qwen2_5_VLForCausalLM
        raise ValueError(f"Unsupported Qwen2.5-VL config: {type(config).__name__}")

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._model_class(config)._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, config=None, **kwargs):
        if config is None:
            config = Qwen2_5_VLConfigLoader.from_pretrained(path, **kwargs)
        return cls._model_class(config).from_pretrained(path, config=config, **kwargs)
