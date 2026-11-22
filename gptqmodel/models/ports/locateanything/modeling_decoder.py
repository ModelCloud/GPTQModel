# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""LocateAnything decoders on the vendored native Qwen2/Qwen3 families (SPEC section 6)."""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel
from ..native.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from .mask_utils import (
    apply_cached_window,
    apply_uncached_windows,
    causal_visibility,
    to_additive_mask,
    training_visibility,
)


DEFAULT_BLOCK_SIZE = 6
DEFAULT_TEXT_MASK_TOKEN_ID = 151676


def inject_visual_features(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor | None,
    visual_features: torch.Tensor | None,
    image_token_index: int | None,
) -> torch.Tensor:
    """Write visual feature rows over the image placeholder embeddings (SPEC 6.1)."""
    if visual_features is None:
        return inputs_embeds
    if input_ids is None:
        raise ValueError("input_ids are required to place visual features")
    features = visual_features.reshape(-1, inputs_embeds.shape[-1])
    placeholders = input_ids == image_token_index
    if int(placeholders.sum()) != features.shape[0]:
        raise ValueError(
            "Image placeholder count does not match visual features: "
            f"{int(placeholders.sum())} placeholders, {features.shape[0]} feature rows"
        )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[placeholders] = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds


def _embed(model: nn.Module, input_ids, inputs_embeds, visual_features, image_token_index):
    if input_ids is None and inputs_embeds is None:
        raise ValueError("You must specify input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)
    return inject_visual_features(inputs_embeds, input_ids, visual_features, image_token_index)


class LocateAnythingGenerationMixin:
    """Generation input preparation shared by both decoder variants (SPEC 6.5)."""

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        del inputs_embeds
        if past_key_values is not None:
            cached = past_key_values.get_seq_length()
            if input_ids.shape[1] > cached:
                input_ids = input_ids[:, cached:]
        position_ids = kwargs.get("position_ids")
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }


class LocateAnythingQwen2Model(Qwen2Model):
    """Native Qwen2 layers under the block-diffusion visibility rules (SPEC 6.2 - 6.4)."""

    def _sliding_window(self) -> int | None:
        layer_types = list(self.config.layer_types or [])
        if "sliding_attention" not in layer_types:
            return None
        if any(kind != "sliding_attention" for kind in layer_types):
            raise NotImplementedError("LocateAnything does not support mixed full and sliding attention layers")
        return self.config.sliding_window

    def build_attention_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_len: int,
        use_cache: bool,
    ) -> torch.Tensor:
        dtype = inputs_embeds.dtype
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask.to(dtype)
        batch_size, query_len = inputs_embeds.shape[:2]
        block_size = getattr(self.config, "block_size", DEFAULT_BLOCK_SIZE)
        causal_attn = bool(getattr(self.config, "causal_attn", False))
        if self.training:
            if past_len != 0:
                raise ValueError("The training block mask requires an empty cache")
            visible = training_visibility(position_ids.expand(batch_size, -1), block_size, causal_attn)
            return to_additive_mask(visible, dtype, build_dtype=torch.bfloat16)

        visible = causal_visibility(
            batch_size,
            query_len,
            past_len,
            inputs_embeds.device,
            sliding_window=self._sliding_window(),
            attention_mask=attention_mask,
        )
        mask_token_id = getattr(self.config, "text_mask_token_id", DEFAULT_TEXT_MASK_TOKEN_ID)
        if query_len > 1 and input_ids is not None and int(input_ids[0, -1]) == mask_token_id:
            if use_cache:
                visible = apply_cached_window(visible, block_size, causal_attn)
            else:
                visible = apply_uncached_windows(visible, input_ids, mask_token_id, causal_attn)
        return to_additive_mask(visible, dtype)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        visual_features: torch.Tensor | None = None,
        image_token_index: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("LocateAnything supports past_key_values as a DynamicCache only")
        inputs_embeds = _embed(self, input_ids, inputs_embeds, visual_features, image_token_index)

        if use_cache and past_key_values is None:
            # No config: the cache must keep the whole prefix (no sliding eviction).
            past_key_values = DynamicCache()
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_len
            position_ids = position_ids.unsqueeze(0)

        mask = self.build_attention_mask(
            input_ids, attention_mask, position_ids, inputs_embeds, past_len, bool(use_cache)
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class LocateAnythingQwen2ForCausalLM(LocateAnythingGenerationMixin, Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = LocateAnythingQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        visual_features: torch.Tensor | None = None,
        image_token_index: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            visual_features=visual_features,
            image_token_index=image_token_index,
            **kwargs,
        )
        outputs.logits = outputs.logits.float()
        return outputs


class LocateAnythingQwen3ForCausalLM(LocateAnythingGenerationMixin, Qwen3ForCausalLM):
    """Native Qwen3 causal LM; only image injection is added (SPEC 6.6)."""

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        visual_features: torch.Tensor | None = None,
        image_token_index: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if visual_features is not None:
            inputs_embeds = _embed(self.model, input_ids, inputs_embeds, visual_features, image_token_index)
            input_ids = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = [
    "LocateAnythingQwen2ForCausalLM",
    "LocateAnythingQwen2Model",
    "LocateAnythingQwen3ForCausalLM",
    "inject_visual_features",
]

