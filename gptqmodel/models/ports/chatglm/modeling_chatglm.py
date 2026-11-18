# SPDX-FileCopyrightText: 2023 THUDM
# SPDX-FileCopyrightText: 2024 Zhipu AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Adapted from THUDM ChatGLM sources.
"""Package-owned ChatGLM checkpoint shells and Transformers 5 cache boundary.

Decoder math and tensor names come from the reviewed upstream cores. Only this
boundary translates legacy cache axes to Transformers' batch/head/sequence axes.
"""

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from . import core, core_glm4
from .configuration_chatglm import ChatGLMConfig


class ChatGLMPreTrainedModel(PreTrainedModel):
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=getattr(self.config, "initializer_range", 0.02))
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (core.RMSNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, core.RotaryEmbedding):
            module.inv_freq.copy_(1.0 / (
                10000 ** (torch.arange(0, module.dim, 2, device=module.inv_freq.device).float() / module.dim)
            ))


class Embedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size, config.hidden_size, device=device, dtype=config.dtype
        )
        self.batch_first = config.rope_ratio is not None
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        hidden = self.word_embeddings(input_ids)
        if not self.batch_first:
            hidden = hidden.transpose(0, 1).contiguous()
        return hidden.float() if self.fp32_residual_connection else hidden


class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config, device=None, empty_init=False):
        super().__init__(config)
        # HF's no-init/meta context owns empty loading; skip_init would leave
        # from_config() uninitialized and bypass current HF initialization.
        del empty_init
        self.batch_first = config.rope_ratio is not None
        implementation = core_glm4 if self.batch_first else core
        self.embedding = Embedding(config, device=device)
        self.num_layers = config.num_layers
        self.seq_length = config.seq_length
        rope_kwargs = {"rope_ratio": config.rope_ratio} if self.batch_first else {}
        self.rotary_pos_emb = implementation.RotaryEmbedding(
            config.kv_channels // 2, original_impl=config.original_rope,
            device=device, dtype=config.dtype, **rope_kwargs,
        )
        self.encoder = implementation.GLMTransformer(config, device=device)
        self.output_layer = nn.Linear(
            config.hidden_size, config.padded_vocab_size, bias=False, device=device, dtype=config.dtype
        )
        self.pre_seq_len = config.pre_seq_len
        if self.pre_seq_len is not None:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.prefix_encoder = core.PrefixEncoder(config).to(device=device, dtype=config.dtype)
            self.dropout = nn.Dropout(0.1)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def get_prompt(self, batch_size, device, dtype):
        tokens = torch.arange(self.pre_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        prefix = self.dropout(self.prefix_encoder(tokens).to(dtype)).view(
            batch_size, self.pre_seq_len, self.num_layers * 2,
            self.config.num_key_value_heads, self.config.kv_channels,
        ).permute(2, 1, 0, 3, 4)
        pairs = tuple((pair[0], pair[1]) for pair in prefix.split(2))
        if self.batch_first:
            pairs = tuple(tuple(t.permute(1, 2, 0, 3) for t in pair) for pair in pairs)
        return pairs

    def _legacy_cache(self, past_key_values):
        if past_key_values is None:
            return None
        if isinstance(past_key_values, DynamicCache):
            if not past_key_values.get_seq_length():
                return None
            pairs = tuple((layer.keys, layer.values) for layer in past_key_values.layers)
            if not self.batch_first:
                pairs = tuple(tuple(t.permute(2, 0, 1, 3) for t in pair) for pair in pairs)
            return pairs
        if isinstance(past_key_values, tuple):
            return past_key_values
        raise TypeError("ChatGLM supports DynamicCache or its legacy forward-only tuple cache")

    def forward(
        self, input_ids=None, position_ids=None, attention_mask=None,
        full_attention_mask=None, past_key_values=None, inputs_embeds=None,
        use_cache=None, output_hidden_states=None, return_dict=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        else:
            # Public Transformers embeddings are always batch first.
            if not self.batch_first:
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            if self.config.fp32_residual_connection:
                inputs_embeds = inputs_embeds.float()
        batch_size = inputs_embeds.shape[0 if self.batch_first else 1]
        seq_length = inputs_embeds.shape[1 if self.batch_first else 0]
        device = inputs_embeds.device
        legacy = self._legacy_cache(past_key_values)
        if legacy is not None and len(legacy) != self.num_layers:
            raise ValueError("ChatGLM cache must contain every decoder layer")
        past_length = legacy[0][0].shape[2 if self.batch_first else 0] if legacy else 0
        if self.pre_seq_len is not None:
            if legacy is None:
                legacy = self.get_prompt(batch_size, device, inputs_embeds.dtype)
                past_length = self.pre_seq_len
            if attention_mask is not None:
                attention_mask = torch.cat((
                    attention_mask.new_ones(batch_size, self.pre_seq_len), attention_mask
                ), dim=-1)
        token_offset = past_length - (self.pre_seq_len or 0)
        if position_ids is None:
            position_ids = torch.arange(token_offset, token_offset + seq_length, device=device)[None]
        if position_ids.ndim != 2 or position_ids.shape[-1] != seq_length:
            raise ValueError("ChatGLM position_ids must be [batch, current_sequence]")
        if token_offset + seq_length > self.seq_length:
            raise ValueError("ChatGLM input exceeds seq_length")
        if full_attention_mask is None:
            # Bool True means masked, matching the original ChatGLM core.
            query_positions = torch.arange(seq_length, device=device)[:, None] + past_length
            key_positions = torch.arange(past_length + seq_length, device=device)[None, :]
            keep = (key_positions <= query_positions).expand(batch_size, -1, -1)
            if attention_mask is not None:
                if attention_mask.shape != (batch_size, past_length + seq_length):
                    raise ValueError("attention_mask must cover cached and current ChatGLM tokens")
                keep = keep & attention_mask[:, None, :].bool()
                # Preserve upstream's all-visible padded query rows on prefill.
                if not past_length:
                    keep = keep | ~attention_mask[:, :, None].bool()
            full_attention_mask = ~keep[:, None]
        rope = self.rotary_pos_emb(self.seq_length)[position_ids]
        if not self.batch_first:
            rope = rope.transpose(0, 1).contiguous()
        hidden, presents, states, attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rope,
            kv_caches=legacy, use_cache=use_cache, output_hidden_states=output_hidden_states,
        )
        cache = None
        if presents is not None:
            # Build after all layers succeed: a failed continuation leaves the
            # caller's cache usable for a retry rather than partly advanced.
            cache = DynamicCache()
            for index, pair in enumerate(presents):
                if not self.batch_first:
                    pair = tuple(t.permute(1, 2, 0, 3) for t in pair)
                cache.update(pair[0], pair[1], index)
        if not return_dict:
            return tuple(value for value in (hidden, cache, states, attentions) if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=cache, hidden_states=states, attentions=attentions
        )


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"transformer.output_layer.weight": "transformer.embedding.word_embeddings.weight"}

    def __init__(self, config, empty_init=False, device=None):
        super().__init__(config)
        self.max_sequence_length = config.seq_length
        self.transformer = ChatGLMModel(config, device=device, empty_init=empty_init)
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.transformer.output_layer

    def set_output_embeddings(self, value):
        self.transformer.output_layer = value

    def forward(
        self, input_ids=None, position_ids=None, attention_mask=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, return_last_logit=False, logits_to_keep=0,
        **kwargs,
    ):
        if output_attentions:
            raise ValueError("ChatGLM does not return attention probabilities")
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.transformer(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_hidden_states=output_hidden_states, return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if not self.transformer.batch_first:
            hidden = hidden.transpose(0, 1)
        if return_last_logit:
            logits_to_keep = 1
        # Without final normalization the encoder still owns an FP32 residual
        # stream. The language head consumes the embedding/model compute dtype.
        hidden = hidden.to(self.transformer.embedding.word_embeddings.weight.dtype)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.transformer.output_layer(hidden[:, indices]).contiguous()
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                ignore_index=-100,
            ).to(hidden.dtype)
        if not return_dict:
            values = (logits,) + tuple(
                value for value in (outputs.past_key_values, outputs.hidden_states) if value is not None
            )
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
