# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for Transformers 5 by ModelCloud.ai.
"""InternLM2/2.5 with checkpoint-native, KV-group-interleaved packed QKV."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_internlm2 import InternLM2Config


class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM2RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        params = config.rope_parameters or {}
        self.base = params.get("rope_theta", config.rope_theta)
        self.kind = params.get("rope_type", params.get("type", "default"))
        self.factor = params.get("factor", 1.0)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        positions = position_ids.float()
        inv_freq = self.inv_freq
        if self.kind == "linear":
            positions = positions / self.factor
        elif self.kind == "dynamic":
            # Request-local frequencies avoid long->short and concurrent-request state leakage.
            seq_len = (positions.max() + 1).clamp(min=self.max_position_embeddings)
            base = self.base * (
                self.factor * seq_len / self.max_position_embeddings - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq[None, :, None].float() @ positions[:, None, :]).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, heads, length, dim = hidden_states.shape
    return hidden_states[:, :, None].expand(batch, heads, n_rep, length, dim).reshape(
        batch, heads * n_rep, length, dim,
    )


class InternLM2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class InternLM2Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.wqkv = nn.Linear(
            config.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=config.bias,
        )
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.rotary_emb = InternLM2RotaryEmbedding(config)

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
        output_attentions=False, use_cache=False, cache_position=None,
    ):
        batch, length, _ = hidden_states.shape
        tp = self.config.pretraining_tp
        implementation = self.config._attn_implementation
        # Upstream eager tensor-parallel simulation omits QKV/output biases.
        if tp > 1 and implementation == "eager":
            width = self.num_key_value_heads * self.head_dim // tp
            qkv = torch.cat([F.linear(hidden_states, w) for w in self.wqkv.weight.split(width, dim=0)], dim=-1)
        else:
            qkv = self.wqkv(hidden_states)
        qkv = qkv.reshape(batch, length, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim)
        query = qkv[..., :self.num_key_value_groups, :].reshape(batch, length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key, value = qkv[..., -2, :].transpose(1, 2), qkv[..., -1, :].transpose(1, 2)
        cos, sin = self.rotary_emb(value, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        query, key = query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)
        key, value = repeat_kv(key, self.num_key_value_groups), repeat_kv(value, self.num_key_value_groups)
        mask = attention_mask[..., :key.shape[-2]] if attention_mask is not None else None
        weights = None
        if implementation == "sdpa" and not output_attentions:
            output = F.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=mask,
                dropout_p=0.0, is_causal=mask is None and length > 1,
            )
        else:
            weights = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            if mask is not None:
                if mask.dtype == torch.bool:
                    weights = weights.masked_fill(~mask, torch.finfo(weights.dtype).min)
                else:
                    weights = weights + mask
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            output = weights @ value
        output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
        if tp > 1 and implementation == "eager":
            parts = output.split(self.config.hidden_size // tp, dim=-1)
            output = sum(F.linear(x, w) for x, w in zip(parts, self.wo.weight.chunk(tp, dim=1)))
        else:
            output = self.wo(output)
        return output, weights if output_attentions else None, past_key_value


class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = InternLM2Attention(config, layer_idx)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        attention_output, weights, _ = self.attention(self.attention_norm(hidden_states), **kwargs)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, weights


class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = "model"
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, InternLM2RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, InternLM2RotaryEmbedding):
            inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float() / module.dim))
            module.inv_freq.copy_(inv_freq)


class InternLM2Model(InternLM2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("InternLM2 local port supports eager and sdpa attention only")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, cache_position=None, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("InternLM2 requires a Transformers Cache, not legacy tuple caches")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past
        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids, allow_is_causal_skip=False,
        )
        hidden_states = inputs_embeds
        all_hidden, all_attentions = (() if output_hidden_states else None), (() if output_attentions else None)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            hidden_states, weights = layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=past_key_values, use_cache=use_cache, output_attentions=output_attentions,
            )
            if output_attentions:
                all_attentions += (weights,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden, attentions=all_attentions,
        )
        return result if return_dict else result.to_tuple()


class InternLM2ForCausalLM(InternLM2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"output.weight": "model.tok_embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, value):
        self.output = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, logits_to_keep=0, **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, **kwargs,
        )
        hidden = outputs.last_hidden_state
        if labels is None:
            hidden = hidden[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) else hidden[:, logits_to_keep, :]
        if self.config.pretraining_tp > 1:
            logits = torch.cat([F.linear(hidden, w) for w in self.output.weight.chunk(self.config.pretraining_tp)], -1)
        else:
            logits = self.output(hidden)
        logits = logits.float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1).to(logits.device),
            )
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return result if return_dict else result.to_tuple()
