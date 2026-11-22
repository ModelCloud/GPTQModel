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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

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
    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        params = config.rope_parameters
        self.base = params["rope_theta"]
        self.kind = params["rope_type"]
        self.factor = params.get("factor", 1.0)
        self.register_buffer("inv_freq", self.compute_inv_freq(), persistent=False)

    def compute_inv_freq(self, base=None, device=None):
        base = self.base if base is None else base
        return 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))

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
            inv_freq = self.compute_inv_freq(base, x.device)
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq[None, :, None].float() @ positions[:, None, :]).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(hidden_states, n_rep):
    batch, heads, length, dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None].expand(batch, heads, n_rep, length, dim).reshape(
        batch, heads * n_rep, length, dim,
    )


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


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
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.wqkv = nn.Linear(
            config.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=config.bias,
        )
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        batch, length, _ = hidden_states.shape
        tp = self.config.pretraining_tp
        # Upstream eager tensor-parallel simulation omits QKV/output biases.
        tp_eager = tp > 1 and self.config._attn_implementation == "eager"
        if tp_eager:
            width = self.num_key_value_heads * self.head_dim // tp
            qkv = torch.cat([F.linear(hidden_states, w) for w in self.wqkv.weight.split(width, dim=0)], dim=-1)
        else:
            qkv = self.wqkv(hidden_states)
        qkv = qkv.reshape(batch, length, self.num_key_value_heads, self.num_key_value_groups + 2, self.head_dim)
        query = qkv[..., :self.num_key_value_groups, :].reshape(batch, length, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key, value = qkv[..., -2, :].transpose(1, 2), qkv[..., -1, :].transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention_interface(
            self, query, key, value, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs,
        )
        output = output.reshape(batch, length, -1).contiguous()
        if tp_eager:
            parts = output.split(self.config.hidden_size // tp, dim=-1)
            output = sum(F.linear(x, w) for x, w in zip(parts, self.wo.weight.chunk(tp, dim=1)))
        else:
            output = self.wo(output)
        return output, weights


class InternLM2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = InternLM2Attention(config, layer_idx)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False,
        position_embeddings=None, **kwargs,
    ):
        attention_output, _ = self.attention(
            self.attention_norm(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": InternLM2DecoderLayer,
        "attentions": InternLM2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, InternLM2RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, InternLM2RotaryEmbedding):
            init.copy_(module.inv_freq, module.compute_inv_freq())


class InternLM2Model(InternLM2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = InternLM2RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                position_embeddings=position_embeddings, **kwargs,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values)


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

    @can_return_tuple
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        hidden = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        if self.config.pretraining_tp > 1:
            logits = torch.cat([F.linear(hidden, w) for w in self.output.weight.chunk(self.config.pretraining_tp)], -1)
        else:
            logits = self.output(hidden)
        logits = logits.float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )
