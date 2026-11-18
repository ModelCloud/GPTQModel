# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LongLLaMA model."""

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)

from .configuration_longllama import LongLlamaConfig
from .longllama_utils import LongLlamaMemCache, LongLlamaMemConfig, mem_apply_update


logger = logging.get_logger(__name__)


@dataclass
class LongLlamaModelOutputWithPast(BaseModelOutputWithPast):
    mem_caches: Optional[LongLlamaMemCache] = None


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LongLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class LongLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=x.device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_one(x, cos, sin, position_ids):
    if len(position_ids.shape) != 2 or x.shape[0] != position_ids.shape[0] or x.shape[-2] != position_ids.shape[1]:
        raise ValueError(f"Position ids shoud have shape [bsz, seq_len] got {position_ids.shape}")
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    x_embed = x * cos + rotate_half(x) * sin
    return x_embed


def rotate_as_if_first(x, rotary_emb):
    cos, sin = rotary_emb(x, x.shape[-2])
    return rotate_one(x, cos, sin, torch.zeros(x.shape[0], x.shape[-2], dtype=torch.long, device=cos.device))


class LongLlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LongLlamaAttention(nn.Module):
    def __init__(self, config: LongLlamaConfig, mem_config: Optional[LongLlamaMemConfig] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.max_cache = self.max_position_embeddings
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LongLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        self.mem_config = mem_config

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        mem_cache: Optional[LongLlamaMemCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is None:
            tgt_seq_len = hidden_states.shape[-2]
            if past_key_value is not None:
                src_seq_len = past_key_value[0].shape[-2] + tgt_seq_len
            else:
                src_seq_len = tgt_seq_len
            attention_mask = torch.zeros(
                hidden_states.shape[0],
                1,
                tgt_seq_len,
                src_seq_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        position_ids = position_ids[:, None, :, None]
        if position_ids.shape != (key_states.shape[0], 1, key_states.shape[-2], 1):
            raise ValueError("position_ids should match batch and seq_len of the input")
        mem_no_local_cache = self.mem_config is not None and past_key_value is None and (not use_cache)
        mem_and_local_cache = self.mem_config is not None and use_cache
        use_positionals = self.mem_config is None or self.mem_config.positionals
        if mem_no_local_cache:
            if use_positionals:
                rfst_key_states = rotate_as_if_first(key_states, self.rotary_emb)
            else:
                rfst_key_states = key_states
            mem_update = LongLlamaMemCache(
                keys=rfst_key_states.to(self.mem_config.cache_dtype),
                values=value_states.to(self.mem_config.cache_dtype),
                masks=attention_mask[..., -1, :, None],
            )
        if past_key_value is not None:
            past_local_cache_size = past_key_value[0].shape[-2]
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)
            position_ids = torch.cat([past_key_value[2], position_ids], dim=-2)
            if attention_mask.shape[-1] != key_states.shape[-2] and attention_mask.shape[-2] != query_states.shape[-2]:
                raise ValueError("attention_mask should be provided for all key_states in local context")
            if key_states.shape[-2] > self.max_cache:
                num_elems_to_drop = past_local_cache_size
                if mem_and_local_cache:
                    drop_keys = key_states[:, :, :num_elems_to_drop, :]
                    drop_values = value_states[:, :, :num_elems_to_drop, :]
                    drop_masks = attention_mask[..., -1, :, None]
                    drop_masks = drop_masks[:, :, :num_elems_to_drop, :]
                    if use_positionals:
                        rfst_drop_keys = rotate_as_if_first(drop_keys, self.rotary_emb)
                    else:
                        rfst_drop_keys = drop_keys
                    mem_update = LongLlamaMemCache(
                        keys=rfst_drop_keys.to(self.mem_config.cache_dtype),
                        values=drop_values.to(self.mem_config.cache_dtype),
                        masks=drop_masks,
                    )
                    if mem_cache is None:
                        mem_cache = mem_update
                    else:
                        mem_cache = mem_apply_update(
                            prev_mem_cache=mem_cache, new_mem_content=mem_update, mem_config=self.mem_config
                        )
                key_states = key_states[:, :, num_elems_to_drop:, :]
                value_states = value_states[:, :, num_elems_to_drop:, :]
                position_ids = position_ids[:, :, num_elems_to_drop:, :]
                attention_mask = attention_mask[..., num_elems_to_drop:]
        past_key_value = (key_states, value_states, position_ids) if use_cache else None
        kv_seq_len = key_states.shape[-2]
        if use_positionals:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            rel_pos_ids = position_ids - torch.min(position_ids, dim=-2, keepdim=True)[0]
            rel_pos_ids = rel_pos_ids.squeeze(3).squeeze(1)
            query_states = rotate_one(query_states, cos, sin, rel_pos_ids[:, -query_states.shape[-2] :])
            key_states = rotate_one(key_states, cos, sin, rel_pos_ids)
        if self.mem_config is not None and self.mem_config.attention_grouping is not None:
            attn_grouping_h, attn_grouping_q = self.mem_config.attention_grouping
            if attn_grouping_h <= 0 or attn_grouping_q <= 0:
                raise ValueError("Attention grouping should be positive")
        else:
            attn_grouping_h, attn_grouping_q = (self.num_heads, q_len)
        attn_output_h = []
        for beg_h in range(0, self.num_heads, attn_grouping_h):
            end_h = min(beg_h + attn_grouping_h, self.num_heads)
            attn_output_q = []
            for beg_q in range(0, q_len, attn_grouping_q):
                end_q = min(beg_q + attn_grouping_q, q_len)
                if self.config.torch_attention and not output_attentions:
                    if mem_cache is not None:
                        attn_keys = torch.concat(
                            [key_states[:, beg_h:end_h], mem_cache.keys[:, beg_h:end_h].to(key_states.dtype)], dim=-2
                        )
                        attn_values = torch.concat(
                            [value_states[:, beg_h:end_h], mem_cache.values[:, beg_h:end_h].to(value_states.dtype)],
                            dim=-2,
                        )
                        mem_mask = mem_cache.masks.squeeze(-1).unsqueeze(-2)
                        assert len(mem_mask.shape) == 4
                        assert mem_mask.shape[2] == 1
                        assert mem_mask.shape[3] == mem_cache.keys.shape[-2]
                        mem_mask = torch.broadcast_to(
                            mem_mask, (mem_mask.shape[0], mem_mask.shape[1], end_q - beg_q, mem_mask.shape[3])
                        )
                        attn_mask = torch.concat([attention_mask[:, :, beg_q:end_q], mem_mask], dim=-1)
                        assert attn_mask.shape[-1] == attn_keys.shape[-2]
                    else:
                        attn_keys = key_states[:, beg_h:end_h]
                        attn_values = value_states[:, beg_h:end_h]
                        attn_mask = attention_mask[:, :, beg_q:end_q]
                    attn_queries = query_states[:, beg_h:end_h, beg_q:end_q]
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query=attn_queries, key=attn_keys, value=attn_values, attn_mask=attn_mask
                    )
                    attn_output_q.append(attn_output)
                else:
                    attn_weights = torch.matmul(
                        query_states[:, beg_h:end_h, beg_q:end_q], key_states[:, beg_h:end_h].transpose(2, 3)
                    ) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, end_h - beg_h, end_q - beg_q, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, end_h - beg_h, end_q - beg_q, kv_seq_len)}, but is {attn_weights.size()}"
                        )
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                    attn_weights = attn_weights + attention_mask[:, :, beg_q:end_q]
                    min_value = (
                        torch.finfo(attn_weights.dtype).min
                        if -1000000.0 < torch.finfo(attn_weights.dtype).min
                        else -1000000.0
                    )
                    attn_weights = torch.max(
                        attn_weights, torch.tensor(min_value, device=attn_weights.device, dtype=attn_weights.dtype)
                    )
                    if mem_cache is not None:
                        mem_mask = mem_cache.masks.squeeze(-1).unsqueeze(-2)
                        mem_attn_weights = torch.matmul(
                            query_states[:, beg_h:end_h, beg_q:end_q],
                            mem_cache.keys[:, beg_h:end_h].transpose(2, 3).to(key_states.dtype),
                        ) / math.sqrt(self.head_dim)
                        assert mem_mask.shape[2] == 1
                        mem_attn_weights = mem_attn_weights + mem_mask
                        min_value = (
                            torch.finfo(mem_attn_weights.dtype).min
                            if -1000000.0 < torch.finfo(mem_attn_weights.dtype).min
                            else -1000000.0
                        )
                        mem_attn_weights = torch.max(
                            mem_attn_weights,
                            torch.tensor(min_value, device=mem_attn_weights.device, dtype=mem_attn_weights.dtype),
                        )
                        attn_weights = torch.concat([attn_weights, mem_attn_weights], dim=-1)
                        combined_value_states = torch.concat(
                            [value_states[:, beg_h:end_h], mem_cache.values[:, beg_h:end_h].to(value_states.dtype)],
                            dim=-2,
                        )
                    else:
                        combined_value_states = value_states[:, beg_h:end_h]
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                        query_states.dtype
                    )
                    attn_output = torch.matmul(attn_weights, combined_value_states)
                    assert attn_output.shape[-2] == end_q - beg_q
                    attn_output_q.append(attn_output)
            attn_output_h.append(torch.concat(attn_output_q, dim=-2))
        attn_output = torch.concat(attn_output_h, dim=-3)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        if mem_no_local_cache:
            if mem_cache is not None:
                mem_cache = mem_apply_update(
                    prev_mem_cache=mem_cache, new_mem_content=mem_update, mem_config=self.mem_config
                )
            else:
                mem_cache = mem_update
        return (attn_output, attn_weights, past_key_value, mem_cache)


class LongLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LongLlamaConfig, mem_config: Optional[LongLlamaMemConfig] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LongLlamaAttention(config=config, mem_config=mem_config)
        self.mlp = LongLlamaMLP(
            hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act
        )
        self.input_layernorm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        mem_cache: Optional[LongLlamaMemCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value, mem_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            mem_cache=mem_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs + (mem_cache,)


class LongLlamaPreTrainedModel(PreTrainedModel):
    config_class = LongLlamaConfig
    _keep_in_fp32_modules_strict = ["rotary_emb.inv_freq"]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LongLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = ["decoder\\.version"]

    @torch.no_grad()
    def _init_weights(self, module):
        # TF5 base init covers Linear/Embedding/RMSNorm via guarded `initialization` helpers.
        super()._init_weights(module)
        if isinstance(module, LongLlamaRotaryEmbedding):
            dim = module.inv_freq.numel() * 2
            init.copy_(
                module.inv_freq,
                1.0 / (10000 ** (torch.arange(0, dim, 2, device=module.inv_freq.device).float() / dim)),
            )


def _prepare_pos_ids(past_key_values, batch_size, input_length, device):
    if past_key_values is not None:
        if past_key_values[0][2].shape[0] != batch_size:
            raise ValueError(
                f"first dimension of past_key_values should match batch size: {batch_size}but got {past_key_values[0][2].shape[0]}"
            )
        next_pos = torch.max(past_key_values[0][2].view(batch_size, -1), dim=-1)[0] + 1
        next_pos = next_pos.view(batch_size, 1)
    else:
        next_pos = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
    position_ids = torch.arange(0, input_length, dtype=torch.long, device=device).view(1, input_length)
    position_ids = position_ids + next_pos
    return position_ids


class LongLlamaModel(LongLlamaPreTrainedModel):
    def __init__(self, config: LongLlamaConfig):
        super().__init__(config)
        self.mem_layers = config.mem_layers
        self.mem_config = LongLlamaMemConfig(
            positionals=config.mem_positionals,
            cache_dtype=getattr(torch, config.mem_dtype),
            attention_grouping=config.mem_attention_grouping,
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        for mem_layer_id in self.mem_layers:
            if mem_layer_id < 0 or mem_layer_id >= config.num_hidden_layers:
                raise ValueError(
                    f"Memory layer ids should be between 0 and {config.num_hidden_layers}, got {mem_layer_id}"
                )
        layers = []
        for layer_id in range(config.num_hidden_layers):
            if layer_id in self.mem_layers:
                layer = LongLlamaDecoderLayer(config, mem_config=self.mem_config)
            else:
                layer = LongLlamaDecoderLayer(config, mem_config=None)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = LongLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mem_caches: Optional[Tuple[Optional[LongLlamaMemCache]]] = None,
    ) -> Union[Tuple, LongLlamaModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = _prepare_pos_ids(past_key_values, batch_size, seq_length, device)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()
        next_mem_caches = ()
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            mem_cache = mem_caches[idx] if mem_caches else None
            if mem_cache is not None and idx not in self.mem_layers:
                raise ValueError("Memory cache provided for a non-memory leayer")
            if (
                self.gradient_checkpointing
                and self.training
                and (mem_cache is None)
                and (idx % self.config.gradient_checkpoint_every_ith == 0)
            ):

                # TF5 installs `_gradient_checkpointing_func` (explicit use_reentrant).
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, output_attentions=output_attentions, use_cache=None, mem_cache=None),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    mem_cache=mem_cache,
                )
            new_mem_cache = layer_outputs[-1]
            layer_outputs = layer_outputs[:-1]
            next_mem_caches += (new_mem_cache,)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            else:
                next_decoder_cache += (None,)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        mem_cache_returned = False
        for mem_cache in next_mem_caches:
            if mem_cache is not None:
                mem_cache_returned = True
        next_mem_caches = next_mem_caches if mem_cache_returned else None
        if not return_dict:
            return tuple(
                (
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, next_mem_caches]
                    if v is not None
                )
            )
        return LongLlamaModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mem_caches=next_mem_caches,
        )


class LongLlamaCache(Cache):
    """FoT's six tensors per layer, not an ordinary rotated-K/V cache.

    Local K/V are unrotated; memory K/V and padding masks have independent
    ownership. Updates return a new cache, leaving caller-owned caches intact.
    """

    def __init__(self, entries):
        super().__init__(layers=[])
        self.entries = tuple(entries)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    @property
    def is_compileable(self):
        return False

    @property
    def is_croppable(self):
        return False

    @property
    def is_initialized(self):
        return bool(self.entries) and self.get_seq_length() > 0

    @property
    def batch_size(self):
        return self.entries[0][0].shape[0] if self.entries else -1

    def get_seq_length(self, layer_idx=0):
        positions = self.entries[layer_idx][2]
        if positions.numel():
            return int(positions.max().item()) + 1
        return self.entries[layer_idx][3].shape[-2]

    def reorder_cache(self, beam_idx):
        self.entries = tuple(tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer) for layer in self.entries)

    def batch_repeat_interleave(self, repeats):
        self.entries = tuple(tuple(t.repeat_interleave(repeats, dim=0) for t in layer) for layer in self.entries)

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def reset(self):
        self.entries = tuple(tuple(t[..., :0, :] for t in layer) for layer in self.entries)

    def crop(self, tokens_to_remove):
        raise ValueError("LongLLaMA memory eviction cannot be rolled back by cropping a local cache")


def _handle_output_of_past_key_values(outputs):
    batch_size = outputs.last_hidden_state.shape[0]
    if outputs.past_key_values is None and outputs.mem_caches is None:
        return None
    if outputs.past_key_values is None:
        out_past_key_values = (None,) * len(outputs.mem_caches)
    else:
        out_past_key_values = outputs.past_key_values
    if outputs.mem_caches is None:
        out_mem_caches = (None,) * len(outputs.past_key_values)
    else:
        out_mem_caches = outputs.mem_caches
    device = outputs.last_hidden_state.device
    past_key_values = ()
    for local_cache, mem_cache in zip(out_past_key_values, out_mem_caches):
        layer = ()
        if local_cache is not None:
            assert len(local_cache) == 3
            layer += local_cache
        else:
            layer += (torch.empty(batch_size, 0, 0, 0, device=device),) * 3
        if mem_cache is not None:
            layer += (mem_cache.keys, mem_cache.values, mem_cache.masks)
        else:
            layer += (torch.empty(batch_size, 0, 0, 0, device=device),) * 3
        assert len(layer) == 6
        past_key_values += (layer,)
    return LongLlamaCache(past_key_values)


def _split_past_key_values(past_key_values):
    if past_key_values is not None and not isinstance(past_key_values, (LongLlamaCache, tuple, list)):
        raise ValueError("LongLLaMA requires its own six-tensor memory/local cache")
    local_cache_preset = False
    mem_caches_present = False
    if past_key_values is not None:
        local_caches = ()
        mem_caches = ()
        for layer in past_key_values:
            if len(layer) != 6:
                raise ValueError(
                    f"Expected elements of past_key_values to contain 6 elements.First 3 describing local cache and last 3 describing memory cache.Instead got {len(layer)} elements"
                )
            else:
                lk, lv, li, memk, memv, memm = layer
                if lk.shape[-2] != 0:
                    local_cache_preset = True
                    local_caches += ((lk, lv, li),)
                else:
                    local_caches += (None,)
                if memk.shape[-2] != 0:
                    mem_caches_present = True
                    mem_caches += (LongLlamaMemCache(keys=memk, values=memv, masks=memm),)
                else:
                    mem_caches += (None,)
    local_caches = local_caches if local_cache_preset else None
    mem_caches = mem_caches if mem_caches_present else None
    return (local_caches, mem_caches)


def _handle_long_input(
    model,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    use_cache,
    output_attentions,
    output_hidden_states,
    return_dict,
    context_window_length,
    last_context_length,
):
    if output_attentions:
        logger.warning(
            "Outputing attentions is not supported in LongLlamaForCausalLM. Attention of the last window will be returned"
        )
    past_key_values, mem_caches = _split_past_key_values(past_key_values)
    if past_key_values is not None and use_cache is False:
        raise ValueError("past_key_values it not None should imply use_cache == True")
    if past_key_values is not None:
        initial_past_key_values_length = past_key_values[0][0].shape[-2]
    else:
        initial_past_key_values_length = 0
    if input_ids is not None:
        batch_size, input_length = input_ids.shape
    else:
        batch_size, input_length, _ = inputs_embeds.shape
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = _prepare_pos_ids(past_key_values, batch_size, input_length, device)
    if position_ids.shape != (batch_size, input_length):
        raise ValueError(f"Shape of position_ids [{position_ids}] should match [{(batch_size, input_length)}]")
    if attention_mask is not None:
        attention_mask = attention_mask[..., -(initial_past_key_values_length + input_length) :]
        if attention_mask is not None and attention_mask.shape != (
            batch_size,
            initial_past_key_values_length + input_length,
        ):
            raise ValueError(
                "Attention mask should be provided for both the local cache and the input",
                f"Expected shape {(batch_size, initial_past_key_values_length + input_length)},got {attention_mask.shape}.",
            )
    mem_input_length = max(input_length - last_context_length, 0)
    outputs_list = []
    attn_offset = initial_past_key_values_length
    if mem_input_length > 0:
        for i in range(0, mem_input_length, context_window_length):
            beg, end = (i, min(mem_input_length, i + context_window_length))
            if attention_mask is not None:
                if past_key_values is not None:
                    local_cache_size = past_key_values[0][0].shape[-2]
                else:
                    local_cache_size = 0
                attn_length = attention_mask.shape[-1]
                attn_beg = beg + attn_offset - local_cache_size
                attn_end = end + attn_offset
                assert attn_end <= attn_length
                assert attn_beg >= 0 and attn_end > attn_beg
            outputs = model(
                input_ids=input_ids[..., beg:end] if input_ids is not None else None,
                attention_mask=attention_mask[..., attn_beg:attn_end] if attention_mask is not None else None,
                position_ids=position_ids[..., beg:end],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[..., beg:end, :] if inputs_embeds is not None else None,
                use_cache=False if past_key_values is None else use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                mem_caches=mem_caches,
            )
            mem_caches = outputs.mem_caches
            outputs.mem_caches = None
            past_key_values = outputs.past_key_values
            outputs.past_key_values = None
            outputs_list.append(outputs)
    remaining_input_length = input_length - mem_input_length
    beg = mem_input_length
    attn_length = remaining_input_length
    if past_key_values is not None:
        attn_length += past_key_values[0][0].shape[-2]
    attention_mask = attention_mask[..., -attn_length:] if attention_mask is not None else None
    outputs = model(
        input_ids=input_ids[..., beg:] if input_ids is not None else None,
        attention_mask=attention_mask,
        position_ids=position_ids[..., beg:],
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds[..., beg:, :] if inputs_embeds is not None else None,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        mem_caches=mem_caches,
    )
    outputs_list.append(outputs)
    past_key_values = _handle_output_of_past_key_values(outputs_list[-1])
    if output_hidden_states:
        hidden_states = ()
        for hd in zip(*[x.hidden_states for x in outputs_list]):
            hidden_states += (torch.cat(hd, dim=-2),)
    else:
        hidden_states = None
    outputs = BaseModelOutputWithPast(
        last_hidden_state=torch.concat([x.last_hidden_state for x in outputs_list], dim=-2),
        past_key_values=past_key_values,
        hidden_states=hidden_states,
        attentions=outputs_list[-1].attentions,
    )
    if not return_dict:
        outputs = tuple(
            (
                v
                for v in [
                    outputs.last_hidden_state,
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                ]
                if v is not None
            )
        )
    return outputs


class LongLlamaForCausalLM(LongLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def __init__(self, config):
        super().__init__(config)
        self.context_window_length = config.max_position_embeddings
        self.model = LongLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _has_generation_cache(self, past_key_values):
        if past_key_values is not None:
            assert len(past_key_values[0]) == 6
            return past_key_values[0][0].shape[-2] != 0
        return False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        last_context_length: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        last_context_length = (
            last_context_length if last_context_length is not None else self.config.last_context_length
        )
        if not 0 < last_context_length <= self.context_window_length:
            raise ValueError("last_context_length must be positive and no larger than the context window")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = _handle_long_input(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context_window_length=self.context_window_length,
            last_context_length=last_context_length,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        last_context_length=None,
        next_sequence_length=None,
        **kwargs,
    ):
        if self._has_generation_cache(past_key_values):
            if next_sequence_length is not None:
                input_ids = input_ids[:, -next_sequence_length:]
            elif input_ids.shape[1] > past_key_values.get_seq_length():
                input_ids = input_ids[:, past_key_values.get_seq_length() :]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill(position_ids < 0, 0)
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :].expand(input_ids.shape[0], -1)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "last_context_length": last_context_length,
            }
        )
        return model_inputs
