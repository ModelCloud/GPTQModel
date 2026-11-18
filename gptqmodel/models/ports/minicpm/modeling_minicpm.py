# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 OpenBMB. Adapted for Transformers 5 by ModelCloud.ai, 2026.
# SPDX-License-Identifier: Apache-2.0
"""Scaled MiniCPM decoder and original w1/w2/w3 MoE, using Transformers 5 caches.

Derived from the OpenBMB implementations pinned in NOTICE. This
is not MiniCPM3: embeddings, both residuals and the output head have MiniCPM's
distinct scaling; eager RoPE retains FP32 trig, while SDPA uses V-dtype trig.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_minicpm import MiniCPMConfig


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().square().mean(-1, keepdim=True)
        normalized = (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).to(hidden_states.dtype)
        return normalized * self.weight


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len):
        config = self.config
        base = config.rope_theta
        positions = position_ids.float()
        kind = config.rope_parameters["rope_type"]
        if kind != "default":
            factor = config.rope_parameters["factor"]
            if kind == "linear":
                positions = positions / factor
            elif seq_len > config.max_position_embeddings:
                base *= (factor * seq_len / config.max_position_embeddings - (factor - 1)) ** (
                    config.head_dim / (config.head_dim - 2)
                )
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv_freq = 1.0 / (base ** (torch.arange(0, config.head_dim, 2, device=x.device).float() / config.head_dim))
            angles = positions.unsqueeze(-1) * inv_freq
            angles = torch.cat((angles, angles), dim=-1)
            return angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)


def _rotate(x, cos, sin):
    dtype = x.dtype
    x = x.float()
    a, b = x.chunk(2, dim=-1)
    return (x * cos + torch.cat((-b, a), dim=-1) * sin).to(dtype)


def _column_parallel(module, x, tp):
    return torch.cat([F.linear(x, w) for w in module.weight.chunk(tp, dim=0)], dim=-1)


def _row_parallel(module, x, tp):
    return sum(F.linear(part, w) for part, w in zip(x.chunk(tp, dim=-1), module.weight.chunk(tp, dim=1)))


class MiniCPMMLP(nn.Module):
    def __init__(self, config, expert=False):
        super().__init__()
        self.config = config
        self.names = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        for name in self.names[:2]:
            setattr(self, name, nn.Linear(config.hidden_size, config.intermediate_size, bias=False))
        setattr(self, self.names[2], nn.Linear(config.intermediate_size, config.hidden_size, bias=False))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate, up, down = [getattr(self, name) for name in self.names]
        tp = self.config.pretraining_tp
        if tp > 1:
            return _row_parallel(down, self.act_fn(_column_parallel(gate, x, tp)) * _column_parallel(up, x, tp), tp)
        return down(self.act_fn(gate(x)) * up(x))


class _AuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        ctx.dtype = loss.dtype
        ctx.required = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, torch.ones((), dtype=ctx.dtype, device=grad.device) if ctx.required else None


class MiniCPMMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MiniCPMMLP(config, expert=True) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        scores = F.softmax(self.gate(x), dim=-1, dtype=torch.float32)
        weights, indices = scores.topk(self.num_experts_per_tok, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                output.index_add_(0, rows, expert(x[rows]) * weights[rows, slots, None])
        if self.training:
            load = indices.reshape(-1).bincount(minlength=self.config.num_experts) / indices.numel()
            loss = self.config.num_experts * (scores.mean(0) * load).sum()
            output = _AuxiliaryLoss.apply(output, loss)
        return output.view(shape)


class MiniCPMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        for name, heads in (("q_proj", config.num_attention_heads), ("k_proj", config.num_key_value_heads),
                            ("v_proj", config.num_key_value_heads)):
            setattr(self, name, nn.Linear(config.hidden_size, heads * config.head_dim, bias=config.attention_bias))
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = MiniCPMRotaryEmbedding(config)

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value=None, output_attentions=False, **kwargs):
        c = self.config
        batch, length, _ = hidden_states.shape
        eager = c._attn_implementation == "eager" or output_attentions
        tp = c.pretraining_tp if eager else 1
        projections = []
        for module, heads in ((self.q_proj, c.num_attention_heads), (self.k_proj, c.num_key_value_heads),
                              (self.v_proj, c.num_key_value_heads)):
            value = _column_parallel(module, hidden_states, tp) if tp > 1 else module(hidden_states)
            projections.append(value.view(batch, length, heads, c.head_dim).transpose(1, 2))
        q, k, v = projections
        seq_len = length + (past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(v.float() if eager else v, position_ids, seq_len)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)
        repeats = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        dropout = c.attention_dropout if self.training else 0.0
        if eager:
            scores = q @ k.transpose(-1, -2) / math.sqrt(c.head_dim)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    # An SDPA-configured model returning attention weights
                    # receives the boolean mask from create_causal_mask.
                    attention_mask = torch.zeros_like(attention_mask, dtype=scores.dtype).masked_fill(
                        ~attention_mask, torch.finfo(scores.dtype).min
                    )
                scores = scores + attention_mask
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            weights = F.dropout(weights, p=dropout, training=self.training)
            output = weights @ v
        else:
            weights = None
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout,
                                                   is_causal=attention_mask is None and length > 1)
        output = output.transpose(1, 2).reshape(batch, length, c.hidden_size)
        output = _row_parallel(self.o_proj, output, tp) if tp > 1 else self.o_proj(output)
        return output, weights if output_attentions else None


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MiniCPMAttention(config, layer_idx)
        self.mlp = MiniCPMMoE(config) if config.num_experts else MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, **kwargs):
        attention, weights = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_ids,
                                           past_key_value=past_key_value, output_attentions=output_attentions)
        hidden_states = hidden_states + attention * self.residual_scale
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) * self.residual_scale
        return (hidden_states, weights) if output_attentions else (hidden_states,)


class MiniCPMPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                if not getattr(module.weight, "_is_hf_initialized", False):
                    init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, MiniCPMRMSNorm):
            init.ones_(module.weight)


class MiniCPMModel(MiniCPMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and (
            not isinstance(past_key_values, DynamicCache) or getattr(past_key_values, "offloading", False)
        ):
            raise ValueError("MiniCPM supports Transformers DynamicCache, not legacy tuples or static/offloaded caches")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + inputs_embeds.shape[1],
                                        device=inputs_embeds.device).unsqueeze(0)
        mask = create_causal_mask(self.config, inputs_embeds, attention_mask, past_key_values,
                                  position_ids=position_ids, allow_is_causal_skip=not output_attentions)
        hidden_states = inputs_embeds
        states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                states += (hidden_states,)
            outputs = layer(hidden_states, attention_mask=mask, position_ids=position_ids,
                            past_key_value=past_key_values, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                attentions += (outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            states += (hidden_states,)
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values,
                                         hidden_states=states, attentions=attentions)
        return output if return_dict else output.to_tuple()


class MiniCPMForCausalLM(MiniCPMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, value):
        self.model = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             return_dict=True, **kwargs)
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_states = hidden_states[:, logits_to_keep, :]
        # The original pretraining_tp branch intentionally omits dim_model_base scaling.
        if self.config.pretraining_tp > 1:
            logits = _column_parallel(self.lm_head, hidden_states, self.config.pretraining_tp).float()
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, self.vocab_size),
                                   labels[..., 1:].reshape(-1).to(logits.device))
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return output if return_dict else output.to_tuple()
