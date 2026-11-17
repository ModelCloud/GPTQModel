# Copyright 2024 Databricks, Inc. All rights reserved.
# Modifications Copyright 2026 ModelCloud.ai.
# Licensed under the Databricks Open Model License; see LICENSE.txt and NOTICE.txt.
"""Transformers 5 port of the ModelCloud converted DBRX checkpoint code.

This is the converted architecture, not the native fused-QKV/packed-expert DBRX.
The pinned source and adaptation boundaries are recorded in PROVENANCE.md.
"""

import math
from contextlib import contextmanager
from functools import partial
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast

from .configuration_dbrx import DbrxConfig


@contextmanager
def _cache_transaction(cache):
    if not isinstance(cache, Cache):
        yield
        return
    if getattr(cache, "_dbrx_invalid", False):
        if any(cache.get_seq_length(index) != 0 for index in range(len(cache.layers))):
            raise ValueError("Failed converted DBRX forward invalidated this cache; call reset() and prefill the full prompt.")
        setattr(cache, "_dbrx_invalid", False)
    version = getattr(cache, "_dbrx_write_version", 0)
    try:
        yield
    except BaseException:
        if getattr(cache, "_dbrx_write_version", 0) != version:
            setattr(cache, "_dbrx_invalid", True)
        raise


class DbrxRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim, base):
        super().__init__()
        self.dim, self.base = dim, base
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (self.inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class DbrxAttention(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.block_idx = block_idx
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.attn_pdrop = config.attn_config.attn_pdrop
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary_emb = DbrxRotaryEmbedding(self.head_dim, config.attn_config.rope_theta)

    def forward(self, hidden_states, position_ids, attention_mask=None, past_key_value=None, **kwargs):
        batch, length, _ = hidden_states.shape
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        if self.clip_qkv is not None:
            q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if past_key_value is not None:
            setattr(past_key_value, "_dbrx_write_version", getattr(past_key_value, "_dbrx_write_version", 0) + 1)
            k, v = past_key_value.update(k, v, self.block_idx)
        repeats = self.num_heads // self.num_key_value_heads
        k, v = (t.repeat_interleave(repeats, dim=1) for t in (k, v))
        weights = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            weights = weights + attention_mask[..., :k.shape[-2]]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        weights = F.dropout(weights, p=self.attn_pdrop, training=self.training)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.out_proj(output), weights


class DbrxNormAttentionNorm(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DbrxAttention(config, block_idx)
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)
        self.norm_1.requires_grad_(False)
        self.norm_2.requires_grad_(False)

    def forward(self, hidden_states, **kwargs):
        attention, weights = self.attn(self.norm_1(hidden_states).to(hidden_states.dtype), **kwargs)
        residual = hidden_states + F.dropout(attention, p=self.resid_pdrop, training=self.training)
        return residual, self.norm_2(residual).to(residual.dtype), weights


class DbrxRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn = config.ffn_config
        self.moe_num_experts, self.moe_top_k = ffn.moe_num_experts, ffn.moe_top_k
        self.moe_jitter_eps = ffn.moe_jitter_eps
        self.moe_normalize_expert_weights = ffn.moe_normalize_expert_weights
        self.uniform_expert_assignment = ffn.uniform_expert_assignment
        self.layer = nn.Linear(config.d_model, self.moe_num_experts, bias=False)
        self.layer.requires_grad_(False)

    def forward(self, x):
        if self.training and self.moe_jitter_eps is not None:
            low, high = 1.0 - self.moe_jitter_eps, 1.0 + self.moe_jitter_eps
            x = x * (low + torch.rand_like(x) * (high - low))
        weights = self.layer(x.reshape(-1, x.shape[-1])).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = weights.topk(self.moe_top_k, dim=-1)
        if self.moe_normalize_expert_weights:
            top_weights = top_weights / torch.norm(
                top_weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True,
            )
        if self.uniform_expert_assignment:
            top_experts = (
                torch.arange(top_experts.numel(), device=x.device) % self.moe_num_experts
            ).reshape(top_experts.shape)
        return weights.to(x.dtype), top_weights.to(x.dtype), top_experts


class DbrxMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn = config.ffn_config
        self.w1 = nn.Linear(config.d_model, ffn.ffn_hidden_size, bias=False)
        self.v1 = nn.Linear(config.d_model, ffn.ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn.ffn_hidden_size, config.d_model, bias=False)
        activation = dict(ffn.ffn_act_fn)
        name = activation.pop("name")
        if not callable(getattr(F, name, None)):
            raise ValueError(f"Unrecognised FFN activation function: {name}")
        self.activation_fn = partial(getattr(F, name), **activation)

    def forward(self, x):
        return self.w2(self.activation_fn(self.w1(x)) * self.v1(x))


class DbrxExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.ModuleList([DbrxMLP(config) for _ in range(config.ffn_config.moe_num_experts)])

    def forward(self, x, weights, top_weights, top_experts):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        output = torch.zeros_like(x)
        expert_mask = F.one_hot(top_experts, num_classes=len(self.mlp)).permute(2, 1, 0)
        for index, expert in enumerate(self.mlp):
            slot, token = torch.where(expert_mask[index])
            if token.numel():
                result = expert(x[token]) * top_weights[token, slot, None]
                output.index_add_(0, token, result)
        return output.reshape(shape)


class DbrxFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = DbrxRouter(config)
        self.experts = DbrxExperts(config)

    def forward(self, x):
        # The converted checkpoint implementation freezes the router and both block norms.
        with torch.no_grad():
            weights, top_weights, top_experts = self.router(x)
        return self.experts(x, weights, top_weights, top_experts), weights


class DbrxBlock(nn.Module):
    def __init__(self, config, block_idx):
        super().__init__()
        self.norm_attn_norm = DbrxNormAttentionNorm(config, block_idx)
        self.ffn = DbrxFFN(config)
        self.resid_pdrop = config.resid_pdrop

    def forward(
        self, hidden_states, position_ids, attention_mask=None, past_key_value=None,
        output_attentions=False, output_router_logits=False, use_cache=False, **kwargs,
    ):
        residual, hidden_states, weights = self.norm_attn_norm(
            hidden_states, position_ids=position_ids, attention_mask=attention_mask, past_key_value=past_key_value,
        )
        hidden_states, router_weights = self.ffn(hidden_states)
        output = (residual + F.dropout(hidden_states, p=self.resid_pdrop, training=self.training),)
        if output_attentions:
            output += (weights,)
        if use_cache:
            output += (past_key_value,)
        if output_router_logits:
            output += (router_weights,)
        return output


class DbrxPreTrainedModel(PreTrainedModel):
    config_class = DbrxConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["DbrxBlock"]
    _skip_keys_device_placement = ["past_key_values"]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.normal_(mean=0.0, std=self.config.initializer_range)
            bias = getattr(module, "bias", None)
            if bias is not None:
                bias.zero_()
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, DbrxRotaryEmbedding):
            module.inv_freq.copy_(module.frequencies().to(module.inv_freq.device))


class DbrxModel(DbrxPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.blocks = nn.ModuleList([DbrxBlock(config, index) for index in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        output_router_logits=None, return_dict=None, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Converted DBRX requires a Transformers 5 Cache, not legacy tuples.")
        if past_key_values is not None and not use_cache:
            raise ValueError("A supplied past_key_values cache requires use_cache=True.")
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if torch.is_autocast_enabled(inputs_embeds.device.type):
            inputs_embeds = inputs_embeds.to(torch.get_autocast_dtype(inputs_embeds.device.type))
        hidden_states = F.dropout(inputs_embeds, p=self.config.emb_pdrop, training=self.training)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        with _cache_transaction(past_key_values):
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            if position_ids is None:
                position_ids = torch.arange(past_length, past_length + hidden_states.shape[1], device=hidden_states.device)[None]
            if position_ids.shape[-1] != hidden_states.shape[1]:
                raise ValueError("position_ids must describe the current tokens.")
            if attention_mask is not None and attention_mask.ndim not in (2, 4):
                raise ValueError("attention_mask must be a 2D padding mask or 4D additive mask.")
            if output_router_logits and attention_mask is not None and attention_mask.ndim != 2:
                raise ValueError("Router auxiliary loss requires a 2D padding mask.")
            causal_mask = create_causal_mask(
                config=self.config, inputs_embeds=hidden_states, attention_mask=attention_mask,
                past_key_values=past_key_values, position_ids=position_ids,
            )
            all_hidden, all_attentions, all_router = (), (), ()
            for block in self.blocks:
                if output_hidden_states:
                    all_hidden += (hidden_states,)
                output = block(
                    hidden_states, position_ids=position_ids, attention_mask=causal_mask, past_key_value=past_key_values,
                    output_attentions=output_attentions, output_router_logits=output_router_logits, use_cache=use_cache,
                )
                hidden_states = output[0]
                if output_attentions:
                    all_attentions += (output[1],)
                if output_router_logits:
                    all_router += (output[-1],)
            hidden_states = self.norm_f(hidden_states)
            if output_hidden_states:
                all_hidden += (hidden_states,)
            output = MoeModelOutputWithPast(
                last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden if output_hidden_states else None,
                attentions=all_attentions if output_attentions else None,
                router_logits=cast(tuple[torch.Tensor], all_router) if output_router_logits else None,
            )
            return output if return_dict else output.to_tuple()


def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    if not isinstance(gate_logits, tuple):
        return torch.tensor(0.0)
    device = gate_logits[0].device
    # Preserve the converted source's probability-valued router output and its second softmax.
    weights = torch.cat([gate.to(device) for gate in gate_logits]).softmax(dim=-1)
    expert_mask = F.one_hot(weights.topk(top_k, dim=-1).indices, num_experts)
    if attention_mask is None:
        tokens_per_expert = expert_mask.float().mean(0)
        probabilities = weights.mean(0)
    else:
        valid = attention_mask.reshape(-1).to(device)
        valid = valid.repeat(weights.shape[0] // valid.numel())
        tokens_per_expert = (expert_mask.float() * valid[:, None, None]).sum(0) / valid.sum()
        probabilities = (weights * valid[:, None]).sum(0) / valid.sum()
    return (tokens_per_expert * probabilities[None]).sum() * num_experts


class DbrxForCausalLM(DbrxPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = DbrxModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.transformer

    def set_decoder(self, decoder):
        self.transformer = decoder

    @staticmethod
    def create_masks_for_generate(*, attention_mask=None, **kwargs):
        # DbrxModel constructs the causal mask; routing still needs the original padding mask.
        return attention_mask

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
        output_router_logits=None, return_dict=None, logits_to_keep=0, **kwargs,
    ):
        return_dict = self.config.return_dict if return_dict is None else return_dict
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        if past_key_values is None and (self.config.use_cache if use_cache is None else use_cache):
            past_key_values = DynamicCache(config=self.config)
        with _cache_transaction(past_key_values):
            router_start = 0
            if output_router_logits and attention_mask is not None and isinstance(past_key_values, Cache):
                router_start = past_key_values.get_seq_length()
                if isinstance(router_start, torch.Tensor):
                    router_start = router_start.clone()
            output = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits, return_dict=True, **kwargs,
            )
            indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(output.last_hidden_state[:, indices]).float()
            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            aux_loss = None
            if output_router_logits:
                router_mask = attention_mask
                if router_mask is not None:
                    positions = torch.arange(output.last_hidden_state.shape[1], device=router_mask.device)
                    if isinstance(router_start, torch.Tensor):
                        router_start = router_start.to(router_mask.device)
                    router_mask = router_mask.index_select(1, positions + router_start)
                aux_loss = load_balancing_loss_func(
                    output.router_logits, self.num_experts, self.num_experts_per_tok, router_mask,
                )
                if loss is not None:
                    loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
            result = MoeCausalLMOutputWithPast(
                loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=output.past_key_values,
                hidden_states=output.hidden_states, attentions=output.attentions, router_logits=output.router_logits,
            )
            return result if return_dict else result.to_tuple()
