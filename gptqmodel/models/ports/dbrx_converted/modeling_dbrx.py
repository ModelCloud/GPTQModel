# Copyright 2024 Databricks, Inc. All rights reserved.
# Modifications Copyright 2026 ModelCloud.ai.
# Licensed under the Databricks Open Model License; see LICENSE.txt and NOTICE.txt.
"""Transformers 5 port of the ModelCloud converted DBRX checkpoint code.

This is the converted architecture (split q/k/v projections, one ``nn.Linear``
per expert), not the native fused-QKV/packed-expert DBRX. Attention, RoPE
application and the load-balancing loss are shared with the native TF5 DBRX.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.dbrx.modeling_dbrx import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .configuration_dbrx import DbrxConfig


class DbrxRotaryEmbedding(nn.Module):
    """Default RoPE; converted checkpoints keep ``rope_theta`` in ``attn_config``."""

    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model // config.n_heads
        self.base = config.attn_config.rope_theta
        self.register_buffer("inv_freq", self.compute_inv_freq(), persistent=False)

    def compute_inv_freq(self):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class DbrxAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.clip_qkv = config.attn_config.clip_qkv
        self.attention_dropout = config.attn_config.attn_pdrop
        self.is_causal = True
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        states = [proj(hidden_states) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        if self.clip_qkv is not None:
            states = [state.clamp(-self.clip_qkv, self.clip_qkv) for state in states]
        query, key, value = (state.view(hidden_shape).transpose(1, 2) for state in states)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, weights = attention_interface(
            self, query, key, value, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
        )
        return self.out_proj(output.reshape(*input_shape, -1).contiguous()), weights


class DbrxNormAttentionNorm(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = nn.LayerNorm(config.d_model, bias=False)
        self.attn = DbrxAttention(config, layer_idx)
        self.norm_2 = nn.LayerNorm(config.d_model, bias=False)
        # The converted checkpoint implementation freezes both block norms.
        self.norm_1.requires_grad_(False)
        self.norm_2.requires_grad_(False)

    def forward(self, hidden_states, **kwargs):
        attention, _ = self.attn(self.norm_1(hidden_states).to(hidden_states.dtype), **kwargs)
        residual = hidden_states + F.dropout(attention, p=self.resid_pdrop, training=self.training)
        return residual, self.norm_2(residual).to(residual.dtype)


class DbrxRouter(nn.Module):
    """Returns softmax probabilities (not logits), as the converted source does."""

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

    def forward(self, x, top_weights, top_experts):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        output = torch.zeros_like(x)
        expert_mask = F.one_hot(top_experts, num_classes=len(self.mlp)).permute(2, 1, 0)
        for index, expert in enumerate(self.mlp):
            slot, token = torch.where(expert_mask[index])
            if token.numel():
                output.index_add_(0, token, expert(x[token]) * top_weights[token, slot, None])
        return output.reshape(shape)


class DbrxFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = DbrxRouter(config)
        self.experts = DbrxExperts(config)

    def forward(self, x):
        # The converted checkpoint implementation freezes the router.
        with torch.no_grad():
            _, top_weights, top_experts = self.router(x)
        return self.experts(x, top_weights, top_experts)


class DbrxBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.resid_pdrop = config.resid_pdrop
        self.norm_attn_norm = DbrxNormAttentionNorm(config, layer_idx)
        self.ffn = DbrxFFN(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        residual, hidden_states = self.norm_attn_norm(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings,
            past_key_values=past_key_values, **kwargs,
        )
        return residual + F.dropout(self.ffn(hidden_states), p=self.resid_pdrop, training=self.training)


class DbrxPreTrainedModel(PreTrainedModel):
    config_class = DbrxConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DbrxBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False  # data-dependent expert dispatch
    _can_record_outputs = {
        "router_logits": OutputRecorder(DbrxRouter, index=0),
        "hidden_states": DbrxBlock,
        "attentions": DbrxAttention,
    }
    _input_embed_layer = "wte"

    @torch.no_grad()
    def _init_weights(self, module):
        # The converted source draws every Linear/Embedding/LayerNorm weight from N(0, std).
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                if not getattr(module.weight, "_is_hf_initialized", False):
                    init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, DbrxRotaryEmbedding):
            init.copy_(module.inv_freq, module.compute_inv_freq())


class DbrxModel(DbrxPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.blocks = nn.ModuleList([DbrxBlock(config, index) for index in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, bias=False)
        self.rotary_emb = DbrxRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = F.dropout(inputs_embeds, p=self.config.emb_pdrop, training=self.training)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for block in self.blocks:
            hidden_states = block(
                hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )
        return MoeModelOutputWithPast(last_hidden_state=self.norm_f(hidden_states), past_key_values=past_key_values)


class DbrxForCausalLM(DbrxPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = DbrxModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k
        self.post_init()

    @staticmethod
    def create_masks_for_generate(*, attention_mask=None, **kwargs):
        # DbrxModel builds the causal mask; the router loss still needs the 2D padding mask.
        return attention_mask

    @can_return_tuple
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_router_logits=None, logits_to_keep=0, **kwargs,
    ):
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        router_mask = None
        if output_router_logits and attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("Router auxiliary loss requires a 2D padding mask.")
            # The padding mask spans cached + current (or static-capacity) positions; route only current ones.
            start = past_key_values.get_seq_length() if past_key_values is not None else 0
            length = (input_ids if input_ids is not None else inputs_embeds).shape[1]
            positions = torch.arange(length, device=attention_mask.device) + start
            router_mask = attention_mask.index_select(1, positions)
        output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_router_logits=output_router_logits, **kwargs,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(output.last_hidden_state[:, indices]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                output.router_logits, self.num_experts, self.num_experts_per_tok, router_mask,
            )
            if loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)
        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=output.past_key_values,
            hidden_states=output.hidden_states, attentions=output.attentions, router_logits=output.router_logits,
        )
