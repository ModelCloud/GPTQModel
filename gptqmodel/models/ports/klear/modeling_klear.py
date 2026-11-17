# SPDX-FileCopyrightText: 2025 Kwai-Klear Team, Kuaishou
# SPDX-FileCopyrightText: 2024-2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Transformers 5 Klear port. Source revisions and changes are in README.md."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import can_return_tuple

from .configuration_klear import KlearConfig


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(hidden_states, n_rep):
    batch, heads, length, dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None].expand(batch, heads, n_rep, length, dim).reshape(
        batch, heads * n_rep, length, dim
    )


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key, value = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, :key.shape[-2]]
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class KlearRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype)


class KlearRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        self.max_seq_len_cached = self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, **kwargs):
        base, dim = config.rope_parameters["rope_theta"], config.head_dim
        return 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim)), 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = self.inv_freq[None, :, None].float().to(x.device) @ position_ids[:, None, :].float()
            emb = torch.cat((freqs.transpose(1, 2), freqs.transpose(1, 2)), dim=-1)
            cos, sin = emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


class KlearAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling, self.attention_dropout, self.is_causal = self.head_dim ** -0.5, config.attention_dropout, True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.attention_bias)
        self.q_norm = KlearRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = KlearRMSNorm(self.head_dim, config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_value=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        query = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx)
        interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        output, weights = interface(
            self, query, key, value, attention_mask, scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            sliding_window=self.sliding_window, **kwargs,
        )
        output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(output), weights


class KlearMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KlearSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts, self.top_k, self.norm_topk_prob = (
            config.num_experts, config.num_experts_per_tok, config.norm_topk_prob
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([KlearMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.shared_experts = KlearMLP(config, config.moe_intermediate_size * config.n_shared_experts)
        self.coefficient = nn.Linear(config.hidden_size, 2)
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        residual = hidden_states
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(flat.float(), self.gate.weight.float())
        probabilities = torch.sigmoid(router_logits)
        selected = torch.topk(probabilities + self.expert_bias.unsqueeze(0), self.top_k, dim=-1).indices
        weights = probabilities.gather(-1, selected)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(flat.dtype)
        result = torch.zeros_like(flat)
        mask = F.one_hot(selected, self.num_experts).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum((-1, -2)), 0).nonzero().flatten():
            slot, token = torch.where(mask[expert_idx])
            expert_output = self.experts[expert_idx](flat[token]) * weights[token, slot, None]
            result.index_add_(0, token, expert_output.to(flat.dtype))
        coefficient = self.coefficient(residual).softmax(-1)
        result = result.reshape_as(residual) * coefficient[..., :1]
        result = result + self.shared_experts(residual) * coefficient[..., 1:]
        return result, router_logits


class KlearDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KlearAttention(config, layer_idx)
        sparse = config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = KlearSparseMoeBlock(config) if sparse and layer_idx not in config.mlp_only_layers else KlearMLP(config)
        self.input_layernorm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_value=None,
                output_attentions=False, output_router_logits=False, **kwargs):
        residual = hidden_states
        output, weights = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_value=past_key_value, **kwargs,
        )
        hidden_states = residual + output
        residual = hidden_states
        output = self.mlp(self.post_attention_layernorm(hidden_states))
        router_logits = None
        if isinstance(output, tuple):
            output, router_logits = output
        result = (residual + output,)
        if output_attentions:
            result += (weights,)
        if output_router_logits:
            result += (router_logits,)
        return result


class KlearPreTrainedModel(PreTrainedModel):
    config_class = KlearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KlearDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules_strict = ["expert_bias"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, KlearRMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, KlearSparseMoeBlock):
            nn.init.zeros_(module.expert_bias)
        elif isinstance(module, KlearRotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq)


class KlearModel(KlearPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([KlearDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = KlearRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = KlearRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                output_router_logits=None, cache_position=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_router_logits = self.config.output_router_logits if output_router_logits is None else output_router_logits
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError("Klear past_key_values must be a Transformers Cache")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            if cache_position is None:
                seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = cache_position.unsqueeze(0)
        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask[self.config.layer_types[0]]
        else:
            attention_mask = mask_function(
                config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values, position_ids=position_ids,
            )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router = () if output_router_logits else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden += (hidden_states,)
            outputs = layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings,
                position_ids=position_ids, past_key_value=past_key_values, use_cache=use_cache,
                output_attentions=output_attentions, output_router_logits=output_router_logits, **kwargs,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
            if output_router_logits:
                all_router += (outputs[-1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)
        if all_router and any(logits is not None for logits in all_router):
            all_router = tuple(logits for logits in all_router if logits is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values,
            hidden_states=all_hidden, attentions=all_attentions, router_logits=all_router,
        )


class KlearMoeForCausalLM(KlearPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = KlearModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, output_router_logits=None, cache_position=None, logits_to_keep=0, **kwargs):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits, cache_position=cache_position, return_dict=True, **kwargs,
        )
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices])
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs) if labels is not None else None
        return MoeCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions, router_logits=outputs.router_logits,
        )


__all__ = ["KlearMoeForCausalLM", "KlearModel", "KlearPreTrainedModel"]
