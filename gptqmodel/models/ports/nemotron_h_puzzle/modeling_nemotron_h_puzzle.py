# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1 AND Apache-2.0
"""Puzzle decoder with explicit heterogeneous construction; no process-global patches.

The Mamba-2 primitive and RMS norms are native Transformers 5. Puzzle's routing,
expert ownership, no-position-embedding attention and per-layer geometry remain local.
"""

from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.nemotron_h.modeling_nemotron_h import (
    NemotronHMamba2Mixer,
    NemotronHPreTrainedModel,
    NemotronHRMSNorm,
)
from .configuration_nemotron_h_puzzle import NemotronHPuzzleConfig


class NemotronHPuzzleCache(DynamicCache):
    """Hybrid attention/Mamba cache whose length is defined even for attention-free Puzzle stacks.

    Recurrent (linear-attention) cache layers carry no sequence axis, and a Puzzle checkpoint may have no
    attention block at all, so the model records how many tokens each successful forward appended.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._processed_tokens = 0

    def get_seq_length(self, layer_idx=0):
        return self._processed_tokens

    def advance(self, num_tokens):
        self._processed_tokens += num_tokens

    def reset(self):
        super().reset()
        self._processed_tokens = 0

    def crop(self, tokens_to_remove):
        raise ValueError("Puzzle recurrent cache cannot be cropped for speculative decoding")


class NemotronHMLP(nn.Module):
    def __init__(self, config, intermediate_size, is_expert=False):
        super().__init__()
        width = config.moe_latent_size if is_expert and config.moe_latent_size is not None else config.hidden_size
        self.up_proj = nn.Linear(width, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, width, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class NemotronHTopkRouter(nn.Module):
    e_score_correction_bias: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        c = self.config
        scores = F.linear(hidden_states.reshape(-1, c.hidden_size).float(), self.weight.float()).sigmoid()
        choices = scores + self.e_score_correction_bias.float()
        group_scores = choices.view(-1, c.n_group, c.n_routed_experts // c.n_group).topk(2, dim=-1).values.sum(-1)
        groups = group_scores.topk(c.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, groups, 1)
        mask = group_mask.unsqueeze(-1).expand(-1, c.n_group, c.n_routed_experts // c.n_group).reshape_as(choices)
        indices = choices.masked_fill(~mask.bool(), 0.0).topk(c.num_experts_per_tok, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * c.routed_scaling_factor


class NemotronHMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = NemotronHTopkRouter(config)
        self.experts = nn.ModuleList([
            NemotronHMLP(config, config.moe_intermediate_size, is_expert=True) for _ in range(config.n_routed_experts)
        ])
        self.shared_experts = NemotronHMLP(config, config.moe_shared_expert_intermediate_size)
        self.fc1_latent_proj = nn.Identity()
        self.fc2_latent_proj = nn.Identity()
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)

    def moe(self, hidden_states, topk_indices, topk_weights):
        output = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        for i, expert in enumerate(self.experts):
            tokens, slots = torch.where(topk_indices == i)
            if tokens.numel():
                weighted = expert(hidden_states[tokens]) * topk_weights[tokens, slots, None]
                output.index_add_(0, tokens, weighted)
            else:
                # Exercise unsampled experts for calibration without adding their biases.
                output = output + expert(torch.zeros_like(hidden_states[:1])).sum(0, keepdim=True) * 0
        return output.to(hidden_states.dtype)

    def forward(self, hidden_states):
        indices, weights = self.gate(hidden_states)
        routed = self.fc1_latent_proj(hidden_states.reshape(-1, hidden_states.shape[-1]))
        routed = self.fc2_latent_proj(self.moe(routed, indices, weights)).reshape_as(hidden_states)
        return routed + self.shared_experts(hidden_states)


class NemotronHPuzzleAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if attention_mask is None:
            attention_mask = _causal_mask(None, hidden_states, k.shape[-2] - length)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, _ = attention_interface(
            self, q, k, v, attention_mask,
            dropout=self.config.attention_dropout if self.training else 0.0, scaling=self.scaling, **kwargs,
        )
        return self.o_proj(output.reshape(batch, length, -1).contiguous())


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, length, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, length, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, length, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = weights.float().softmax(-1).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


def _causal_mask(mask, hidden_states, past_length):
    batch, length = hidden_states.shape[:2]
    total = past_length + length
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise ValueError("Puzzle expects a tensor padding mask or additive causal mask")
    if mask is not None and mask.ndim == 4:
        if mask.shape[-2:] != (length, total) or mask.shape[0] not in (1, batch) or not mask.is_floating_point():
            raise ValueError("Puzzle expects an additive 4D causal mask matching query and KV lengths")
        return mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    positions = torch.arange(total, device=hidden_states.device)
    query = torch.arange(past_length, total, device=hidden_states.device)
    blocked = positions[None, :] > query[:, None]
    blocked = blocked[None, None].expand(batch, 1, -1, -1)
    if mask is not None:
        if mask.ndim != 2 or mask.shape != (batch, total):
            raise ValueError("Puzzle padding mask must match batch and full cached sequence length")
        blocked = blocked | ~mask.to(device=hidden_states.device, dtype=torch.bool)[:, None, None, :]
    return torch.zeros(blocked.shape, dtype=hidden_states.dtype, device=hidden_states.device).masked_fill(
        blocked, torch.finfo(hidden_states.dtype).min
    )


class NemotronHPuzzleBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config.get_nemotron_h_config_for_layer(layer_idx)
        self.block_type = config.block_configs[layer_idx]["block_type"]
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.residual_in_fp32 = config.residual_in_fp32
        if self.block_type == "mamba":
            self.mixer = NemotronHMamba2Mixer(self.config, layer_idx)
        elif self.block_type == "attention":
            self.mixer = NemotronHPuzzleAttention(self.config, layer_idx)
        else:
            self.mixer = NemotronHMoE(self.config)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, **kwargs):
        residual = hidden_states.float() if self.residual_in_fp32 else hidden_states
        hidden_states = self.norm(hidden_states.to(self.norm.weight.dtype))
        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states, past_key_values=past_key_values, attention_mask=attention_mask)
        else:
            hidden_states = self.mixer(hidden_states)
        return residual + hidden_states


class NemotronHPuzzlePreTrainedModel(NemotronHPreTrainedModel):
    config_class = NemotronHPuzzleConfig
    _no_split_modules = ["NemotronHPuzzleBlock"]
    _supports_flash_attn = False
    _supports_flex_attn = False
    supports_gradient_checkpointing = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": NemotronHPuzzleBlock}
    _checkpoint_conversion_mapping = {}
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, NemotronHTopkRouter):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)


class NemotronHPuzzleModel(NemotronHPuzzlePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHPuzzleBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None,
                use_cache=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if kwargs.get("output_attentions"):
            raise ValueError("Puzzle attention weights are not exposed by this port")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if past_key_values is not None and not isinstance(past_key_values, NemotronHPuzzleCache):
            raise ValueError("Puzzle requires its own dynamic hybrid cache")
        if not use_cache and past_key_values is not None:
            raise ValueError("Puzzle past_key_values requires use_cache=True")
        if use_cache and past_key_values is None:
            past_key_values = NemotronHPuzzleCache(self.config)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        causal = _causal_mask(attention_mask, inputs_embeds, past_length)
        recurrent_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            recurrent_mask = attention_mask[:, -inputs_embeds.shape[1]:].to(inputs_embeds.device)
        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_mask = causal if layer.block_type == "attention" else recurrent_mask
            hidden_states = layer(hidden_states, past_key_values=past_key_values, attention_mask=layer_mask)
        hidden_states = self.norm_f(hidden_states)
        if past_key_values is not None:
            past_key_values.advance(inputs_embeds.shape[1])
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class NemotronHPuzzleForCausalLM(NemotronHPuzzlePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = NemotronHPuzzleModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def _prepare_cache_for_generation(self, generation_config, *args, **kwargs):
        if generation_config.cache_implementation is not None:
            raise ValueError("Puzzle supports only its default dynamic hybrid cache")
        return cast(Any, super())._prepare_cache_for_generation(generation_config, *args, **kwargs)

    def create_masks_for_generate(self, attention_mask, **kwargs):
        # Forward builds both the additive attention and recurrent padding masks.
        return attention_mask

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, past_key_values=None,
                labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        hidden_states = outputs.last_hidden_state
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, indices].to(self.lm_head.weight.dtype)).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(logits=logits, loss=loss, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states)
