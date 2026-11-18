# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
"""Package-local DeepSeek-VL2 text decoder on Transformers 5 cache/mask/attention APIs.

Weights, MLA latent caching and the gate/expert math follow the checkpoint's
remote code; the RMSNorm and eager attention kernel are the native TF5 ones.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RMSNorm
from transformers.models.llama.modeling_llama import eager_attention_forward, rotate_half
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_deepseek import DeepseekV2Config


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity
    return torch.clamp((torch.arange(dim, dtype=torch.float32) - min) / (max - min), 0, 1)


class DeepseekV2RotaryEmbedding(nn.Module):
    """DeepSeek's legacy ``rope_scaling`` (linear/dynamic/yarn) RoPE.

    Frequencies are computed per call, so dynamic scaling never mutates shared state.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.qk_rope_head_dim if config.use_mla else config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.scaling = dict(config.rope_parameters or {})

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len):
        scaling = self.scaling
        kind = scaling.get("rope_type", scaling.get("type", "default"))
        factor = scaling.get("factor", 1.0)
        base = self.base
        if kind == "dynamic" and seq_len > self.max_position_embeddings:
            base *= (factor * seq_len / self.max_position_embeddings - (factor - 1)) ** (self.dim / (self.dim - 2))
        inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        amplitude = 1.0
        if kind == "linear":
            inv = inv / factor
        elif kind == "yarn":
            low, high = yarn_find_correction_range(
                scaling.get("beta_fast", 32),
                scaling.get("beta_slow", 1),
                self.dim,
                base,
                scaling.get("original_max_position_embeddings", 4096),
            )
            mask = 1.0 - yarn_linear_ramp_mask(low, high, self.dim // 2).to(x.device)
            inv = inv / factor * (1 - mask) + inv * mask
            amplitude = yarn_get_mscale(factor, scaling.get("mscale", 1)) / yarn_get_mscale(
                factor, scaling.get("mscale_all_dim", 0)
            )
        elif kind not in ("default", "dynamic"):
            raise ValueError(f"Unknown RoPE scaling type {kind}")
        freqs = position_ids[..., None].float() * inv
        emb = torch.cat((freqs, freqs), -1)
        return (emb.cos() * amplitude).to(x.dtype), (emb.sin() * amplitude).to(x.dtype)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """MLA RoPE: de-interleave the rope dims, then rotate-half (DeepSeek remote-code layout)."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _group_mask(self, group_scores, tokens):
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        return (
            group_mask.unsqueeze(-1)
            .expand(tokens, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(tokens, -1)
            .bool()
        )

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(tokens, self.n_group, -1).max(dim=-1).values
            tmp_scores = scores.masked_fill(~self._group_mask(group_scores, tokens), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "noaux_tc":
            if self.training:
                raise NotImplementedError("noaux_tc routing is inference-only")
            scores_for_choice = scores.view(tokens, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = scores_for_choice.view(tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            tmp_scores = scores_for_choice.masked_fill(~self._group_mask(group_scores, tokens), 0.0)
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        aux_loss = None
        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device)
                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                aux_loss = (scores.mean(0) * mask_ce.float().mean(0) * self.n_routed_experts).sum() * self.alpha
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """Identity on ``x`` that back-propagates a unit gradient into the auxiliary loss."""

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekV2MoE(nn.Module):
    """Routed experts plus optional shared experts."""

    def __init__(self, config):
        super().__init__()
        if getattr(config, "ep_size", 1) > 1:
            raise NotImplementedError("Expert parallelism (ep_size > 1) is not supported by the local VL2 port")
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = MoEGate(config)
        self.shared_experts = None
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.training:
            flat_topk_idx = topk_idx.view(-1)
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            if aux_loss is not None:
                y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        tokens_per_expert = torch.bincount(topk_ids.view(-1), minlength=len(self.experts)).tolist()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for expert, num_tokens in zip(self.experts, tokens_per_expert):
            if num_tokens == 0:
                continue
            outputs.append(expert(sorted_tokens[start_idx : start_idx + num_tokens]))
            start_idx += num_tokens
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class DeepseekV2Attention(nn.Module):
    """Multi-head latent attention; the cache stores the normalized latent and the roped shared key."""

    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_groups = 1  # keys/values are expanded to every head from the latent
        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        self.scaling = self.q_head_dim ** (-0.5)
        rope_parameters = config.rope_parameters or {}
        if mscale_all_dim := rope_parameters.get("mscale_all_dim", 0):
            mscale = yarn_get_mscale(rope_parameters.get("factor", 1.0), mscale_all_dim)
            self.scaling = self.scaling * mscale * mscale

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv, k_pe = torch.split(
            self.kv_a_proj_with_mqa(hidden_states), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv).view(bsz, 1, q_len, self.kv_lora_rank)
        k_pe = k_pe.view(bsz, 1, q_len, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin)
        if past_key_values is not None:
            compressed_kv, k_pe = past_key_values.update(compressed_kv, k_pe, self.layer_idx)

        # Invoke the checkpoint-owned projection (also after quantization) instead of absorbing its weight.
        kv_len = compressed_kv.shape[-2]
        kv = self.kv_b_proj(compressed_kv.squeeze(1))
        kv = kv.view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe.expand(-1, self.num_heads, -1, -1)), dim=-1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class DenseAttention(nn.Module):
    """The VL2-tiny Llama-style GQA layout (not MLA's interleaved RoPE)."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        if self.num_heads % self.kv_heads or self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError("Invalid dense attention head dimensions")
        self.num_key_value_groups = self.num_heads // self.kv_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q, k, v = (proj(hidden_states).view(hidden_shape).transpose(1, 2) for proj in (self.q_proj, self.k_proj, self.v_proj))
        cos, sin = position_embeddings
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask,
            dropout=0.0 if not self.training else self.config.attention_dropout, scaling=self.scaling, **kwargs,
        )
        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights


class DeepseekV2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        attention_class = DeepseekV2Attention if config.use_mla else DenseAttention
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)
        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = DeepseekV2MoE(config) if is_moe else DeepseekV2MLP(config)
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class DeepseekV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False  # data-dependent expert dispatch
    _can_record_outputs = {
        "hidden_states": DeepseekV2DecoderLayer,
        "attentions": [DeepseekV2Attention, DenseAttention],
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MoEGate):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "e_score_correction_bias"):
                init.zeros_(module.e_score_correction_bias)


class DeepseekV2Model(DeepseekV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV2RotaryEmbedding(config)
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
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        length = inputs_embeds.shape[1]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[-1] < past + length:
            raise ValueError("attention_mask must cover cached and current tokens")
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                # Left-padding aware positions, as the checkpoint's generation code derives them.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 1)[:, past : past + length]
            else:
                position_ids = torch.arange(past, past + length, device=inputs_embeds.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        seq_len = max(past + length, int(position_ids.max()) + 1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids, seq_len)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values)


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, logits_to_keep=0, **kwargs):
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
