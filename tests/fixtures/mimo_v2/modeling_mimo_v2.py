# coding=utf-8
#
# Copyright 2026 Xiaomi Corporation.
# Copyright 2026 The HuggingFace Inc. team.
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

"""MiMo reference text blocks for generated-weight integration fixtures.

Adapted from XiaomiMiMo/MiMo-V2.6-Flash-RL revision
3b38d063180c3e4aed9691fdc735f3d10b266ee4; vision/audio runtime omitted.
"""

from copy import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from .configuration_mimo_v2 import MiMoV2Config


logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if sinks is not None:
        sinks = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1
        )
        attn_weights = torch.cat([attn_weights, sinks], dim=-1)

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if sinks is not None:
        probs = probs[..., :-1]

    attn_weights = nn.functional.dropout(probs, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MiMoV2RMSNorm(nn.Module):
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


class MiMoV2MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class MiMoV2MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor
            if config.routed_scaling_factor is not None
            else 1.0
        )
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"Unsupported scoring function for MoE gating: {self.scoring_func}"
            )

        if self.topk_method == "noaux_tc":
            if self.training:
                raise ValueError(
                    "MiMoV2 noaux_tc routing is only implemented for inference."
                )
            scores_for_choice = scores.view(
                bsz * seq_len, -1
            ) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"Unsupported TopK function for MoE gating: {self.topk_method}"
            )

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class MiMoV2MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                MiMoV2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MiMoV2MoEGate(config)

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                final_hidden_states.index_add_(
                    0, token_indices, expert_output * expert_weights.unsqueeze(-1)
                )

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(
            *orig_shape
        )
        return hidden_states


class MiMoV2Attention(nn.Module):
    """MiMoV2 attention.

    `projection_layout` only controls how checkpoint weights are named and
    stored: Flash uses separate q/k/v projections, while Pro uses fused qkv.
    The attention computation after projection is shared.
    """

    def __init__(
        self, config, is_swa: bool, layer_idx: int, projection_layout: str = "split"
    ):
        super().__init__()
        if projection_layout not in {"split", "fused_qkv"}:
            raise ValueError(
                f"Unsupported MiMoV2 attention projection layout: {projection_layout}"
            )

        self.config = config
        self.layer_idx = layer_idx
        self.is_swa = is_swa
        self.is_causal = True
        self.projection_layout = projection_layout

        default_head_dim = config.hidden_size // config.num_attention_heads
        default_v_head_dim = getattr(config, "v_head_dim", default_head_dim)

        if is_swa:
            self.head_dim = getattr(
                config, "swa_head_dim", getattr(config, "head_dim", default_head_dim)
            )
            self.v_head_dim = getattr(config, "swa_v_head_dim", default_v_head_dim)
            self.num_attention_heads = getattr(
                config, "swa_num_attention_heads", config.num_attention_heads
            )
            self.num_key_value_heads = getattr(
                config, "swa_num_key_value_heads", config.num_key_value_heads
            )
        else:
            self.head_dim = getattr(config, "head_dim", default_head_dim)
            self.v_head_dim = getattr(config, "v_head_dim", self.head_dim)
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.rope_dim = int(
            self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        if self.rope_dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {self.rope_dim} from "
                f"head_dim={self.head_dim} and "
                f"partial_rotary_factor={getattr(config, 'partial_rotary_factor', 1.0)}"
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = (
            getattr(config, "sliding_window", None) if is_swa else None
        )
        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.o_hidden_size = self.num_attention_heads * self.v_head_dim
        self.v_scale = getattr(config, "attention_value_scale", None)
        self.attention_sink_bias = (
            nn.Parameter(torch.empty(self.num_attention_heads), requires_grad=False)
            if (
                (getattr(config, "add_full_attention_sink_bias", False) and not is_swa)
                or (getattr(config, "add_swa_attention_sink_bias", False) and is_swa)
            )
            else None
        )

        attention_bias = getattr(config, "attention_bias", False)
        if self.projection_layout == "fused_qkv":
            self.qkv_proj = nn.Linear(
                config.hidden_size,
                self.q_size + self.k_size + self.v_size,
                bias=attention_bias,
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, self.q_size, bias=attention_bias
            )
            self.k_proj = nn.Linear(
                config.hidden_size, self.k_size, bias=attention_bias
            )
            self.v_proj = nn.Linear(
                config.hidden_size, self.v_size, bias=attention_bias
            )
        self.o_proj = nn.Linear(self.o_hidden_size, config.hidden_size, bias=False)

    def _forward_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_shape: torch.Size,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split(
            [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        key_rope, key_nope = key_states.split(
            [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        query_rope, key_rope = apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_implementation = self.config._attn_implementation
        if attn_implementation is not None and attn_implementation.startswith("paged|"):
            raise ValueError(
                "MiMoV2 remote code does not support paged attention cache. "
                "Please use eager, sdpa, flex_attention, or flash_attention_2."
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_implementation, eager_attention_forward
        )
        if self.attention_sink_bias is not None and attn_implementation == "sdpa":
            logger.warning_once(
                "MiMoV2 attention sink bias is not supported by SDPA; "
                "falling back to eager attention for correctness."
            )
            attention_interface = eager_attention_forward

        attention_kwargs = {
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "scaling": self.scaling,
            "position_ids": position_ids,
            "is_causal": self.is_causal,
        }
        if attention_interface is eager_attention_forward:
            attention_kwargs["sinks"] = self.attention_sink_bias
        else:
            if self.attention_sink_bias is not None:
                attention_kwargs["s_aux"] = self.attention_sink_bias
            if self.sliding_window is not None:
                attention_kwargs["sliding_window"] = self.sliding_window

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            **attention_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]

        if self.projection_layout == "fused_qkv":
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split(
                [self.q_size, self.k_size, self.v_size], dim=-1
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            *input_shape, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            *input_shape, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            *input_shape, self.num_key_value_heads, self.v_head_dim
        ).transpose(1, 2)
        return self._forward_attention(
            query_states,
            key_states,
            value_states,
            input_shape,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
        )


class MiMoV2DecoderLayer(nn.Module):
    attention_projection_layout = "split"

    def __init__(
        self, config, layer_idx: int, attention_projection_layout: Optional[str] = None
    ):
        super().__init__()
        attention_projection_layout = (
            attention_projection_layout or self.attention_projection_layout
        )
        is_swa_layer = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = (
            "sliding_window_attention" if is_swa_layer else "full_attention"
        )
        self.self_attn = MiMoV2Attention(
            config,
            is_swa_layer,
            layer_idx,
            projection_layout=attention_projection_layout,
        )
        self.mlp = (
            MiMoV2MoE(config)
            if getattr(config, "n_routed_experts", None) is not None
            and config.moe_layer_freq[layer_idx]
            else MiMoV2MLP(config)
        )
        self.input_layernorm = MiMoV2RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = MiMoV2RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiMoV2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, is_swa: bool, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = copy(config)
        self.config.rope_parameters = copy(
            getattr(config, "rope_parameters", None) or {}
        )
        if is_swa:
            self.config.rope_theta = getattr(
                config, "swa_rope_theta", config.rope_theta
            )
            self.config.head_dim = getattr(
                config, "swa_head_dim", getattr(config, "head_dim", None)
            )
            if self.config.rope_parameters:
                self.config.rope_parameters["rope_theta"] = self.config.rope_theta
        self.rope_init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config, device=None, seq_len=None, layer_type=None
    ):
        config.standardize_rope_params()
        rope_parameters = (
            config.rope_parameters[layer_type]
            if layer_type is not None
            else config.rope_parameters
        )
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        if dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {dim} from "
                f"head_dim={head_dim} and partial_rotary_factor={partial_rotary_factor}"
            )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiMoV2Model(PreTrainedModel):
    config_class = MiMoV2Config
    attention_projection_layout = "split"

    def __init__(self, config):
        super().__init__(config)
        self.attention_projection_layout = getattr(
            config, "attention_projection_layout", self.attention_projection_layout
        )
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MiMoV2DecoderLayer(
                    config,
                    layer_idx,
                    attention_projection_layout=self.attention_projection_layout,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=False)
        self.swa_rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=True)
        self.has_sliding_layers = any(
            pattern == 1 for pattern in config.hybrid_layer_pattern
        )
        self.config.layer_types = [
            "sliding_attention"
            if config.hybrid_layer_pattern[i] == 1
            else "full_attention"
            for i in range(config.num_hidden_layers)
        ]
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                if getattr(self.config, "sliding_window", None) is None:
                    raise ValueError(
                        "MiMoV2 config `sliding_window` must be set "
                        "when hybrid_layer_pattern uses SWA."
                    )
                causal_mask_mapping["sliding_window_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings
                if decoder_layer.attention_type == "full_attention"
                else swa_position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MiMoV2ForCausalLM(PreTrainedModel, GenerationMixin):
    """Small test wrapper; modality modules only exercise tensor preservation."""

    config_class = MiMoV2Config
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]
    _supports_sdpa = True

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__(config)
        self.model = MiMoV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.visual = nn.Linear(16, 16, bias=False)
        self.audio_encoder = nn.Linear(16, 16, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return CausalLMOutputWithPast(
            logits=self.lm_head(outputs.last_hidden_state),
            past_key_values=outputs.past_key_values,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value
