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

from copy import copy
from types import SimpleNamespace
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen2.modeling_qwen2 import Qwen2Model
from .configuration_mimo_v2 import MiMoV2Config


logger = logging.get_logger(__name__)


def mimo_mask_config(config):
    # SDPA's mask optimizer can return None for causal prefill. Sink attention
    # uses eager equations even when SDPA was requested and needs that causal
    # mask explicitly; otherwise prefill can attend to future tokens.
    if config._attn_implementation == "sdpa" and (
        getattr(config, "add_full_attention_sink_bias", False)
        or getattr(config, "add_swa_attention_sink_bias", False)
    ):
        config = copy(config)
        config._attn_implementation = "eager"
    return config


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
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
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
        sinks = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
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
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiMoV2MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        # Native MiMo-V2-Flash/DeepSeek-V3 default to one routing group.
        self.n_group = config.n_group or 1
        self.topk_group = config.topk_group or 1
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"Unsupported scoring function for MoE gating: {self.scoring_func}")

        if self.topk_method == "noaux_tc":
            if self.training:
                raise ValueError("MiMoV2 noaux_tc routing is only implemented for inference.")
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"Unsupported TopK function for MoE gating: {self.topk_method}")

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
            [MiMoV2MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = MiMoV2MoEGate(config)

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                final_hidden_states.index_add_(0, token_indices, expert_output * expert_weights.unsqueeze(-1))

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states


class MiMoV2Attention(nn.Module):
    """MiMoV2 attention.

    `projection_layout` only controls how checkpoint weights are named and
    stored: Flash uses separate q/k/v projections, while Pro uses fused qkv.
    The attention computation after projection is shared.
    """

    def __init__(self, config, is_swa: bool, layer_idx: int, projection_layout: str = "split"):
        super().__init__()
        if projection_layout not in {"split", "fused_qkv"}:
            raise ValueError(f"Unsupported MiMoV2 attention projection layout: {projection_layout}")

        self.config = config
        self.layer_idx = layer_idx
        self.is_swa = is_swa
        self.is_causal = True
        self.projection_layout = projection_layout

        default_head_dim = config.hidden_size // config.num_attention_heads
        default_v_head_dim = getattr(config, "v_head_dim", default_head_dim)

        if is_swa:
            self.head_dim = getattr(config, "swa_head_dim", getattr(config, "head_dim", default_head_dim))
            self.v_head_dim = getattr(config, "swa_v_head_dim", default_v_head_dim)
            self.num_attention_heads = getattr(config, "swa_num_attention_heads", config.num_attention_heads)
            self.num_key_value_heads = getattr(config, "swa_num_key_value_heads", config.num_key_value_heads)
        else:
            self.head_dim = getattr(config, "head_dim", default_head_dim)
            self.v_head_dim = getattr(config, "v_head_dim", self.head_dim)
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.rope_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        if self.rope_dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {self.rope_dim} from "
                f"head_dim={self.head_dim} and partial_rotary_factor={getattr(config, 'partial_rotary_factor', 1.0)}"
            )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = getattr(config, "sliding_window", None) if is_swa else None
        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.o_hidden_size = self.num_attention_heads * self.v_head_dim
        self.v_scale = getattr(config, "attention_value_scale", None)
        self.attention_sink_bias = (
            nn.Parameter(torch.zeros(self.num_attention_heads), requires_grad=False)
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
            self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=attention_bias)
            self.k_proj = nn.Linear(config.hidden_size, self.k_size, bias=attention_bias)
            self.v_proj = nn.Linear(config.hidden_size, self.v_size, bias=attention_bias)
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
        position_ids: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        key_rope, key_nope = key_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        query_rope, key_rope = apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_implementation = self.config._attn_implementation
        if attn_implementation is not None and attn_implementation.startswith("paged|"):
            raise ValueError(
                "MiMoV2 does not support paged attention cache. "
                "Please use eager, sdpa, flex_attention, or flash_attention_2."
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_implementation, eager_attention_forward
        )
        if self.attention_sink_bias is not None and attn_implementation == "sdpa":
            logger.warning_once(
                "MiMoV2 attention sink bias is not supported by SDPA; falling back to eager attention for correctness."
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
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]

        if self.projection_layout == "fused_qkv":
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split([self.q_size, self.k_size, self.v_size], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(*input_shape, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(*input_shape, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)
        return self._forward_attention(
            query_states,
            key_states,
            value_states,
            input_shape,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )


class MiMoV2DecoderLayer(GradientCheckpointingLayer):
    attention_projection_layout = "split"

    def __init__(self, config, layer_idx: int, attention_projection_layout: Optional[str] = None):
        super().__init__()
        attention_projection_layout = attention_projection_layout or self.attention_projection_layout
        is_swa_layer = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = "sliding_window_attention" if is_swa_layer else "full_attention"
        self.self_attn = MiMoV2Attention(
            config, is_swa_layer, layer_idx, projection_layout=attention_projection_layout
        )
        self.mlp = (
            MiMoV2MoE(config)
            if getattr(config, "n_routed_experts", None) is not None and config.moe_layer_freq[layer_idx]
            else MiMoV2MLP(config)
        )
        self.input_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
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
        self.is_swa = is_swa
        self.rope_type = config.rope_parameters.get("rope_type", "default")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = copy(config)
        self.config.rope_parameters = copy(getattr(config, "rope_parameters", None) or {})
        if is_swa:
            self.config.rope_theta = getattr(config, "swa_rope_theta", config.rope_theta)
            self.config.head_dim = getattr(config, "swa_head_dim", getattr(config, "head_dim", None))
            if self.config.rope_parameters:
                self.config.rope_parameters["rope_theta"] = self.config.rope_theta
        self.rope_init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        config.standardize_rope_params()
        rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        if dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {dim} from "
                f"head_dim={head_dim} and partial_rotary_factor={partial_rotary_factor}"
            )
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def _as_namespace(config_like):
    if config_like is None:
        return SimpleNamespace()
    if isinstance(config_like, dict):
        return SimpleNamespace(**config_like)
    return config_like


def _parse_maybe_list(value: str | int, length: int) -> list[int]:
    if isinstance(value, str) and "-" in value:
        return [int(x) for x in value.split("-")]
    return [int(value)] * length


def _build_speech_embeddings(config) -> nn.ModuleList:
    audio_channels = getattr(config, "audio_channels")
    input_local_dim = getattr(config, "input_local_dim")
    speech_empty_ids = _parse_maybe_list(getattr(config, "speech_zeroemb_idx"), audio_channels)
    speech_vocab_sizes = _parse_maybe_list(getattr(config, "speech_vocab_size"), audio_channels)
    return nn.ModuleList(
        [
            nn.Embedding(speech_vocab_sizes[i], input_local_dim, padding_idx=speech_empty_ids[i])
            for i in range(audio_channels)
        ]
    )


def _pad_and_group_audio_codes(
    audio_codes: torch.Tensor, audio_channels: int, group_size: int
) -> torch.Tensor:
    """Slice to `audio_channels`, pad to `group_size` boundary, reshape to [G, group_size, C]."""
    if audio_codes.dim() != 2:
        raise ValueError(f"`audio_codes` must be 2D [T, C], got shape={tuple(audio_codes.shape)}")
    audio_codes = audio_codes[:, :audio_channels]
    T = audio_codes.shape[0]
    padded_T = ((T + group_size - 1) // group_size) * group_size
    if padded_T > T:
        audio_codes = torch.cat([audio_codes, audio_codes[-1:].expand(padded_T - T, -1)], dim=0)
    return audio_codes.reshape(padded_T // group_size, group_size, audio_channels)


def _replace_modal_embeddings_inplace(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    token_id: int | None,
    modal_embeds: torch.Tensor | None,
) -> None:
    if token_id is None or modal_embeds is None:
        return

    if modal_embeds.dim() != 2:
        raise ValueError(f"`modal_embeds` must be 2D [N, H], got shape={tuple(modal_embeds.shape)}")

    mask = input_ids.eq(token_id)
    num_slots = int(mask.sum().item())
    if num_slots == 0:
        return

    if modal_embeds.shape[0] != num_slots:
        raise ValueError(
            f"Modal embedding count mismatch for token_id={token_id}: "
            f"found {num_slots} placeholders but got {modal_embeds.shape[0]} embeddings."
        )

    inputs_embeds[mask] = modal_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------


def _rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half_vision(q) * sin)
    k_embed = (k * cos) + (_rotate_half_vision(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class MiMoVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim, self.theta = dim, theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MiMoVisionPatchEmbed(nn.Module):
    def __init__(
        self, patch_size: int = 16, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1280
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class MiMoVisionSwiGLUMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=True)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoVisionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_sinks: bool = False,
        window_size: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.window_size = window_size

        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(dim, qkv_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=True)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads)) if use_sinks else None

    def _build_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.window_size <= 0:
            return None
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        mask = mask.masked_fill((row_idx - col_idx).abs() > self.window_size, float("-inf"))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attn: bool = False,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].view(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim :].view(seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        q_chunks = torch.split(q, lengths.tolist(), dim=0)
        k_chunks = torch.split(k, lengths.tolist(), dim=0)
        v_chunks = torch.split(v, lengths.tolist(), dim=0)

        outputs = []
        for q_c, k_c, v_c in zip(q_chunks, k_chunks, v_chunks):
            q_c = q_c.unsqueeze(0).transpose(1, 2)
            k_c = k_c.unsqueeze(0).transpose(1, 2)
            v_c = v_c.unsqueeze(0).transpose(1, 2)

            if self.num_kv_groups > 1:
                k_c = k_c.repeat_interleave(self.num_kv_groups, dim=1)
                v_c = v_c.repeat_interleave(self.num_kv_groups, dim=1)

            attn_mask = None
            if not full_attn:
                attn_mask = self._build_window_mask(q_c.shape[2], q_c.device, q_c.dtype)

            if self.sinks is not None:
                sink_bias = torch.zeros(
                    1, self.num_heads, q_c.shape[2], k_c.shape[2], device=q_c.device, dtype=q_c.dtype
                )
                sink_bias[..., 0] = self.sinks.view(1, self.num_heads, 1)
                attn_mask = sink_bias if attn_mask is None else attn_mask + sink_bias

            attn_out = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=attn_mask, scale=self.scaling)
            outputs.append(attn_out.squeeze(0).transpose(0, 1))

        attn_output = torch.cat(outputs, dim=0)
        attn_output = attn_output.reshape(seq_len, -1)
        return self.proj(attn_output)


class MiMoVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        use_sinks: bool = False,
        window_size: int = -1,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=rms_norm_eps)
        self.attn = MiMoVisionAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            use_sinks=use_sinks, window_size=window_size,
        )
        self.mlp = MiMoVisionSwiGLUMLP(dim=dim, intermediate_dim=intermediate_dim, hidden_act=hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        full_attn: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings, full_attn=full_attn,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MiMoVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class MiMoVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "qk_channels", 64)
        spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.fullatt_block_indexes = getattr(config, "fullatt_block_indexes", [])
        use_sink = getattr(config, "use_sink", False)
        visual_token_window_size = getattr(config, "visual_token_window_size", -1)
        self.vit_window_attn_types = getattr(config, "vit_window_attn_types", None) or [-1] * depth

        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size

        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=getattr(config, "in_channels", None) or getattr(config, "in_chans", 3),
            embed_dim=hidden_size,
        )

        self.rotary_pos_emb = MiMoVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    dim=hidden_size,
                    intermediate_dim=config.intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_act=config.hidden_act,
                    rms_norm_eps=rms_norm_eps,
                    use_sinks=use_sink and (i not in self.fullatt_block_indexes),
                    window_size=visual_token_window_size,
                )
                for i in range(depth)
            ]
        )

        self.merger = MiMoVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        return tensor.flatten(0, 1)

    def get_window_index_1d(self, grid_thw: torch.Tensor, col: bool = True) -> torch.Tensor:
        window_index = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            index_new = index.transpose(1, 2).reshape(-1) if col else index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        return torch.cat(window_index, dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(device=self.patch_embed.proj.weight.device, dtype=self.dtype)
        x = self.patch_embed(x)

        rotary_emb = self.rot_pos_emb(grid_thw)
        rotary_emb = rotary_emb.to(device=x.device)
        emb = torch.cat((rotary_emb, rotary_emb), dim=-1)

        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(device=x.device)
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col).to(device=x.device)

        row_based_embeddings = (emb.cos(), emb.sin())
        col_emb = self.apply_index(emb, window_index_1d_col)
        col_based_embeddings = (col_emb.cos(), col_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(device=x.device)

        for i, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[i]

            if window_attn_type == 1 and (i == 0 or self.vit_window_attn_types[i - 1] != 1):
                x = self.apply_index(x, window_index_1d_col)

            if i > 0 and window_attn_type != 1 and self.vit_window_attn_types[i - 1] == 1:
                x = self.apply_index(x, reverse_window_index_1d_col)

            position_embeddings = col_based_embeddings if window_attn_type == 1 else row_based_embeddings
            full_attn = i in self.fullatt_block_indexes
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, full_attn=full_attn)

        return self.merger(x)


# ---------------------------------------------------------------------------
# Audio encoder
# ---------------------------------------------------------------------------


class AudioProjection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MiMoAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.audio_channels = getattr(config, "audio_channels")
        self.group_size = getattr(config, "group_size")
        self.input_local_dim = getattr(config, "input_local_dim")
        self.out_hidden_size = getattr(config, "out_hidden_size")
        self.input_full_attention = getattr(config, "input_full_attention", True)

        input_local_config = Qwen2Config(
            hidden_size=getattr(config, "input_local_dim"),
            num_hidden_layers=getattr(config, "input_local_layers"),
            num_attention_heads=getattr(config, "input_local_attn_heads"),
            num_key_value_heads=getattr(config, "input_local_attn_heads"),
            intermediate_size=getattr(config, "input_local_intermediate_size"),
            attention_dropout=getattr(config, "input_local_hidden_dropout", 0.0),
            rope_theta=getattr(config, "rope_theta", 640000.0),
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
        )
        self.input_local_transformer = Qwen2Model(input_local_config)

        if not getattr(config, "add_post_norm", True):
            self.input_local_transformer.norm = nn.Identity()

        proj_in = self.input_local_dim * self.group_size
        projection_layers = getattr(config, "projection_layers", 2)
        if projection_layers == 1:
            self.projection = nn.Linear(proj_in, self.out_hidden_size, bias=False)
        elif projection_layers == 2:
            self.projection = AudioProjection(proj_in, proj_in * 4, self.out_hidden_size)
        else:
            raise ValueError(f"Unsupported projection_layers={projection_layers}, expected 1 or 2.")

    def _apply_speech_embeddings(self, audio_codes: torch.Tensor, speech_embeddings: nn.ModuleList) -> torch.Tensor:
        num_segments = audio_codes.shape[0]
        out = torch.zeros(
            (num_segments, self.group_size, self.input_local_dim),
            dtype=speech_embeddings[0].weight.dtype,
            device=audio_codes.device,
        )
        for i in range(self.audio_channels):
            out.add_(speech_embeddings[i](audio_codes[:, :, i].long()))
        return out

    def _apply_input_local_transformer(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        output = self.input_local_transformer(
            inputs_embeds=speech_embeddings, use_cache=False,
            is_causal=not self.input_full_attention,
        )
        return output.last_hidden_state

    def _process_audio_codes(self, audio_codes: torch.Tensor, speech_embeddings: nn.ModuleList) -> torch.Tensor:
        audio_codes = _pad_and_group_audio_codes(audio_codes, self.audio_channels, self.group_size)
        audio_embs = self._apply_speech_embeddings(audio_codes, speech_embeddings)
        audio_hidden = self._apply_input_local_transformer(audio_embs)
        return self.projection(audio_hidden.reshape(audio_hidden.shape[0], -1))

    def forward(
        self,
        speech_embeddings: nn.ModuleList,
        audio_codes: torch.Tensor | None = None,
        audio_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_embeds is not None:
            if audio_embeds.dim() != 2:
                raise ValueError(f"`audio_embeds` must be 2D [N, H], got shape={tuple(audio_embeds.shape)}")
            if audio_embeds.shape[-1] != self.out_hidden_size:
                raise ValueError(
                    f"Unexpected audio_embeds hidden size {audio_embeds.shape[-1]}, expected {self.out_hidden_size}"
                )
            return audio_embeds

        if audio_codes is None:
            raise ValueError("Either `audio_codes` or `audio_embeds` must be provided.")

        return self._process_audio_codes(audio_codes, speech_embeddings)


# ---------------------------------------------------------------------------
# LLM backbone
# ---------------------------------------------------------------------------


class MiMoV2PreTrainedModel(PreTrainedModel):
    _no_split_modules = ["MiMoV2DecoderLayer"]
    _can_record_outputs = {
        "hidden_states": MiMoV2DecoderLayer,
        "attentions": MiMoV2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # HF does not recognize custom direct Parameters as affine children.
        # Checkpoint assignment overwrites these defaults, while tiny models and
        # missing-key initialization must not route using uninitialized memory.
        if isinstance(module, MiMoV2MoEGate):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "e_score_correction_bias"):
                init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiMoV2Attention) and module.attention_sink_bias is not None:
            init.zeros_(module.attention_sink_bias)
        elif isinstance(module, MiMoVisionAttention) and module.sinks is not None:
            init.zeros_(module.sinks)
        elif isinstance(module, MiMoVisionRotaryEmbedding):
            inv = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float32,
                                                     device=module.inv_freq.device) / module.dim))
            init.copy_(module.inv_freq, inv)


class MiMoV2Model(MiMoV2PreTrainedModel):
    base_model_prefix = "model"
    _supports_sdpa = True
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
        self.has_sliding_layers = any(pattern == 1 for pattern in config.hybrid_layer_pattern)
        self.config.layer_types = [
            "sliding_attention" if config.hybrid_layer_pattern[i] == 1 else "full_attention"
            for i in range(config.num_hidden_layers)
        ]
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": mimo_mask_config(self.config),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                if getattr(self.config, "sliding_window", None) is None:
                    raise ValueError("MiMoV2 config `sliding_window` must be set when hybrid_layer_pattern uses SWA.")
                causal_mask_mapping["sliding_window_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

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
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MiMoV2ForCausalLM(MiMoV2PreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _supports_sdpa = True
    config_class = MiMoV2Config
    model_class = MiMoV2Model
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [
        r"model\.(swa_)?rotary_emb\.inv_freq",
        r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq",
        r"model\.layers\.\d+\.self_attn\.rotary_emb\.(cos_cached|sin_cached)",
        r"model\.mtp\..*",
    ]
    _keys_to_ignore_on_load_missing = [
        r"audio_encoder\.input_local_transformer\.embed_tokens\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = self.model_class(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.vision_config:
            self.visual = MiMoVisionTransformer(_as_namespace(config.vision_config))
        if config.audio_config:
            audio_cfg = _as_namespace(config.audio_config)
            self.speech_embeddings = _build_speech_embeddings(audio_cfg)
            self.audio_encoder = MiMoAudioEncoder(audio_cfg)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _get_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        video_pixel_values: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_image = image_embeds is not None or pixel_values is not None
        has_video = video_embeds is not None or video_pixel_values is not None
        has_audio = audio_embeds is not None or audio_codes is not None

        if not (has_image or has_video or has_audio):
            return inputs_embeds

        inputs_embeds = inputs_embeds.clone()

        if has_image:
            cur_image_embeds = image_embeds if image_embeds is not None else self.visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
            _replace_modal_embeddings_inplace(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                token_id=getattr(self.config, "image_token_id", None), modal_embeds=cur_image_embeds,
            )

        if has_video:
            cur_video_embeds = video_embeds if video_embeds is not None else self.visual(pixel_values=video_pixel_values, grid_thw=video_grid_thw)
            _replace_modal_embeddings_inplace(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                token_id=getattr(self.config, "video_token_id", None), modal_embeds=cur_video_embeds,
            )

        if has_audio:
            _replace_modal_embeddings_inplace(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                token_id=getattr(self.config, "audio_token_id", None),
                modal_embeds=self.audio_encoder(
                    speech_embeddings=self.speech_embeddings, audio_codes=audio_codes, audio_embeds=audio_embeds,
                ),
            )

        return inputs_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        video_pixel_values: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if any(x is not None for x in [pixel_values, image_embeds, video_pixel_values, video_embeds, audio_codes, audio_embeds]):
                inputs_embeds = self._get_multimodal_embeds(
                    input_ids=input_ids, inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw, image_embeds=image_embeds,
                    video_pixel_values=video_pixel_values, video_grid_thw=video_grid_thw, video_embeds=video_embeds,
                    audio_codes=audio_codes, audio_embeds=audio_embeds,
                )
            input_ids = None

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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


__all__ = [
    "MiMoV2Attention",
    "MiMoV2DecoderLayer",
    "MiMoV2ForCausalLM",
    "MiMoV2MLP",
    "MiMoV2MoE",
    "MiMoV2MoEGate",
    "MiMoV2Model",
    "MiMoV2RMSNorm",
    "MiMoV2RotaryEmbedding",
]
