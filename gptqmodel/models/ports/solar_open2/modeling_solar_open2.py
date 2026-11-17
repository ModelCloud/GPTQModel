# Modified by ModelCloud: package-local eager KDA, released expert layout and Transformers 5.23 cache API.
# Copyright 2026 Upstage AI and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_solar_open2 import SolarOpen2Config


class SolarOpen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        SolarOpen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SolarOpen2RMSNormGated(nn.Module):
    """Gated RMSNorm used at the KDA output: ``RMSNorm(x) * sigmoid(gate)``.

    Equivalent to ``fla.modules.FusedRMSNormGated(..., activation="sigmoid")``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight.to(torch.float32)
        hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for the KDA log-decay gate.

    ``g = -exp(A_log) * softplus(g_raw + dt_bias)``, a per-(head, dim) log-decay (<= 0). ``A_log`` is per-head and
    ``dt_bias`` is per-(head, dim). Mirrors ``fla.ops.kda.gate.fused_kda_gate``.

    Args:
        g (`torch.Tensor`): raw gate of shape `(..., num_heads, head_dim)`.
        A_log (`torch.Tensor`): per-head log scale, broadcastable to `num_heads`.
        dt_bias (`torch.Tensor`): per-(head, dim) bias of shape `(num_heads * head_dim,)`.
        lower_bound (`float`, *optional*): clamps the gate from below when set.
    """
    num_heads = A_log.numel()
    head_dim = dt_bias.numel() // num_heads
    g = g.to(torch.float32)
    g = g + dt_bias.to(torch.float32).reshape(num_heads, head_dim)
    g = -A_log.to(torch.float32).reshape(num_heads, 1).exp() * F.softplus(g)
    if lower_bound is not None:
        g = g.clamp(min=lower_bound)
    return g


def torch_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Pure-PyTorch reference for the Kimi-Delta gated delta rule.

    Differs from the Qwen3-Next gated delta rule by a per-(head, dim) **vector** decay ``g`` (GLA-style), applied
    along the key axis of the recurrent state. Used for both prefill and decode in the fallback path.

    Shapes: ``query/key/value/g`` are `(batch, seq_len, num_heads, head_dim)`, ``beta`` is `(batch, seq_len, num_heads)`.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, g))
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (k_head_dim**0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device)
    state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        # per-key-dim decay (vector gate): each row k of the state decays by exp(g[k]).
        g_t = g[:, :, i].exp().unsqueeze(-1)  # (batch, num_heads, k_head_dim, 1)
        beta_t = beta[:, :, i].unsqueeze(-1)  # (batch, num_heads, 1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (batch, num_heads, v_head_dim)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, state


class SolarOpen2LinearAttention(nn.Module):
    """Kimi-Delta-Attention (gated delta-net) linear-attention layer."""

    def __init__(self, config: SolarOpen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_kv_heads = config.linear_attn_config.get("num_kv_heads", None) or self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.use_full_proj = config.kda_use_full_proj
        self.gate_lower_bound = config.kda_gate_lower_bound
        self.allow_neg_eigval = config.kda_allow_neg_eigval

        projection_size = self.head_dim * self.num_heads
        kv_projection_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_projection_size, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # depthwise causal short convolutions (weight `[channels, 1, kernel]`), applied with silu, no bias.
        self.q_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            self.conv_size,
            groups=projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            kv_projection_size,
            kv_projection_size,
            self.conv_size,
            groups=kv_projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            kv_projection_size,
            kv_projection_size,
            self.conv_size,
            groups=kv_projection_size,
            padding=self.conv_size - 1,
            bias=False,
        )

        if self.use_full_proj:
            self.f_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.empty(projection_size))
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1))

        self.o_norm = SolarOpen2RMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def _conv_weight(self) -> torch.Tensor:
        # concatenate the per-projection depthwise conv weights so q/k/v share a single grouped conv / conv-state.
        return torch.cat([self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        recurrent_state = None
        if use_precomputed_states:
            recurrent_state = past_key_values.layers[self.layer_idx].recurrent_states[0]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        mixed = torch.cat((q, k, v), dim=-1).transpose(1, 2)  # (batch, 3 * projection, seq_len)
        conv_weight = self._conv_weight()

        if past_key_values is not None:
            mixed = past_key_values.update_conv_state(
                mixed,
                self.layer_idx,
                conv_kernel_size=self.conv_size,
            )
        mixed = F.silu(
            F.conv1d(
                mixed,
                conv_weight,
                bias=None,
                padding=self.conv_size - 1,
                groups=conv_weight.shape[0],
            )[..., : mixed.shape[-1]]
        )[..., -seq_len:]

        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(
            mixed, [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features], dim=-1
        )
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        g_raw = self.f_proj(hidden_states) if self.use_full_proj else self.f_b_proj(self.f_a_proj(hidden_states))
        g_raw = g_raw.reshape(batch_size, -1, self.num_heads, self.head_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = torch_kda_gate(g_raw, self.A_log, self.dt_bias, self.gate_lower_bound)
        core_attn_out, recurrent_state = torch_recurrent_kda(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=past_key_values is not None,
            use_qk_l2norm_in_kernel=True,
        )

        if past_key_values is not None:
            past_key_values.update_recurrent_state(recurrent_state, self.layer_idx)

        g_out = self.g_proj(hidden_states) if self.use_full_proj else self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = g_out.reshape(batch_size, -1, self.num_heads, self.head_dim)
        core_attn_out = self.o_norm(core_attn_out, g_out)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.o_proj(core_attn_out), None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
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
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class SolarOpen2Attention(nn.Module):
    """Grouped-query full attention with optional NoPE, qk-norm, and an output gate."""

    def __init__(self, config: SolarOpen2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = config.rope_parameters
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = SolarOpen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = SolarOpen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.use_rope = config.use_rope
        self.use_gqa_gate = config.use_gqa_gate
        if self.use_gqa_gate:
            self.g_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_gqa_gate_bias
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.use_gqa_gate:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states))
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SolarOpen2MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SolarOpen2TopkRouter(nn.Module):
    def __init__(self, config: SolarOpen2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.register_buffer("e_score_correction_bias", torch.zeros((self.num_experts), dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


class SolarOpen2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: SolarOpen2Config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                SolarOpen2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = SolarOpen2TopkRouter(config)
        self.shared_experts = (
            SolarOpen2MLP(config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
            if config.n_shared_experts
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        combined = torch.zeros_like(hidden_states)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, top_k_pos = torch.where(topk_indices == expert_idx)
            if token_idx.numel():
                values = expert(hidden_states[token_idx])
                values = values * topk_weights[token_idx, top_k_pos, None]
                combined.index_add_(0, token_idx, values.to(combined.dtype))
        hidden_states = combined.view(*orig_shape)
        if self.shared_experts is not None:
            hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class SolarOpen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: SolarOpen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: SolarOpen2Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SolarOpen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SolarOpen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear_attn = self.layer_type == "linear_attention"

        if self.is_linear_attn:
            self.self_attn = SolarOpen2LinearAttention(config, layer_idx)
        else:
            self.self_attn = SolarOpen2Attention(config, layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = SolarOpen2MoE(config)
        else:
            self.mlp = SolarOpen2MLP(config)

        self.input_layernorm = SolarOpen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SolarOpen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SolarOpen2PreTrainedModel(PreTrainedModel):
    config_class = SolarOpen2Config
    config: SolarOpen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SolarOpen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": SolarOpen2DecoderLayer,
        "attentions": SolarOpen2Attention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = None
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SolarOpen2TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        if isinstance(module, SolarOpen2LinearAttention):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(1, 16).log_())
        elif isinstance(module, SolarOpen2RMSNormGated):
            init.ones_(module.weight)


class SolarOpen2DynamicCache(DynamicCache):
    """Per-generation hybrid state; sequence length also exists for all-KDA layouts."""

    def __init__(self, config):
        super().__init__(config=config)
        self._solar_seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        return self._solar_seen_tokens

    def reset(self):
        super().reset()
        self._solar_seen_tokens = 0

    def crop(self, *args, **kwargs):
        raise NotImplementedError("Solar-Open2 recurrent state cannot be rolled back")


class SolarOpen2Model(SolarOpen2PreTrainedModel):
    def __init__(self, config: SolarOpen2Config):
        super().__init__(config)
        if getattr(config, "hc_rate", 0) > 0:
            raise NotImplementedError("Hyper-Connections (hc_rate > 0) are not supported by this implementation.")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SolarOpen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = SolarOpen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SolarOpen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = SolarOpen2DynamicCache(config=self.config)

        if past_key_values is not None and not isinstance(past_key_values, SolarOpen2DynamicCache):
            raise ValueError("Solar-Open2 requires its package-owned SolarOpen2DynamicCache")
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs)
                if "full_attention" in self.config.layer_types
                else None,
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if past_key_values is not None:
            past_key_values._solar_seen_tokens += inputs_embeds.shape[1]

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class SolarOpen2ForCausalLM(SolarOpen2PreTrainedModel, GenerationMixin):
    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = SolarOpen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Run a Solar Open 2 causal language modeling forward pass."""
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


__all__ = ["SolarOpen2PreTrainedModel", "SolarOpen2Model", "SolarOpen2ForCausalLM"]
