# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The multi-head latent attention, MoE gating and sparse MoE block in this file are
# adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py). They have been
# extensively modified and extended for the Kimi-Linear architecture.
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the Kimi K3 License (see the LICENSE file in this repository).
#
# Apache License, Version 2.0:
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
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from ....utils.leapquant import LeapQuantConfig, LeapQuantState, mask_is_unpadded
from ..native.kimi_linear.modeling_kimi_linear import KimiLinearRMSNorm as KimiRMSNorm
from ..native.kimi_linear.modeling_kimi_linear import eager_attention_forward
from .checkpoint import ActiveKDAHeads, KimiCheckpointMixin
from .configuration_kimi_k3 import KimiLinearConfig
from .eager_kda import FusedRMSNormGated, ShortConvolution, kda_forward


# Register Moonshot-specific activation functions
class SituAndMul(nn.Module):
    """
    SituAndMul activation: beta * tanh(gate / beta) * sigmoid(gate) * up
    When linear_beta is set, up is also transformed by linear_beta * tanh(up / linear_beta).
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


def _get_situ_activation_params(config: KimiLinearConfig):
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta


def index_first_axis(x, indices):
    return x[indices]


def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = attention_mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = F.pad(lens.cumsum(0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    out = hidden_states.new_zeros((batch_size * seq_len, *hidden_states.shape[1:]))
    out[indices] = hidden_states
    return out.view(batch_size, seq_len, *hidden_states.shape[1:])


class KimiDeltaCacheLayer(LinearAttentionLayer):
    """KDA cache layer: q/k/v short-convolution states plus the delta-rule recurrent state.

    The states are the eager FLA kernels' own final states, stored as-is when
    LeapQuant is off. When it is on, the recurrent state lives in a packed
    :class:`LeapQuantState` and the dense slot stays empty. The layer also
    counts the sequence positions it has consumed, because Kimi stacks may
    contain no attention layer to derive a length from.
    """

    def __init__(self, state_quant: LeapQuantConfig | None = None):
        super().__init__(number_of_states=3)
        self.cumulative_length = 0
        self.state_quant = state_quant
        self._leapquant: LeapQuantState | None = None
        self._leapquant_shape = None

    def leapquant_state(self, num_sequences, heads, key_dim, value_dim, device):
        shape = (int(num_sequences), int(heads), int(key_dim), int(value_dim))
        device = torch.device(device)
        if self._leapquant is None or self._leapquant_shape != shape or self._leapquant.device != device:
            if self.state_quant is None:
                raise RuntimeError("Kimi KDA cache layer has no LeapQuant config")
            self._leapquant = LeapQuantState(
                num_sequences=shape[0],
                num_heads=shape[1],
                key_dim=shape[2],
                value_dim=shape[3],
                config=self.state_quant,
                gate_dim=shape[2],
                device=device,
            )
            self._leapquant_shape = shape
        return self._leapquant

    def get_kda_states(self):
        """Return ``((conv_q, conv_k, conv_v), recurrent)`` or ``(None, None)`` before the first update."""
        if not self.has_previous_state[0]:
            return None, None
        recurrent = None if self.state_quant is not None else self.recurrent_states[0]
        return (self.conv_states[0], self.conv_states[1], self.conv_states[2]), recurrent

    def set_kda_states(self, conv_states, recurrent_state, num_tokens: int):
        for index, state in enumerate(conv_states):
            self.conv_states[index] = state
            self.is_conv_states_initialized[index] = state is not None
            self.has_previous_state[index] = True
        if self.state_quant is None:
            self.recurrent_states[0] = recurrent_state
            self.is_recurrent_states_initialized[0] = recurrent_state is not None
            if recurrent_state is not None:
                self.device = recurrent_state.device
        else:
            self.recurrent_states[0] = None
            self.is_recurrent_states_initialized[0] = False
            if conv_states[0] is not None:
                self.device = conv_states[0].device
        self.cumulative_length += num_tokens

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0
        if self._leapquant is not None:
            self._leapquant.reset()

    @property
    def is_croppable(self) -> bool:
        if self.state_quant is not None:
            return False
        return super().is_croppable

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        if self._leapquant_shape is not None and self._leapquant is not None:
            self._leapquant.reorder_(beam_idx)
        super().reorder_cache(beam_idx)

    def offload(self) -> None:
        super().offload()
        if self._leapquant_shape is not None and self._leapquant is not None:
            self._leapquant.to("cpu")

    def prefetch(self) -> None:
        super().prefetch()
        if self._leapquant_shape is not None and self._leapquant is not None and self.device is not None:
            self._leapquant.to(self.device)


class KimiDynamicCache(Cache):
    """Hybrid K3 cache: ``DynamicLayer`` for MLA layers and ``KimiDeltaCacheLayer`` for KDA layers.

    It also carries the merged multimodal attention/input masks so cached continuations can extend them.
    """

    def __init__(self, config: KimiLinearConfig, state_quant: LeapQuantConfig | None = None):
        layers = [
            KimiDeltaCacheLayer(state_quant=state_quant) if config.is_kda_layer(index) else DynamicLayer()
            for index in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.multimodal_attention_mask = None
        self.multimodal_input_mask = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + query_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for name in ("multimodal_attention_mask", "multimodal_input_mask"):
            mask = getattr(self, name)
            if mask is not None:
                setattr(self, name, mask.index_select(0, beam_idx.to(mask.device)))
        super().reorder_cache(beam_idx)

    def reset(self):
        super().reset()
        self.multimodal_attention_mask = None
        self.multimodal_input_mask = None

    def crop(self, tokens_to_remove: int) -> None:
        raise ValueError("Kimi K3 recurrent cache cannot be cropped")


class KimiBlockSparseMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)  # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)  # up

        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
            current_hidden_states = self.act_fn(gate_up)
        else:
            current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "situ":
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(
                beta=beta,
                linear_beta=linear_beta,
            )
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.hidden_act == "situ":
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
            down_proj = self.down_proj(self.act_fn(gate_up))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class KimiMLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from deepseek-v3
    """

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        try:
            self.q_lora_rank = config.q_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.v_head_dim = config.v_head_dim
            self.qk_nope_head_dim = config.qk_nope_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.use_nope = config.mla_use_nope
            self.scaling = self.q_head_dim ** (-0.5)
        except Exception as e:
            raise ValueError(f"Kimi MLA config is not found or not properly formatted: {e}")

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
            )
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        self.is_causal = True
        assert self.use_nope

        self.use_output_gate = getattr(config, "mla_use_output_gate", False)
        if self.use_output_gate:
            projection_size = self.num_heads * self.v_head_dim
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            g = self.g_proj(hidden_states).sigmoid()
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)
        return attn_output


class KimiDeltaAttention(nn.Module):
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "A_log"
        if key in state_dict:
            state_dict[key] = ActiveKDAHeads(self.num_heads).convert({key: state_dict[key]}, [key], [key])[key]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads

        self.layer_idx = layer_idx

        assert self.mode in ["chunk", "fused_recurrent"], f"Not supported mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: KimiDynamicCache | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        quant_layer = cache_params.layers[self.layer_idx] if cache_params is not None else None
        state_quant = getattr(quant_layer, "state_quant", None)
        if state_quant is not None:
            if not mask_is_unpadded(attention_mask):
                raise ValueError(
                    "LeapQuant requires unpadded input; pass no attention_mask or an all-ones mask"
                )
            attention_mask = None
            if kwargs.get("cu_seqlens") is not None:
                raise ValueError("LeapQuant does not support packed cu_seqlens")
        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(hidden_states.flatten(0, 1), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            conv_states, recurrent_state = cache_params.layers[self.layer_idx].get_kda_states()
            if conv_states is not None:
                conv_state_q, conv_state_k, conv_state_v = conv_states

        q_proj_states = self.q_proj(hidden_states)
        k_proj_states = self.k_proj(hidden_states)
        v_proj_states = self.v_proj(hidden_states)
        q, conv_state_q = self.q_conv1d(
            x=q_proj_states,
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k, conv_state_k = self.k_conv1d(
            x=k_proj_states,
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v, conv_state_v = self.v_conv1d(
            x=v_proj_states,
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = g.unflatten(-1, (self.num_heads, self.head_dim))
        beta = self.b_proj(hidden_states).float()

        q, k = (x.unflatten(-1, (self.num_heads, self.head_k_dim)) for x in (q, k))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))

        if state_quant is not None:
            quantizer = quant_layer.leapquant_state(
                batch_size, self.num_heads, self.head_k_dim, self.head_dim, q.device
            )
            if q_len == 1:
                o = quantizer.kda_step(
                    q[:, 0],
                    k[:, 0],
                    v[:, 0],
                    g[:, 0],
                    beta[:, 0],
                    self.A_log,
                    self.dt_bias,
                    self.gate_lower_bound,
                ).unsqueeze(1)
                recurrent_state = None
            else:
                o, last_state = kda_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    self.A_log,
                    self.dt_bias,
                    initial_state=quantizer.materialize(),
                    lower_bound=self.gate_lower_bound,
                    cu_seqlens=None,
                )
                quantizer.quantize_(last_state)
                recurrent_state = None
            o = o.to(v.dtype)
        else:
            o, recurrent_state = kda_forward(
                q,
                k,
                v,
                g,
                beta,
                self.A_log,
                self.dt_bias,
                initial_state=recurrent_state,
                lower_bound=self.gate_lower_bound,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.layers[self.layer_idx].set_kda_states(
                (conv_state_q, conv_state_k, conv_state_v), recurrent_state, q_len
            )

        if self.use_full_rank_gate:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.unflatten(-1, (self.num_heads, self.head_dim))
        o = self.o_norm(o, g)

        o = o.flatten(-2)
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o


class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim)),
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}",
            )

        # select top-k experts
        assert not self.training
        scores = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, num_expert_group]
            group_idx = torch.topk(
                group_scores,
                k=self.topk_group,
                dim=-1,
                sorted=False,
            )[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
        else:
            tmp_scores = scores_for_choice
        _, topk_idx = torch.topk(
            tmp_scores,
            k=self.top_k,
            dim=-1,
            sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight


class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation
    The namings are consistent with Kimi's version.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ],
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config,
                intermediate_size=intermediate_size,
            )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size,
                    eps=config.rms_norm_eps,
                )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        if config.is_kda_layer(layer_idx):
            self.is_linear_attn = True
            self.self_attn = KimiDeltaAttention(config=config, layer_idx=layer_idx)
        elif config.is_mla:
            self.is_linear_attn = False
            self.self_attn = KimiMLAAttention(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(config)
        else:
            self.mlp = KimiMLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention residual
        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if self.use_attn_residuals:
            return self._forward_attn_residual(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                block_residual,
                **kwargs,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        block_residual: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.is_linear_attn is False:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        return prefix_sum, block_residual


class KimiPreTrainedModel(KimiCheckpointMixin, PreTrainedModel):
    config_class = KimiLinearConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_flash_attn = False

    _is_stateful = True
    _keep_in_fp32_modules_strict = ["A_log", "dt_bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        # TF5 `transformers.initialization` helpers skip tensors already loaded from the checkpoint.
        super()._init_weights(module)  # Linear/Conv1d/Embedding and *RMSNorm* weights
        if isinstance(module, KimiMoEGate):
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, KimiDeltaAttention):
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            init.zeros_(module.dt_bias)


def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    """
    prefix_sum:     (num_tokens, hidden_size)
    block_residual: (num_tokens, num_blocks, hidden_size)
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


class KimiLinearModel(KimiPreTrainedModel):
    supports_state_quant = True

    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _update_linear_attn_mask(self, attention_mask):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if attention_mask is not None and torch.all(attention_mask == 1):
            linear_attn_mask = None
        return linear_attn_mask

    # @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and not isinstance(past_key_values, KimiDynamicCache):
            raise ValueError("Kimi K3 requires its package-owned KimiDynamicCache")

        # Get inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = KimiDynamicCache(
                config=self.config,
                state_quant=getattr(self, "_gptqmodel_state_quant", None),
            )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        causal_mask = _causal_mask(inputs_embeds, attention_mask, past_seen_tokens)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask)

        hidden_states = inputs_embeds

        block_residual = None
        if self.use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]
            )

        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask

            if self.use_attn_residuals:
                hidden_states, block_residual = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    past_key_values=past_key_values,
                    block_residual=block_residual,
                    **kwargs,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

        if self.use_attn_residuals:
            hidden_states = self._apply_output_attn_res(hidden_states, block_residual)

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _apply_output_attn_res(self, hidden_states, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        return _apply_attn_res(
            hidden_states.view(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch_size, seq_len, hidden_size)


class KimiLinearForCausalLM(KimiPreTrainedModel, GenerationMixin):
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = KimiLinearModel(config)
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
        past_key_values: KimiDynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
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
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


def _causal_mask(input_embeds, attention_mask, past):
    length = input_embeds.shape[1]
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask
    keys = torch.arange(past + length, device=input_embeds.device)
    queries = torch.arange(past, past + length, device=input_embeds.device)
    allowed = keys[None, :] <= queries[:, None]
    allowed = allowed[None, None].expand(input_embeds.shape[0], 1, length, -1)
    if attention_mask is not None:
        if attention_mask.shape != (input_embeds.shape[0], past + length):
            raise ValueError("attention_mask must include the cached prefix")
        allowed = allowed & attention_mask[:, None, None, :].bool()
    return torch.zeros(allowed.shape, dtype=input_embeds.dtype, device=input_embeds.device).masked_fill(
        ~allowed,
        torch.finfo(input_embeds.dtype).min,
    )


class KimiModelFactory:
    @classmethod
    def _from_config(cls, config, **kwargs):
        from ..native.kimi_linear.configuration_kimi_linear import KimiLinearConfig as NativeConfig

        if type(config) is NativeConfig:
            from ..native.kimi_linear.modeling_kimi_linear import KimiLinearForCausalLM as NativeModel

            return NativeModel._from_config(config, **kwargs)
        from .configuration_kimi_k3 import KimiK3Config

        if isinstance(config, KimiK3Config):
            from .modeling_kimi_k3 import KimiK3ForConditionalGeneration

            return KimiK3ForConditionalGeneration._from_config(config, **kwargs)
        return KimiLinearForCausalLM._from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from ..native.kimi_linear.configuration_kimi_linear import KimiLinearConfig as NativeConfig

        config = kwargs["config"]
        if type(config) is NativeConfig:
            from ..native.kimi_linear.modeling_kimi_linear import KimiLinearForCausalLM as NativeModel

            return NativeModel.from_pretrained(path, **kwargs)
        from .configuration_kimi_k3 import KimiK3Config

        if isinstance(config, KimiK3Config):
            from .modeling_kimi_k3 import KimiK3ForConditionalGeneration

            return KimiK3ForConditionalGeneration.from_pretrained(path, **kwargs)
        return KimiLinearForCausalLM.from_pretrained(path, **kwargs)
