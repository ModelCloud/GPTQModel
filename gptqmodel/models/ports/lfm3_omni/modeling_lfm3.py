# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source, modifications and qualification limits: README.md and NOTICE.
import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, torch_compilable_check
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.import_utils import is_tracing
from transformers.utils.output_capturing import capture_outputs

from .configuration_lfm3 import Lfm3Config


class Lfm3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Lfm3RMSNorm is equivalent to T5LayerNorm
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


class Lfm3RotaryEmbedding(nn.Module):
    """
    Rotary embedding per attention layer type, like Gemma 3; `conv` layers register no buffers. Partial RoPE keeps the
    frequency progression of the full head dimension and zeroes the tail instead of shrinking the rotated dimension.
    """

    def __init__(self, config: Lfm3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = sorted(set(config.layer_types) & set(config.rope_parameters))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            curr_inv_freq, curr_attention_scaling = self.compute_rope_parameters(config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Lfm3Config, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Inverse frequencies of the default RoPE with LFM3's zero-tail partial rotation: the first
        `partial_rotary_factor * head_dim // 2` frequencies follow the progression of the full head dimension and the
        rest are zero, leaving those dimensions unrotated.

        Returns:
            Tuple of (`torch.Tensor`, `float`): the inverse frequencies and the attention scaling factor (always 1).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_angles = int(partial_rotary_factor * dim) // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / dim))
        inv_freq = torch.cat((inv_freq, torch.zeros(dim // 2 - rope_angles, dtype=torch.float)))
        return inv_freq.to(device), 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @classmethod
    def compute_rope_parameters(
        cls, config: Lfm3Config, device=None, layer_type: str | None = None
    ) -> tuple[torch.Tensor, float]:
        """Inverse frequencies of any RoPE type in the zero-tail layout: scaled types are evaluated over the full head
        dimension and truncated to the rotated prefix."""
        rope_params = config.rope_parameters[layer_type]
        if rope_params["rope_type"] == "default":
            return cls.compute_default_rope_parameters(config, device, layer_type=layer_type)
        full_config = copy.copy(config)
        full_config.rope_parameters = {
            **config.rope_parameters,
            layer_type: {**rope_params, "partial_rotary_factor": 1.0},
        }
        inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_params["rope_type"]](
            full_config, device, layer_type=layer_type
        )
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rope_angles = int(rope_params.get("partial_rotary_factor", 1.0) * dim) // 2
        inv_freq = torch.cat((inv_freq[:rope_angles], inv_freq.new_zeros(dim // 2 - rope_angles)))
        return inv_freq, attention_factor


class Lfm3MLP(nn.Module):
    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies RoPE to the query and key tensors in float32, like the reference implementation."""
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class Lfm3Attention(nn.Module):
    """Llama attention with QK RMSNorm, a sigmoid output gate per head, a sliding window on `sliding_attention`
    layers and RoPE in float32."""

    def __init__(self, config: Lfm3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        gate = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        attn_output = (attn_output * gate).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


class Lfm3ShortConv(nn.Module):
    """LFM2 gated short convolution with packed-document boundaries (`seq_idx`), the padding mask re-applied after a
    biased `in_proj`, and package-owned PyTorch convolution for every kernel width."""

    def __init__(
        self,
        config: Lfm3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.conv_kernel_size - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv")
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        seq_idx: torch.IntTensor | None = None,
    ):
        seq_len = hidden_states.shape[1]

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        hidden_states = B * x
        if attention_mask is not None:
            # with `conv_bias` the projection makes the padded positions non-zero again
            hidden_states = hidden_states * attention_mask[:, None, :].to(hidden_states.dtype)

        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        use_kernels = 2 <= self.conv_kernel_size <= 4

        if use_precomputed_states and seq_len == 1 and not past_key_values.layers[self.layer_idx].record_past:
            conv_state = past_key_values.layers[self.layer_idx].conv_states[0]
            if use_kernels:
                hidden_states = causal_conv1d_update(
                    hidden_states, conv_state, self.conv.weight.squeeze(1), self.conv.bias
                )
            else:
                hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
                conv_state.copy_(hidden_states[:, :, -conv_state.shape[-1] :])
                hidden_states = self._conv(hidden_states)[:, :, -seq_len:]
        else:
            if past_key_values is not None and seq_idx is not None:
                # `seq_idx` is only set on an empty cache, and the decode steps read the cached window without it:
                # only the last document (after any positional left padding) may enter the window
                last_document = (seq_idx == seq_idx[:, -1:])[:, None].to(hidden_states.dtype)
                past_key_values.update_conv_state(
                    hidden_states * last_document, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            elif past_key_values is not None:
                hidden_states = past_key_values.update_conv_state(
                    hidden_states, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            if seq_idx is not None:
                hidden_states = self._segmented_conv(hidden_states, seq_idx)
            elif use_kernels:
                hidden_states = causal_conv1d_fn(hidden_states, self.conv.weight.squeeze(1), self.conv.bias)
            else:
                hidden_states = self._conv(hidden_states)

            # Drop the additional previous states
            if past_key_values is not None:
                hidden_states = hidden_states[:, :, -seq_len:]

        y = C * hidden_states
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

    def _conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv in torch: the `causal_conv1d` kernels only accept widths 2 to 4."""
        conv_out = self.conv(hidden_states.to(self.conv.weight.dtype))[..., : hidden_states.shape[-1]]
        return conv_out.to(hidden_states.dtype)

    def _segmented_conv(self, hidden_states: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv that only reads positions sharing the current token's document index."""
        kernel_size = self.conv_kernel_size
        seq_idx = F.pad(seq_idx, (hidden_states.shape[-1] - seq_idx.shape[-1] + kernel_size - 1, 0), value=-1)
        windows = F.pad(hidden_states, (kernel_size - 1, 0)).unfold(-1, kernel_size, 1)
        seq_idx_windows = seq_idx.unfold(-1, kernel_size, 1)
        same_document = (seq_idx_windows == seq_idx_windows[..., -1:]).unsqueeze(1)
        conv_out = torch.einsum("bdtk,dk->bdt", windows * same_document.to(windows.dtype), self.conv.weight.squeeze(1))
        if self.bias:
            conv_out = conv_out + self.conv.bias[None, :, None]
        return conv_out.to(hidden_states.dtype)


class Lfm3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm3Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] in config._attention_layer_types
        self.use_sandwich_norm = config.use_sandwich_norm

        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(config, layer_idx)
        else:
            self.conv = Lfm3ShortConv(config, layer_idx)
        self.feed_forward = Lfm3MLP(config)
        self.operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        seq_idx: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
            )
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = hidden_states + residual

        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        hidden_states = hidden_states + ffn_out

        return hidden_states


def create_masks_by_layer_type(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
) -> dict:
    """The cache sizes each mask from the layer it is asked about, and the generic mask utilities ask about the first
    non-sliding layer, which for a leading `conv` layer redirects to the first attention layer of any type: name the
    first layer of each mask's own type instead."""
    config = config.get_text_config()
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    layer_types, causal_mask_mapping = config.layer_types, {}
    if "full_attention" in layer_types:
        layer_idx = layer_types.index("full_attention")
        causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs, layer_idx=layer_idx)
    if "sliding_attention" in layer_types:
        layer_idx = layer_types.index("sliding_attention")
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs, layer_idx=layer_idx
        )
    if "conv" in layer_types:
        causal_mask_mapping["conv"] = create_recurrent_attention_mask(**mask_kwargs)
    return causal_mask_mapping


@auto_docstring
class Lfm3PreTrainedModel(PreTrainedModel):
    config: Lfm3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Lfm3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Lfm3DecoderLayer,
        "attentions": Lfm3Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3RotaryEmbedding):
            for layer_type in module.layer_types:
                curr_inv_freq, _ = module.compute_rope_parameters(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


def packed_under_padding(position_ids: torch.Tensor, padding_mask: torch.Tensor, filled) -> torch.Tensor:
    """Padding repeats position 0, so packing is a restart between two kept tokens, or a kept position 0 on a filled
    cache whose row holds a kept token; attention under a padding mask cannot isolate it."""
    restarts = torch.diff(position_ids, prepend=position_ids[:, :1] - 1, dim=-1) != 1
    valid = padding_mask[:, -restarts.shape[-1] :].bool()
    packed = restarts[:, 1:] & valid[:, 1:] & valid[:, :-1]
    if filled:
        # a row whose cached tokens are all padding (chunked prefill of a left-padded batch) has no document to pack
        # against; a mask covering only the chunk keeps the strict rule
        cached = padding_mask[:, : padding_mask.shape[-1] - restarts.shape[-1]]
        cached_valid = cached.bool().any(-1, keepdim=True) if cached.shape[-1] else torch.ones_like(valid[:, :1])
        packed = packed | (valid[:, :1] & (position_ids[:, :1] == 0) & cached_valid)
    return packed


@auto_docstring
class Lfm3Model(Lfm3PreTrainedModel):
    """LFM2 model with sliding-attention layers as a third layer type: the mask and the rotary embedding are built per
    layer type, and packed documents (`position_ids` restarting) reset the short-conv window through `seq_idx`."""

    def __init__(self, config: Lfm3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Lfm3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Lfm3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.embedding_norm = Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `dict`, *optional*):
            Mask to avoid performing attention on padding token indices (1 keeps a token, 0 masks it), or the mask
            dict of `create_masks_for_generate`. A mask dict is used as given: `generate` refuses packed
            `position_ids` before its prefill.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # `seq_idx` is a shared kwarg (`DataCollatorWithFlattening(return_seq_idx=True)`, read by Lfm2); LFM3 derives
        # the documents from the restarting `position_ids` that come with it
        if kwargs.pop("seq_idx", None) is not None and position_ids is None:
            raise ValueError("Pass packed documents as `position_ids` restarting at 0; LFM3 does not read `seq_idx`")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        filled = past_key_values is not None and (
            past_key_values.get_seq_length() > 0
            if any(t != "conv" for t in self.config.layer_types)
            else past_key_values.has_previous_state()
        )
        packed_cache = getattr(past_key_values, "lfm3_packed", None)
        if isinstance(packed_cache, torch.Tensor) or packed_cache:
            # Reset layers have no history to mix. Ordinary caches carry a Python `False` and skip this, so the decode
            # steps do no tensor work here; only a compiled packed prefill leaves a tensor flag, and `filled` stays
            # tensor-valued for static caches.
            packed_message = (
                "A cache filled by a packed-document forward cannot be continued: its attention spans the documents"
            )
            packed_cache = packed_cache & filled if isinstance(packed_cache, torch.Tensor) else filled
            if isinstance(packed_cache, torch.Tensor):
                torch_compilable_check(~packed_cache, packed_message)
                if not is_tracing(packed_cache):
                    past_key_values.lfm3_packed = bool(packed_cache)
            elif packed_cache:
                raise ValueError(packed_message)
            else:
                past_key_values.lfm3_packed = False

        # The mask utilities detect packing only with `attention_mask=None`, so a multi-token forward with explicit
        # positions drops the all-ones mask tokenizers emit (a data-dependent check, hence not when traced).
        explicit_positions = position_ids is not None and inputs_embeds.shape[1] > 1
        if (
            explicit_positions
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and not is_tracing(position_ids)
            and bool(attention_mask.all())
        ):
            attention_mask = None
        seq_idx = None
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        elif explicit_positions and not isinstance(attention_mask, dict):
            # a mask dict comes from `generate`, which refused packing before its prefill
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                # data-dependent: a traced forward skips it (`generate` refuses packing eagerly before its prefill)
                if not is_tracing(position_ids):
                    torch_compilable_check(
                        ~packed_under_padding(position_ids, attention_mask, filled),
                        "Packed documents (`position_ids` restarting) cannot be combined with an attention mask: "
                        "attention would not isolate the documents. Pass `attention_mask=None` without a static cache.",
                    )
            else:
                restarts = torch.diff(position_ids, prepend=position_ids[:, :1] - 1, dim=-1) != 1
                document_ids = restarts.cumsum(-1)
                if is_tracing(position_ids) or bool((document_ids[..., -1] > 0).any()):
                    seq_idx = document_ids.to(torch.int32)
                # Packed documents are prefilled in one forward on an empty cache (cached tokens would join the first
                # document), and their cache is never continued
                if filled:
                    torch_compilable_check(
                        (~restarts[:, 1:]).all() & (position_ids[:, 0] != 0).all(),
                        "Packed documents must be prefilled in one forward; the cache already holds tokens",
                    )
                    # Ordinary continuation must retain the convolution window from the cached document.
                    seq_idx = None
                if attention_mask is None:
                    # without a mask only the cacheless packed-sequence mask isolates repeated zero positions (left
                    # padding): a cache-sized mask and every later decode step would attend them, so they count too
                    packed = (document_ids[..., -1] > 0).any()
                else:
                    # a prepared 4D mask is the caller's and carries the padding (repeated zero positions), but may
                    # or may not isolate documents, so the cache records the packing and is never continued
                    steps = torch.diff(position_ids, dim=-1)
                    packed = ((steps != 1) & ~((steps == 0) & (position_ids[:, :-1] == 0))).any()
                if past_key_values is not None:
                    # resolved here, where the prefill already syncs, so the decode steps carry a Python flag: no
                    # per-step host sync, data-dependent op or CUDA-graph output on the cache
                    past_key_values.lfm3_packed = packed if is_tracing(position_ids) else bool(packed)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # The mask utilities only isolate packed documents without a cache, and an empty cache adds nothing.
            layer_types = self.config.layer_types
            if seq_idx is not None and past_key_values is not None:
                attention_layer = next((i for i, t in enumerate(layer_types) if t != "conv"), None)
                if attention_layer is not None and past_key_values.get_seq_length(attention_layer) == 0:
                    if past_key_values.is_compileable:
                        # static layers always return their full length, so the masks are sized by the cache, which
                        # isolates neither documents nor positional-only left padding
                        torch_compilable_check(
                            ~packed,
                            "Packed documents, or left padding without an attention mask, cannot be prefilled into a "
                            "static cache: its masks are sized by the cache. Pass the padding as `attention_mask`, or "
                            "use a `DynamicCache` for packed documents",
                        )
                    else:
                        mask_kwargs["past_key_values"] = None
            causal_mask_mapping = create_masks_by_layer_type(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                past_key_values=past_key_values,
                seq_idx=seq_idx,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



__all__ = ["Lfm3Model"]
