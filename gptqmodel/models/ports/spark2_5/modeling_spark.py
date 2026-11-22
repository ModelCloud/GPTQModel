# Modified by ModelCloud.ai: validated package-local Transformers 5 modeling.
# coding=utf-8
# Copyright 2026 The XHToken team and the HuggingFace Inc. team. All rights reserved.
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

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_spark import Spark2_5Config


logger = logging.get_logger(__name__)


def projection_input_dtype(projection: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Read packed compute metadata or dense dtype without requiring dense weights."""
    compute_dtype = getattr(projection, "compute_dtype", None)
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    weight = getattr(projection, "weight", None)
    return weight.dtype if isinstance(weight, torch.Tensor) else fallback


_CONFIG_FOR_DOC = "Spark2_5Config"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_cos_sin(positions, head_dim, rope_theta, partial_rotary_factor=1.0, device="cpu"):
    rope_head_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_head_dim, 2, dtype=torch.int64).to(device="cpu", dtype=torch.float) / rope_head_dim)
    )
    inv_freq = inv_freq.to(device)
    t = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    rope_head_dim = cos.shape[-1]
    x_f32 = x.float()
    if x_f32.shape[-1] > rope_head_dim:
        x_rot = x_f32[..., :rope_head_dim]
        x_pass = x_f32[..., rope_head_dim:]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        x_rot = x_rot * c + rotate_half(x_rot) * s
        result = torch.cat([x_rot, x_pass], dim=-1)
    else:
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        result = x_f32 * c + rotate_half(x_f32) * s
    return result.to(x.dtype)


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
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Spark2_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Spark2_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        if config.hidden_act != "gelu":
            raise ValueError("Spark2.5 supports hidden_act='gelu' only")

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Spark2_5Attention(nn.Module):
    def __init__(self, config: Spark2_5Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.gate_attn_act_mode = config.gate_attn_act_mode
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        qkv_out_dim = self.q_dim + 2 * self.kv_dim
        self.q_k_v_proj = nn.Linear(self.hidden_size, qkv_out_dim, bias=config.attention_bias)
        self.g_proj = (
            nn.Linear(self.hidden_size, self.num_heads, bias=config.attention_bias)
            if self.headwise_attn_output_gate
            else None
        )
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        bsz, seq_len = input_shape

        qkv = self.q_k_v_proj(hidden_states)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim :]
        gate_score = self.g_proj(hidden_states) if self.g_proj is not None else None

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if gate_score is not None:
            gate_score = gate_score.view(bsz, seq_len, self.num_heads, 1)

        cos, sin = position_embeddings
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if gate_score is not None:
            if self.gate_attn_act_mode == "sigmoid":
                gate = torch.sigmoid(gate_score.float())
            elif self.gate_attn_act_mode == "silu":
                gate = F.silu(gate_score.float())
            else:
                raise ValueError(f"Unsupported gate_attn_act_mode: {self.gate_attn_act_mode}")
            gate = gate.to(attn_output.dtype)
            attn_output = attn_output * gate

        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Spark2_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Spark2_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Spark2_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = Spark2_5MLP(config)
        self.input_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        if self.layer_type == "sliding_attention" and config.sliding_window is not None:
            self.self_attn.sliding_window = config.sliding_window
        else:
            self.self_attn.sliding_window = None
        self.self_attn.partial_rotary_factor = config.get_partial_rotary_factor(self.layer_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(
            projection_input_dtype(self.self_attn.q_k_v_proj, self.input_layernorm.weight.dtype)
        )

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.to(
            projection_input_dtype(self.mlp.gate_proj, self.post_attention_layernorm.weight.dtype)
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Spark2_5PreTrainedModel(PreTrainedModel):
    config_class = Spark2_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _no_split_modules = ["Spark2_5DecoderLayer"]  # noqa: RUF012
    _skip_keys_device_placement = ["past_key_values"]  # noqa: RUF012
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Spark2_5DecoderLayer, "attentions": Spark2_5Attention}  # noqa: RUF012

    def _init_weights(self, module):
        if isinstance(module, Spark2_5RMSNorm):
            nn.init.ones_(module.weight)
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Spark2_5Model(Spark2_5PreTrainedModel):
    def __init__(self, config: Spark2_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Spark2_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("Spark2.5 supports native DynamicCache only")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Upstream rotates by absolute cache slots, not by (possibly padding-adjusted) position_ids.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_slots = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_slots.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds.float()

        device = hidden_states.device
        dtype = self.embedding.weight.dtype

        head_dim = self.config.head_dim
        rope_cache = {}
        for lt in set(self.config.layer_types):
            rope_theta = self.config.get_rope_theta(lt)
            prf = self.config.get_partial_rotary_factor(lt)
            cos, sin = compute_rope_cos_sin(
                cache_slots, head_dim, rope_theta, partial_rotary_factor=prf, device=device
            )
            rope_cache[lt] = (cos, sin)

        for decoder_layer in self.layers:
            layer_type = decoder_layer.layer_type
            position_embeddings = rope_cache.get(layer_type, rope_cache.get("full_attention"))
            layer_attention_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention"))

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(dtype)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Spark2_5ForCausalLM(Spark2_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}  # noqa: RUF012

    def __init__(self, config):
        super().__init__(config)
        self.model = Spark2_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
        token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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
        hidden_states = hidden_states[:, slice_indices, :]

        if self.config.tie_word_embeddings:
            embed_weight = self.model.embedding.weight
            logits = F.linear(hidden_states, embed_weight)
        else:
            logits = self.lm_head(hidden_states)

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


__all__ = ["Spark2_5Config", "Spark2_5ForCausalLM", "Spark2_5Model"]
