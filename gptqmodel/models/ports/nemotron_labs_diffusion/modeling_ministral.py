# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: 2026 HuggingFace Inc.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's Ministral-shaped decoder with read-only diffusion prefix caches.

Unlike a native causal Ministral decoder, a denoising call reads the prefix
without appending speculative K/V. See NOTICE for the pinned source revision.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.output_capturing import capture_outputs

from ..native.ministral3.modeling_ministral3 import Ministral3MLP, Ministral3RMSNorm
from .configuration_nemotron_labs_diffusion import NemotronLabsDiffusionConfig


class Ministral3RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        inv_freq, self.attention_scaling = self.compute_parameters()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def compute_parameters(self, device=None):
        config = self.config
        if self.rope_type == "default":
            inv_freq = 1.0 / (config.rope_parameters["rope_theta"] ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim
            ))
            attention_scaling = 1.0
        else:
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, device=device)
        return inv_freq, attention_scaling

    def _apply(self, fn, recurse=True):
        inv_freq, original = self.inv_freq, self.original_inv_freq
        super()._apply(fn, recurse=recurse)
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        self.original_inv_freq = original.to(device=self.original_inv_freq.device, dtype=torch.float32)
        return self

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        with torch.autocast(device_type=x.device.type, enabled=False):
            frequencies = position_ids[..., None].float() * self.inv_freq.to(x.device).float()
            angles = torch.cat([frequencies, frequencies], dim=-1)
            return ((angles.cos() * self.attention_scaling).to(x.dtype),
                    (angles.sin() * self.attention_scaling).to(x.dtype))


def get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings):
    scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
    return scaling[:, None, :, None]


def apply_rotary(x, cos, sin):
    first, last = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat([-last, first], dim=-1) * sin[:, None]


def attention_keep_mask(length, prefix, device, *, causal, sliding_window=None, block_size=None):
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(prefix + length, device=device)[None, :]
    if block_size is not None:
        half = length // 2
        noisy_q, noisy_k = q < half, k < half
        qb = torch.where(noisy_q, q, q - half) // block_size
        kb = torch.where(noisy_k, k, k - half) // block_size
        return ((noisy_q & noisy_k & (qb == kb))
                | (noisy_q & ~noisy_k & (qb > kb))
                | (~noisy_q & ~noisy_k & (q >= k)))
    if causal:
        keep = k <= prefix + q
        if sliding_window is not None:
            keep = keep & (k > prefix + q - sliding_window)
        return keep
    return torch.ones(length, prefix + length, dtype=torch.bool, device=device)


class Ministral3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        # The pinned source ignores attention_bias, including when serialized True.
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                use_cache=False, scale_position_ids=None, diffusion_lm=True, block_training=False, **kwargs):
        batch, length, _ = hidden_states.shape
        q, k, v = [proj(hidden_states).view(batch, length, -1, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
        cos, sin = position_embeddings
        if block_training:
            cos, sin = cos.repeat(1, 2, 1), sin.repeat(1, 2, 1)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        rope = self.config.rope_parameters
        q = q * get_llama_4_attn_scale(
            scale_position_ids, rope["llama_4_scaling_beta"], rope["original_max_position_embeddings"]
        ).to(q.dtype)
        prefix = 0
        if past_key_values is not None:
            prefix = past_key_values.get_seq_length(self.layer_idx)
            if use_cache:
                k, v = past_key_values.update(k, v, self.layer_idx)
            elif prefix:
                layer = past_key_values.layers[self.layer_idx]
                k, v = torch.cat([layer.keys, k], dim=-2), torch.cat([layer.values, v], dim=-2)
        keep = attention_keep_mask(length, prefix, q.device, causal=not diffusion_lm,
                                   sliding_window=self.config.sliding_window,
                                   block_size=self.config.block_size if block_training else None)
        keep = keep[None, None]
        additive = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape[-1] != prefix + length:
                    raise ValueError("attention_mask must cover prefix and current tokens")
                keep = keep & attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim == 4:
                if attention_mask.dtype == torch.bool:
                    keep = keep & attention_mask
                else:
                    additive = attention_mask
            else:
                raise ValueError("attention_mask must have rank 2 or 4")
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        mask = torch.zeros(keep.shape, dtype=q.dtype, device=q.device).masked_fill(~keep, -torch.inf)
        if additive is not None:
            mask = mask + additive
        dropout = self.config.attention_dropout if self.training else 0.0
        weights = None
        if self.config._attn_implementation == "eager" or kwargs.get("output_attentions", False):
            scores = (q @ k.transpose(-1, -2)) * self.scaling + mask
            weights = scores.softmax(-1, dtype=torch.float32).to(q.dtype).nan_to_num()
            out = F.dropout(weights, p=dropout, training=self.training) @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout,
                                                is_causal=False, scale=self.scaling)
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(out), weights


class Ministral3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Ministral3Attention(config, layer_idx)
        self.mlp = Ministral3MLP(config)
        self.input_layernorm = Ministral3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ministral3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        attn, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attn
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Ministral3PreTrainedModel(PreTrainedModel):
    config_class = NemotronLabsDiffusionConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ministral3DecoderLayer"]
    _can_record_outputs = {"hidden_states": Ministral3DecoderLayer, "attentions": Ministral3Attention}
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["rotary_emb"]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.normal_(mean=0, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, Ministral3RMSNorm):
            module.weight.fill_(1)
        elif isinstance(module, Ministral3RotaryEmbedding):
            inv_freq, module.attention_scaling = module.compute_parameters(module.inv_freq.device)
            module.inv_freq.copy_(inv_freq)
            module.original_inv_freq.copy_(inv_freq)


class Ministral3Model(Ministral3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Ministral3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Ministral3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Ministral3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    # `use_cache` here means "append this call's K/V to the cache"; denoising calls read the prefix only,
    # so it deliberately does not default from config.use_cache.
    @capture_outputs
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, diffusion_lm=None,
                use_causal_mask=False, is_training=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise ValueError("Only a full DynamicCache is qualified for diffusion prefix reads")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        prefix = past_key_values.get_seq_length() if past_key_values is not None else 0
        block_training = self.config.dlm_paradigm == "block_diff" and is_training
        if block_training and (prefix or use_cache or inputs_embeds.shape[1] % 2):
            raise ValueError("Block-diff training requires equal noisy/clean halves without a cache")
        length = inputs_embeds.shape[1]
        if position_ids is None:
            position_ids = torch.arange(prefix, prefix + length, device=inputs_embeds.device).unsqueeze(0)
            if block_training:
                position_ids = position_ids[:, :length // 2]
        # The source scales queries by their absolute token index; block-diff training concatenates the noisy
        # and clean halves, so its query indices run over the full doubled length.
        if block_training:
            scale_position_ids = torch.arange(length, device=inputs_embeds.device).unsqueeze(0)
        else:
            scale_position_ids = position_ids
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        if diffusion_lm is None:
            diffusion_lm = self.config.dlm_paradigm != "autoregressive"
        if use_causal_mask:
            diffusion_lm = False
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask,
                           past_key_values=past_key_values, use_cache=use_cache,
                           scale_position_ids=scale_position_ids, diffusion_lm=diffusion_lm,
                           block_training=block_training, **kwargs)
        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden,
                                       past_key_values=past_key_values if use_cache else None)
