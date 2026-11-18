# SPDX-FileCopyrightText: Nanbeige contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Adapted from Nanbeige/Nanbeige4.2-3B at b82e54bd609793562a75cbf9337970a93369eab5.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging

from .configuration_nanbeige import NanbeigeConfig


logger = logging.get_logger(__name__)
DepthAttentionCacheEntry = Tuple[int, torch.Tensor, torch.Tensor]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    limit = math.isqrt(value)
    for factor in range(3, limit + 1, 2):
        if value % factor == 0:
            return False
    return True


def _next_prime_after(value: float) -> int:
    candidate = int(value) + 1
    if candidate <= 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def _ngram_embedding_vocab_sizes(m: float, num_tables: int, force_prime: bool) -> List[int]:
    if not force_prime:
        return [int(m + index * 2 + 1) for index in range(num_tables)]
    vocab_sizes = []
    previous = m
    for _ in range(num_tables):
        previous = _next_prime_after(previous)
        vocab_sizes.append(previous)
    return vocab_sizes


def _ngram_hash_base(vocab_size: int, force_prime: bool) -> int:
    if not force_prime:
        return vocab_size
    return _next_prime_after(vocab_size)


def _get_loop_cache_layer_idx(
    layer_idx: Optional[int], loop_idx: int, num_hidden_layers: int, cache_layer_idx: Optional[int] = None
) -> int:
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop-aware caching is enabled.")
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers


def _get_num_physical_layers(config: NanbeigeConfig) -> int:
    physical = getattr(config, "num_physical_layers", None)
    if physical is not None:
        return physical
    return config.num_hidden_layers


def _apply_loop_shared_kv(
    loop_share_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layer_idx: Optional[int],
    mhc_loop_idx: Optional[int],
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if loop_share_kv_cache is None or mhc_loop_idx is None:
        return (key_states, value_states)
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop_share_kv is enabled.")
    if mhc_loop_idx == 0:
        loop_share_kv_cache[layer_idx] = (key_states, value_states)
        return (key_states, value_states)
    if layer_idx not in loop_share_kv_cache:
        raise RuntimeError(f"loop_share_kv missing first-pass KV for layer {layer_idx}.")
    return loop_share_kv_cache[layer_idx]


def _reduce_query_to_kv_groups(query: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
    num_query_heads = query.shape[1]
    if num_query_heads == num_kv_groups:
        return query
    if num_query_heads % num_kv_groups != 0:
        raise ValueError(f"query heads ({num_query_heads}) must be divisible by KV groups ({num_kv_groups}).")
    return query.reshape(
        query.shape[0], num_kv_groups, num_query_heads // num_kv_groups, query.shape[2], query.shape[3]
    ).mean(dim=2)


def _depth_attention_mix_value(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    source_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    source_kv = list(source_kv)
    if not source_kv:
        return current_value
    num_kv_groups = current_key.shape[1]
    query_for_kv = _reduce_query_to_kv_groups(query, num_kv_groups)
    if query_for_kv.shape != current_key.shape:
        raise ValueError(f"query/K shape mismatch after GQA grouping: {query_for_kv.shape} vs {current_key.shape}.")
    keys = [key for key, _ in source_kv] + [current_key]
    values = [value for _, value in source_kv] + [current_value]
    for key in keys:
        if key.shape != current_key.shape:
            raise ValueError(f"source key shape {key.shape} does not match {current_key.shape}.")
    for value in values:
        if value.shape != current_value.shape:
            raise ValueError(f"source value shape {value.shape} does not match {current_value.shape}.")
    key_stack = torch.stack(keys, dim=0)
    value_stack = torch.stack(values, dim=0)
    logits = (query_for_kv.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** (-0.5)
    depth_probs = torch.softmax(logits * softmax_scale, dim=0).to(value_stack.dtype)
    return (depth_probs.unsqueeze(-1) * value_stack).sum(dim=0).to(current_value.dtype)


def _apply_depth_attention(
    config: NanbeigeConfig,
    layer_idx: Optional[int],
    depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depth_attention_kv_cache is None:
        return (key_states, value_states)
    if layer_idx is None:
        raise ValueError("layer_idx must be set when enable_depth_attention=True.")
    source_kv = [(key, value) for _, key, value in depth_attention_kv_cache]
    value_states = _depth_attention_mix_value(
        query_states, key_states, value_states, source_kv, softmax_scale=softmax_scale
    )
    if layer_idx % config.depth_attention_stride == 0:
        depth_attention_kv_cache.append((layer_idx, key_states, value_states))
    return (key_states, value_states)


def _apply_depth_attention_then_update_cache(
    config: NanbeigeConfig,
    layer_idx: Optional[int],
    depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: Optional[Cache],
    loop_idx: int,
    loop_cache_layer_idx: Optional[int],
    cache_kwargs: Dict[str, Any],
    skip_cache_update: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states, value_states = _apply_depth_attention(
        config,
        layer_idx,
        depth_attention_kv_cache,
        query_states,
        key_states,
        value_states,
        softmax_scale=softmax_scale,
    )
    if past_key_value is not None and (not skip_cache_update):
        cache_layer_idx = _get_loop_cache_layer_idx(
            layer_idx, loop_idx, _get_num_physical_layers(config), loop_cache_layer_idx
        )
        key_states, value_states = past_key_value.update(key_states, value_states, cache_layer_idx, cache_kwargs)
    return (key_states, value_states)


def _get_double_loop_split_layer_order(num_hidden_layers: int, loop_middle_layers: Optional[int] = None) -> List[int]:
    return [
        layer_idx
        for layer_idx, _ in _get_double_loop_split_layer_order_with_mhc_loop_indices(
            num_hidden_layers, loop_middle_layers
        )
    ]


def _get_double_loop_split_layer_order_with_mhc_loop_indices(
    num_hidden_layers: int, loop_middle_layers: Optional[int] = None
) -> List[Tuple[int, Optional[int]]]:
    if num_hidden_layers <= 0:
        raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
    if loop_middle_layers is None:
        if num_hidden_layers % 2 != 0:
            raise ValueError(
                "enable_double_loop_split requires num_hidden_layers to be divisible by 2 when loop_middle_layers is not set."
            )
        loop_middle_layers = num_hidden_layers // 2
    if loop_middle_layers <= 0:
        raise ValueError("loop_middle_layers must be greater than 0.")
    if num_hidden_layers % loop_middle_layers != 0:
        raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")
    first_unlooped_layers = (num_hidden_layers - loop_middle_layers) // 2
    middle_start = first_unlooped_layers
    middle_end = middle_start + loop_middle_layers
    middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
    return (
        [(idx, None) for idx in range(0, middle_start)]
        + [(idx, repeat_idx) for repeat_idx in range(middle_repeats) for idx in range(middle_start, middle_end)]
        + [(idx, None) for idx in range(middle_end, num_hidden_layers)]
    )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(attn_weights.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SinkhornKnopp(torch.autograd.Function):
    @staticmethod
    def _normalize(matrix: torch.Tensor, iterations: int, eps: float = 1e-06) -> torch.Tensor:
        for _ in range(iterations):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=eps)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp(min=eps)
        return matrix

    @staticmethod
    def forward(ctx, logits: torch.Tensor, iterations: int):
        base = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        result = SinkhornKnopp._normalize(base, iterations)
        ctx.save_for_backward(base)
        ctx.iterations = iterations
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (base,) = ctx.saved_tensors
        with torch.enable_grad():
            base_input = base.detach().requires_grad_(True)
            current = SinkhornKnopp._normalize(base_input, ctx.iterations)
            (grad_base,) = torch.autograd.grad(
                outputs=current, inputs=base_input, grad_outputs=grad_output, create_graph=False, retain_graph=False
            )
        return (grad_base * base, None)


class NanbeigeNgramLayerFusion(nn.Module):
    def __init__(self, config: NanbeigeConfig):
        super().__init__()
        self.fusion_size = config.ngram_layer_downproject_size or config.hidden_size
        if config.ngram_layer_downproject_size is None:
            self.hidden_down_proj = None
            self.output_proj = None
        else:
            self.hidden_down_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)
            self.output_proj = nn.Linear(self.fusion_size, config.hidden_size, bias=False)
        self.hidden_norm = NanbeigeRMSNorm(self.fusion_size, eps=config.rms_norm_eps)
        self.ngram_norm = NanbeigeRMSNorm(self.fusion_size, eps=config.rms_norm_eps)
        self.key_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, ngram_embeddings: torch.Tensor) -> torch.Tensor:
        key = self.key_proj(ngram_embeddings)
        normed_key = self.ngram_norm(key)
        hidden_for_gate = hidden_states
        if self.hidden_down_proj is not None:
            hidden_for_gate = self.hidden_down_proj(hidden_states)
        normed_hidden = self.hidden_norm(hidden_for_gate)
        gate = (normed_hidden * normed_key).sum(dim=-1, keepdim=True) / math.sqrt(self.fusion_size)
        gate = gate.abs().clamp_min(1e-06).sqrt() * gate.sign()
        gate = gate.sigmoid()
        fused = gate * self.value_proj(ngram_embeddings)
        if self.output_proj is not None:
            fused = self.output_proj(fused)
        return hidden_states + fused


class NanbeigeHyperConnectionModule(nn.Module):
    def __init__(
        self, config: NanbeigeConfig, layer_idx: int, module_name: str, num_residual_streams: Optional[int] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.enable_mhc = config.enable_mhc
        self.enable_h_res_identity = config.enable_h_res_identity
        self.mhc_identity_nohresparam = getattr(config, "mhc_identity_nohresparam", False)
        self.num_residual_streams = (
            config.num_residual_streams if num_residual_streams is None else num_residual_streams
        )
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.norm_eps = 1e-06
        in_dim = self.num_residual_streams * self.hidden_size
        init_alpha = config.mhc_init_gating_factor
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))
        if self.enable_mhc:
            out_dim = (
                2 * self.num_residual_streams
                if self.mhc_identity_nohresparam
                else self.num_residual_streams * self.num_residual_streams + 2 * self.num_residual_streams
            )
            self.mapping_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            out_dim = self.num_residual_streams * self.num_residual_streams + 2 * self.num_residual_streams
            self.mapping_proj = nn.Linear(in_dim, out_dim, bias=True)
            self.bias = None
        self._build_static_mappings()
        self._init_dynamic_zero()
        self._disable_h_res_identity_unused_params()

    def _disable_h_res_identity_unused_params(self):
        if not self.enable_h_res_identity:
            return
        self.alpha_res.requires_grad_(False)

    def _static_mappings(self):
        n = self.num_residual_streams
        h_pre_static = torch.zeros(n)
        h_pre_static[self.layer_idx % n] = 1.0
        return {"h_pre_static": h_pre_static, "h_post_static": torch.ones(n), "h_res_static": torch.eye(n)}

    def _build_static_mappings(self):
        for name, value in self._static_mappings().items():
            self.register_buffer(name, value)

    def _init_static_mappings(self):
        for name, value in self._static_mappings().items():
            init.copy_(getattr(self, name), value)

    def _init_dynamic_zero(self):
        init.zeros_(self.mapping_proj.weight)
        n = self.num_residual_streams
        if self.enable_mhc:
            bias = torch.zeros(self.bias.shape, dtype=self.bias.dtype, device=self.bias.device)
            bias[:n] = -20.0
            bias[self.layer_idx % n] = 20.0
            if not self.mhc_identity_nohresparam:
                h_res_init = bias.new_full((n, n), -20.0)
                h_res_init[torch.arange(n), torch.arange(n)] = 20.0
                bias[2 * n :] = h_res_init.reshape(-1)
            init.copy_(self.bias, bias)
        else:
            init.zeros_(self.mapping_proj.bias)

    @staticmethod
    def input_expand(hidden_states: torch.Tensor, num_residual_streams: int) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        expanded = hidden_states.unsqueeze(2).expand(batch_size, seq_len, num_residual_streams, hidden_size)
        return expanded.contiguous().view(batch_size, seq_len, num_residual_streams * hidden_size)

    @staticmethod
    def output_contract(hidden_states: torch.Tensor, num_residual_streams: int) -> torch.Tensor:
        batch_size, seq_len, n_hidden_size = hidden_states.shape
        if n_hidden_size % num_residual_streams != 0:
            raise RuntimeError(
                f"HC output_contract shape mismatch: hidden={n_hidden_size}, streams={num_residual_streams}"
            )
        hidden_size = n_hidden_size // num_residual_streams
        streams = hidden_states.view(batch_size, seq_len, num_residual_streams, hidden_size)
        return streams.mean(dim=2)

    @staticmethod
    def convert_stream_count(
        hidden_states: torch.Tensor, hidden_size: int, target_num_residual_streams: int
    ) -> torch.Tensor:
        batch_size, seq_len, n_hidden_size = hidden_states.shape
        if n_hidden_size == hidden_size:
            return NanbeigeHyperConnectionModule.input_expand(hidden_states, target_num_residual_streams)
        if n_hidden_size % hidden_size != 0:
            raise RuntimeError(
                f"HC convert_stream_count shape mismatch: hidden={n_hidden_size}, base_hidden={hidden_size}"
            )
        current_num_residual_streams = n_hidden_size // hidden_size
        if current_num_residual_streams == target_num_residual_streams:
            return hidden_states
        streams = hidden_states.view(batch_size, seq_len, current_num_residual_streams, hidden_size)
        if target_num_residual_streams % current_num_residual_streams == 0:
            repeat = target_num_residual_streams // current_num_residual_streams
            streams = streams.repeat_interleave(repeat, dim=2)
            return streams.contiguous().view(batch_size, seq_len, target_num_residual_streams * hidden_size)
        if current_num_residual_streams % target_num_residual_streams == 0:
            group = current_num_residual_streams // target_num_residual_streams
            streams = streams.view(batch_size, seq_len, target_num_residual_streams, group, hidden_size)
            return (
                streams.mean(dim=3).contiguous().view(batch_size, seq_len, target_num_residual_streams * hidden_size)
            )
        contracted = NanbeigeHyperConnectionModule.output_contract(hidden_states, current_num_residual_streams)
        return NanbeigeHyperConnectionModule.input_expand(contracted, target_num_residual_streams)

    def _compute_mappings(self, hidden_states: torch.Tensor):
        n = self.num_residual_streams
        h_res_identity = self.h_res_static.view(1, 1, n, n).to(dtype=hidden_states.dtype)
        if self.enable_mhc:
            if self.enable_h_res_identity:
                proj_weight = (
                    self.mapping_proj.weight if self.mhc_identity_nohresparam else self.mapping_proj.weight[: 2 * n, :]
                )
                proj = F.linear(hidden_states, proj_weight)
                n_channels = hidden_states.shape[-1]
                r = hidden_states.norm(dim=-1, keepdim=True) / math.sqrt(n_channels)
                r = 1.0 / (r + self.norm_eps)
                bias = self.bias.to(dtype=hidden_states.dtype)
                alpha_pre = self.alpha_pre.to(dtype=hidden_states.dtype)
                alpha_post = self.alpha_post.to(dtype=hidden_states.dtype)
                h_pre_logits = r * proj[..., :n] * alpha_pre + bias[:n].view(1, 1, n)
                h_post_logits = r * proj[..., n : 2 * n] * alpha_post + bias[n : 2 * n].view(1, 1, n)
                h_pre = h_pre_logits.sigmoid()
                h_post = h_post_logits.sigmoid() * 2.0
            else:
                proj = self.mapping_proj(hidden_states)
                n_channels = hidden_states.shape[-1]
                r = hidden_states.norm(dim=-1, keepdim=True) / math.sqrt(n_channels)
                r = 1.0 / (r + self.norm_eps)
                alpha = torch.cat(
                    [self.alpha_pre.expand(n), self.alpha_post.expand(n), self.alpha_res.expand(n * n)], dim=0
                ).to(dtype=hidden_states.dtype)
                h = r * proj * alpha + self.bias.to(dtype=hidden_states.dtype).view(1, 1, -1)
                h_pre = h[..., :n].sigmoid()
                h_post = h[..., n : 2 * n].sigmoid() * 2.0
            if self.enable_h_res_identity:
                h_res = h_res_identity.expand(hidden_states.shape[0], hidden_states.shape[1], n, n)
            else:
                h_res_logits = h[..., 2 * n :].view(hidden_states.shape[0], hidden_states.shape[1], n, n)
                h_res = SinkhornKnopp.apply(h_res_logits, self.sinkhorn_iterations)
        else:
            normalized = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            logits = torch.tanh(self.mapping_proj(normalized))
            h_pre_logits = logits[..., :n]
            h_post_logits = logits[..., n : 2 * n]
            h_pre = h_pre_logits * self.alpha_pre + self.h_pre_static.view(1, 1, n).to(dtype=hidden_states.dtype)
            h_post = h_post_logits * self.alpha_post + self.h_post_static.view(1, 1, n).to(dtype=hidden_states.dtype)
            if self.enable_h_res_identity:
                h_res = h_res_identity.expand(hidden_states.shape[0], hidden_states.shape[1], n, n)
            else:
                h_res_logits = logits[..., 2 * n :].view(logits.shape[0], logits.shape[1], n, n)
                h_res = h_res_logits * self.alpha_res + h_res_identity
        return (h_pre, h_post, h_res)

    def forward(self, hidden_states: torch.Tensor):
        h_pre, h_post, h_res = self._compute_mappings(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        streams = hidden_states.view(batch_size, seq_len, self.num_residual_streams, self.hidden_size)
        aggregated = (streams * h_pre.unsqueeze(-1)).sum(dim=2)
        return (aggregated, h_res, h_post)

    def fuse_residual(self, h_res: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, output: torch.Tensor):
        batch_size, seq_len, _ = residual.shape
        if self.enable_h_res_identity:
            mixed_residual = residual
        else:
            residual_streams = residual.view(batch_size, seq_len, self.num_residual_streams, self.hidden_size)
            mixed_residual = torch.matmul(h_res, residual_streams).view(
                batch_size, seq_len, self.num_residual_streams * self.hidden_size
            )
        expanded_output = (
            (h_post.unsqueeze(-1) * output.unsqueeze(2))
            .contiguous()
            .view(batch_size, seq_len, self.num_residual_streams * self.hidden_size)
        )
        return mixed_residual + expanded_output


class NgramCache(DynamicCache):
    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        if config is not None and config.emb_neighbor_num is not None:
            self.max_context_len = config.emb_neighbor_num - 1
        else:
            self.max_context_len = 0

    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        if self.max_context_len == 0:
            return
        if self.ngram_context is None:
            self.ngram_context = new_tokens.clone()
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len :]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        super().reorder_cache(beam_idx)
        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(0, beam_idx.to(self.ngram_context.device))
        return self


class NanbeigeNgramEmbedding(nn.Module):
    def __init__(self, config, base_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings
        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.tp = config.emb_tp_num
        self.ngram_mod_force_prime = getattr(config, "ngram_mod_force_prime", False)
        self.ngram_fused_mode = getattr(config, "ngram_fused_mode", "average")
        self.ngram_hash_base = _ngram_hash_base(config.vocab_size, self.ngram_mod_force_prime)
        self._init_ngram_embeddings()
        self._vocab_mods_cache = None
        self.use_compressed_tokenizer = getattr(config, "ngram_compressed_tokenizer", False)

    def _init_ngram_embeddings(self) -> None:
        num_embedders = self.k * (self.n - 1)
        ngram_hidden_size = (
            self.config.ngram_embedding_hidden_size
            if self.config.ngram_embedding_hidden_size is not None
            else self.config.hidden_size
        )
        emb_dim = ngram_hidden_size // num_embedders
        embedders = []
        post_projs = []
        self._ngram_vocab_dims = _ngram_embedding_vocab_sizes(self.m, num_embedders, self.ngram_mod_force_prime)
        for vocab_size in self._ngram_vocab_dims:
            padded_vocab_size = (vocab_size + self.tp - 1) // self.tp * self.tp
            emb = nn.Embedding(padded_vocab_size, emb_dim, padding_idx=self.config.pad_token_id)
            proj = (
                nn.Linear(emb_dim, self.config.hidden_size, bias=False) if self.ngram_fused_mode == "average" else None
            )
            embedders.append(emb)
            if proj is not None:
                post_projs.append(proj)
        self.embedders = nn.ModuleList(embedders)
        if self.ngram_fused_mode == "concat":
            self.concat_proj = nn.Linear(emb_dim * num_embedders, self.config.hidden_size, bias=False)
            self.post_projs = nn.ModuleList()
        else:
            self.post_projs = nn.ModuleList(post_projs)

    def _shift_right_ignore_eos(
        self, tensor: torch.Tensor, n: int, eos_token_id: int = 2, eos_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        if eos_mask is None and eos_token_id is None:
            eos_mask = torch.zeros_like(tensor, dtype=torch.bool)
        elif eos_mask is None:
            eos_mask = tensor == eos_token_id
        else:
            eos_mask = eos_mask.to(device=tensor.device, dtype=torch.bool)
        for i in range(batch_size):
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            prev_idx = 0
            for eos_idx in eos_positions:
                end_idx = eos_idx.item() + 1
                if end_idx - prev_idx > n:
                    result[i, prev_idx + n : end_idx] = tensor[i, prev_idx : end_idx - n]
                prev_idx = end_idx
            if prev_idx < seq_len and seq_len - prev_idx > n:
                result[i, prev_idx + n : seq_len] = tensor[i, prev_idx : seq_len - n]
        return result

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = self._ngram_vocab_dims[index]
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = power_mod * self.ngram_hash_base % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[i, j] = mods
        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self, input_ids: torch.Tensor, shifted_ids: Dict[int, torch.Tensor], vocab_mods: List[int], ngram: int
    ) -> torch.Tensor:
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def _compress_input_ids(self, input_ids: torch.Tensor, lookup_table: torch.Tensor) -> torch.Tensor:
        pos_mask = input_ids >= 0
        out = input_ids.clone()
        valid_ids = input_ids[pos_mask]
        out[pos_mask] = lookup_table[valid_ids]
        return out

    def compute_ngram_embeddings(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None,
        lookup_table: Optional[torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        seq_len = input_ids.size(-1)
        if ngram_context is not None:
            context = torch.cat([ngram_context[..., -(self.n - 1) :], input_ids], dim=-1)
        else:
            context = input_ids
        device = self.word_embeddings.weight.device
        if self.use_compressed_tokenizer and lookup_table is not None:
            compressed_context = self._compress_input_ids(context, lookup_table)
        else:
            compressed_context = context
        vocab_mods = self._precompute_vocab_mods()
        shifted_ids = {}
        eos_mask = None if self.config.eos_token_id is None else context == self.config.eos_token_id
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                compressed_context, i - 1, eos_token_id=self.config.eos_token_id, eos_mask=eos_mask
            )
        if self.ngram_fused_mode == "average":
            x = torch.zeros(
                input_ids.shape[0],
                seq_len,
                self.config.hidden_size,
                device=device,
                dtype=self.word_embeddings.weight.dtype,
            )
        else:
            x = None
        ngram_embedding_parts = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = self._ngram_vocab_dims[index]
                ngram_ids = self._get_ngram_ids(compressed_context, shifted_ids, vocab_mods[i, j], ngram=i)
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                embedder_device = self.embedders[index].weight.device
                x_ngram = self.embedders[index](new_ids.to(embedder_device))
                if self.ngram_fused_mode == "concat":
                    ngram_embedding_parts.append(x_ngram.to(device))
                    continue
                proj_device = self.post_projs[index].weight.device
                x_proj = self.post_projs[index](x_ngram.to(proj_device))
                x = x + x_proj.to(x.device)
        if self.ngram_fused_mode == "concat":
            concat_device = self.concat_proj.weight.device
            x_concat = torch.cat(ngram_embedding_parts, dim=-1).to(concat_device)
            return self.concat_proj(x_concat).to(device)
        if average:
            x = x / (self.k * (self.n - 1))
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None,
        lookup_table: Optional[torch.Tensor] = None,
        return_ngram_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        x = self.word_embeddings(input_ids.to(self.word_embeddings.weight.device)).clone()
        ngram_embeddings = None
        if return_ngram_embeddings or not self.config.skip_ngram_for_input:
            ngram_embeddings = self.compute_ngram_embeddings(
                input_ids,
                ngram_context=ngram_context,
                lookup_table=lookup_table,
                average=self.ngram_fused_mode == "average",
            )
            if not self.config.skip_ngram_for_input:
                if self.ngram_fused_mode == "concat":
                    x = x + ngram_embeddings
                else:
                    x = (x + ngram_embeddings * (self.k * (self.n - 1))) / (1 + self.k * (self.n - 1))
        if return_ngram_embeddings:
            return (x, ngram_embeddings)
        return x


class NanbeigeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", self._compute_inv_freq(device), persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    def _compute_inv_freq(self, device=None):
        return 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim)

    def _apply(self, fn, recurse=True):
        inv_freq = self.inv_freq
        result = super()._apply(fn, recurse=recurse)
        if inv_freq.is_meta:
            self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        else:
            self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        return result

    @torch.no_grad()
    def forward(self, x, position_ids, inv_freq=None):
        inv_freq = self.inv_freq if inv_freq is None else inv_freq
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))


class NanbeigeLinearScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    def forward(self, x, position_ids):
        return super().forward(x, position_ids.float() / self.scaling_factor)


class NanbeigeDynamicNTKScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    def forward(self, x, position_ids):
        # Grown frequencies are per call (as in Transformers' dynamic_rope_update, which resets
        # below the original length); never persist them into the shared buffer.
        seq_len = torch.max(position_ids) + 1
        inv_freq = None
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
        return super().forward(x, position_ids, inv_freq=inv_freq)


class NanbeigeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NanbeigeAttention(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        if config.qk_layernorm:
            self.q_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None
        self._init_rope()

    def _init_rope(self):
        rope_scaling = self.config.rope_parameters
        scaling_type = None
        if isinstance(rope_scaling, dict):
            scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if rope_scaling is None or scaling_type in (None, "default"):
            self.rotary_emb = NanbeigeRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
            )
        else:
            scaling_factor = rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = NanbeigeLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = NanbeigeDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        loop_idx = kwargs.pop("loop_idx", 0)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.q_layernorm is not None:
            query_states = self.q_layernorm(query_states)
        if self.k_layernorm is not None:
            key_states = self.k_layernorm(key_states)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        use_loop_shared_kv = loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        skip_cache_update = use_loop_shared_kv and loop_share_kv_repeat_idx > 0
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and (not skip_cache_update):
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, _get_num_physical_layers(self.config), loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache, self.layer_idx, loop_share_kv_repeat_idx, key_states, value_states
            )
        else:
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache, self.layer_idx, loop_share_kv_repeat_idx, key_states, value_states
            )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim ** (-0.5),
            )
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        implementation = "eager" if output_attentions else self.config._attn_implementation
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            causal_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )
        if not output_attentions:
            attn_weights = None
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights, past_key_value)


class NanbeigeDecoderLayer(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.enable_hyper_connection = config.enable_hyper_connection
        self.layer_idx = layer_idx
        self._mhc_loop_middle_layer = (
            self._is_mhc_loop_middle_layer()
            if getattr(config, "enable_double_loop_split", False)
            or getattr(config, "mhc_diff_for_loop", False)
            or getattr(config, "mhc_double_stream_position_for_loop", None) is not None
            else False
        )
        self.num_residual_streams = self._get_layer_num_residual_streams()
        self.self_attn = NanbeigeAttention(config=config, layer_idx=layer_idx)
        self.mlp = NanbeigeMLP(config)
        self.input_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.enable_hyper_connection:
            self.self_attn_hc = NanbeigeHyperConnectionModule(
                config,
                layer_idx=layer_idx,
                module_name="self_attention",
                num_residual_streams=self.num_residual_streams,
            )
            self.mlp_hc = NanbeigeHyperConnectionModule(
                config, layer_idx=layer_idx, module_name="mlp", num_residual_streams=self.num_residual_streams
            )
            if getattr(config, "mhc_diff_for_loop", False) and self._mhc_loop_middle_layer:
                self.self_attn_mhc_loop_hcs = nn.ModuleList(
                    [
                        NanbeigeHyperConnectionModule(
                            config,
                            layer_idx=layer_idx,
                            module_name=f"self_attention_loop_{loop_idx}",
                            num_residual_streams=self.num_residual_streams,
                        )
                        for loop_idx in range(1, self._get_mhc_loop_count())
                    ]
                )
                self.mlp_mhc_loop_hcs = nn.ModuleList(
                    [
                        NanbeigeHyperConnectionModule(
                            config,
                            layer_idx=layer_idx,
                            module_name=f"mlp_loop_{loop_idx}",
                            num_residual_streams=self.num_residual_streams,
                        )
                        for loop_idx in range(1, self._get_mhc_loop_count())
                    ]
                )
            else:
                self.self_attn_mhc_loop_hcs = None
                self.mlp_mhc_loop_hcs = None
        else:
            self.self_attn_hc = None
            self.mlp_hc = None
            self.self_attn_mhc_loop_hcs = None
            self.mlp_mhc_loop_hcs = None

    def _get_mhc_loop_count(self) -> int:
        physical = _get_num_physical_layers(self.config)
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            if physical <= 0 or physical % 2 != 0:
                raise ValueError("mhc_diff_for_loop requires loop_middle_layers or even num_hidden_layers.")
            loop_middle_layers = physical // 2
        return (physical + loop_middle_layers) // loop_middle_layers

    def _get_mhc_loop_middle_bounds(self) -> Tuple[int, int]:
        physical = _get_num_physical_layers(self.config)
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            if physical <= 0 or physical % 2 != 0:
                raise ValueError("mhc_diff_for_loop requires loop_middle_layers or even num_hidden_layers.")
            loop_middle_layers = physical // 2
        first_unlooped_layers = (physical - loop_middle_layers) // 2
        return (first_unlooped_layers, first_unlooped_layers + loop_middle_layers)

    def _is_mhc_loop_middle_layer(self) -> bool:
        middle_start, middle_end = self._get_mhc_loop_middle_bounds()
        return middle_start <= self.layer_idx < middle_end

    def _get_layer_num_residual_streams(self) -> int:
        num_residual_streams = self.config.num_residual_streams
        double_stream_position = getattr(self.config, "mhc_double_stream_position_for_loop", None)
        if double_stream_position is None:
            return num_residual_streams
        is_middle_layer = self._mhc_loop_middle_layer
        if (
            double_stream_position == "mid"
            and is_middle_layer
            or (double_stream_position == "edge" and (not is_middle_layer))
        ):
            return num_residual_streams * 2
        return num_residual_streams

    def get_num_residual_streams(self) -> int:
        return self.num_residual_streams

    def _get_self_attn_hc(self, mhc_loop_idx: Optional[int] = None) -> Optional[NanbeigeHyperConnectionModule]:
        if mhc_loop_idx is not None and self.self_attn_mhc_loop_hcs is not None and (mhc_loop_idx > 0):
            return self.self_attn_mhc_loop_hcs[mhc_loop_idx - 1]
        return self.self_attn_hc

    def _get_mlp_hc(self, mhc_loop_idx: Optional[int] = None) -> Optional[NanbeigeHyperConnectionModule]:
        if mhc_loop_idx is not None and self.mlp_mhc_loop_hcs is not None and (mhc_loop_idx > 0):
            return self.mlp_mhc_loop_hcs[mhc_loop_idx - 1]
        return self.mlp_hc

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        loop_idx: int = 0,
        loop_cache_layer_idx: Optional[int] = None,
        mhc_loop_idx: Optional[int] = None,
        loop_share_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if self.enable_hyper_connection:
            self_attn_hc = self._get_self_attn_hc(mhc_loop_idx)
            hidden_states, h_res, h_post = self_attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            loop_idx=loop_idx,
            loop_cache_layer_idx=loop_cache_layer_idx,
            loop_share_kv_cache=loop_share_kv_cache,
            loop_share_kv_repeat_idx=mhc_loop_idx,
            depth_attention_kv_cache=depth_attention_kv_cache,
            **kwargs,
        )
        if self.enable_hyper_connection:
            hidden_states = self_attn_hc.fuse_residual(h_res, residual, h_post, hidden_states)
        else:
            hidden_states = residual + hidden_states
        residual = hidden_states
        if self.enable_hyper_connection:
            mlp_hc = self._get_mlp_hc(mhc_loop_idx)
            hidden_states, h_res, h_post = mlp_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_hyper_connection:
            hidden_states = mlp_hc.fuse_residual(h_res, residual, h_post, hidden_states)
        else:
            hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class NanbeigePreTrainedModel(PreTrainedModel):
    config_class = NanbeigeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanbeigeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = False

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    @torch.no_grad()
    def _init_weights(self, module):
        # transformers.initialization skips tensors already loaded from the checkpoint, so a
        # partially missing module never re-initializes its loaded siblings.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                module.weight[module.padding_idx].zero_()
        elif isinstance(module, NanbeigeRMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, NanbeigeRotaryEmbedding):
            init.copy_(module.inv_freq, module._compute_inv_freq(module.inv_freq.device))
        elif isinstance(module, NanbeigeHyperConnectionModule):
            for name in ("alpha_pre", "alpha_post", "alpha_res"):
                init.constant_(getattr(module, name), self.config.mhc_init_gating_factor)
            module._init_static_mappings()
            module._init_dynamic_zero()


class NanbeigeModel(NanbeigePreTrainedModel):
    def __init__(self, config: NanbeigeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if (
            config.emb_neighbor_num is not None
            and config.emb_split_num is not None
            and (config.ngram_vocab_size_ratio is not None)
        ):
            self.ngram_embeddings = NanbeigeNgramEmbedding(config, self.embed_tokens)
            use_compressed_tokenizer = getattr(config, "ngram_compressed_tokenizer", False)
            if use_compressed_tokenizer:
                lookup_table = torch.arange(config.vocab_size, dtype=torch.long)
                self.register_buffer("lookup_table", lookup_table, persistent=True)
            else:
                self.lookup_table = None
        else:
            self.ngram_embeddings = None
            self.lookup_table = None
        physical = config.num_hidden_layers
        self.layers = nn.ModuleList([NanbeigeDecoderLayer(config, layer_idx) for layer_idx in range(physical)])
        self.ngram_layer_fusion = nn.ModuleDict(
            {
                str(layer_idx): NanbeigeNgramLayerFusion(config)
                for layer_idx in (
                    range(physical)
                    if getattr(config, "ngram_insert_all_layers", False)
                    else getattr(config, "insert_ngram_layer_idx", [])
                )
            }
        )
        self.norm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_num_loops(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return 1
        loop_weights = getattr(self.config, "loop_loss_weights", [])
        if loop_weights is not None and len(loop_weights) > 0:
            return len(loop_weights) + 1
        return getattr(self.config, "num_loops", 1)

    def _get_layer_order(self) -> List[int]:
        if getattr(self.config, "enable_double_loop_split", False):
            return _get_double_loop_split_layer_order(
                _get_num_physical_layers(self.config), getattr(self.config, "loop_middle_layers", None)
            )
        return list(range(len(self.layers)))

    def _get_layer_execution_order(self) -> List[Tuple[int, Optional[int]]]:
        if getattr(self.config, "enable_double_loop_split", False):
            return _get_double_loop_split_layer_order_with_mhc_loop_indices(
                _get_num_physical_layers(self.config), getattr(self.config, "loop_middle_layers", None)
            )
        return [(layer_idx, None) for layer_idx in range(len(self.layers))]

    def _get_layer_num_residual_streams(self, layer_idx: int) -> int:
        layer = self.layers[layer_idx]
        if hasattr(layer, "get_num_residual_streams"):
            return layer.get_num_residual_streams()
        return self.config.num_residual_streams

    def _convert_hyper_connection_streams(self, hidden_states: torch.Tensor, target_layer_idx: int) -> torch.Tensor:
        return NanbeigeHyperConnectionModule.convert_stream_count(
            hidden_states, self.config.hidden_size, self._get_layer_num_residual_streams(target_layer_idx)
        )

    def _contract_hyper_connection_streams(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n_hidden_size = hidden_states.shape[-1]
        if n_hidden_size == self.config.hidden_size:
            return hidden_states
        if n_hidden_size % self.config.hidden_size != 0:
            raise RuntimeError(
                f"HC output_contract shape mismatch: hidden={n_hidden_size}, base_hidden={self.config.hidden_size}"
            )
        return NanbeigeHyperConnectionModule.output_contract(hidden_states, n_hidden_size // self.config.hidden_size)

    def _get_cache_seq_length(self, past_key_values: Optional[Cache]) -> int:
        if past_key_values is None:
            return 0
        max_seq_length = 0
        for loop_idx in range(self._get_num_loops()):
            layer_idx = loop_idx * _get_num_physical_layers(self.config)
            max_seq_length = max(max_seq_length, past_key_values.get_seq_length(layer_idx))
        return max_seq_length

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        num_loops = self._get_num_loops()
        double_loop_split = getattr(self.config, "enable_double_loop_split", False)
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise ValueError("Nanbeige requires a Transformers DynamicCache, not legacy/static/quantized caches.")
        if past_key_values is not None and not use_cache:
            raise ValueError("use_cache must be enabled when past_key_values is supplied.")
        input_tensor = input_ids if inputs_embeds is None else inputs_embeds
        batch_size, sequence_length = input_tensor.shape[:2]
        past_length = self._get_cache_seq_length(past_key_values)
        expected_position = torch.arange(past_length, past_length + sequence_length, device=input_tensor.device)
        if cache_position is not None and not torch.equal(cache_position, expected_position):
            raise ValueError("cache_position must describe the contiguous append to the dynamic cache.")
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape != (batch_size, past_length + sequence_length):
                raise ValueError("attention_mask must cover the full cached and new token sequence.")
            if attention_mask.ndim not in (2, 4):
                raise ValueError("attention_mask must be a 2D padding or 4D attention mask.")
        if self.ngram_embeddings is not None and inputs_embeds is not None:
            raise ValueError("N-gram hidden-state inputs cannot preserve token history; use input_ids.")
        if (
            self.ngram_embeddings is not None
            and past_key_values is not None
            and not isinstance(past_key_values, NgramCache)
        ):
            if past_length:
                raise ValueError("N-gram cache has no key/value history with corresponding token context.")
            past_key_values = NgramCache(self.config)
        if past_length and self.ngram_embeddings is not None and past_key_values.ngram_context is None:
            raise ValueError("N-gram cache has no token history.")
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(self.config) if self.ngram_embeddings is not None else DynamicCache()
        if use_cache and self.config.enable_depth_attention and self.config.loop_share_kv:
            raise ValueError("enable_depth_attention with loop_share_kv does not support use_cache=True/generation.")
        ngram_context = None
        if self.ngram_embeddings is not None and isinstance(past_key_values, NgramCache):
            ngram_context = past_key_values.ngram_context
        ngram_layer_embeddings = None
        if inputs_embeds is None:
            if self.ngram_embeddings is not None:
                if len(self.ngram_layer_fusion) > 0:
                    inputs_embeds, ngram_layer_embeddings = self.ngram_embeddings(
                        input_ids,
                        ngram_context=ngram_context,
                        lookup_table=self.lookup_table,
                        return_ngram_embeddings=True,
                    )
                else:
                    inputs_embeds = self.ngram_embeddings(
                        input_ids, ngram_context=ngram_context, lookup_table=self.lookup_table
                    )
            else:
                inputs_embeds = self.embed_tokens(input_ids)
        if self.ngram_embeddings is not None and len(self.ngram_layer_fusion) > 0 and (ngram_layer_embeddings is None):
            raise RuntimeError("N-gram layer fusion requires input_ids and ngram embeddings.")
        if use_cache and isinstance(past_key_values, NgramCache) and (input_ids is not None):
            past_key_values.update_ngram_context(input_ids)
        if cache_position is None:
            past_seen_tokens = self._get_cache_seq_length(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        last_loop_all_hidden_states = None
        last_loop_all_self_attns = None
        last_loop_next_decoder_cache = None
        layer_order = self._get_layer_execution_order()
        layer_lookup = list(self.layers)
        loop_share_kv_cache = {} if getattr(self.config, "loop_share_kv", False) else None
        if loop_share_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("loop_share_kv does not support gradient checkpointing during training.")
        depth_attention_kv_cache = [] if getattr(self.config, "enable_depth_attention", False) else None
        if depth_attention_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("enable_depth_attention does not support gradient checkpointing during training.")
        for loop_idx in range(num_loops):
            current_loop_all_hidden_states = () if output_hidden_states else None
            current_loop_all_self_attns = () if output_attentions else None
            current_loop_next_decoder_cache = None
            if self.config.enable_hyper_connection and len(self.layers) > 0:
                hidden_states = self._convert_hyper_connection_streams(hidden_states, 0)
            for execution_idx, (layer_idx, mhc_loop_idx) in enumerate(layer_order):
                decoder_layer = layer_lookup[layer_idx]
                if self.config.enable_hyper_connection:
                    hidden_states = self._convert_hyper_connection_streams(hidden_states, layer_idx)
                if output_hidden_states:
                    if self.config.enable_hyper_connection:
                        current_loop_all_hidden_states += (self._contract_hyper_connection_streams(hidden_states),)
                    else:
                        current_loop_all_hidden_states += (hidden_states,)
                fusion_key = str(layer_idx)
                fusion = self.ngram_layer_fusion[fusion_key] if fusion_key in self.ngram_layer_fusion else None
                if fusion is not None:
                    if ngram_layer_embeddings is None:
                        raise RuntimeError("N-gram layer fusion requires input_ids and ngram embeddings.")
                    if self.config.enable_hyper_connection:
                        contracted = self._contract_hyper_connection_streams(hidden_states)
                        contracted = fusion(contracted, ngram_layer_embeddings)
                        hidden_states = self._convert_hyper_connection_streams(contracted, layer_idx)
                    else:
                        hidden_states = fusion(hidden_states, ngram_layer_embeddings)
                if self.gradient_checkpointing and self.training:
                    cache_layer_idx = (
                        (layer_idx if loop_share_kv_cache is not None else execution_idx)
                        if double_loop_split
                        else None
                    )
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        loop_idx,
                        cache_layer_idx,
                        mhc_loop_idx,
                        loop_share_kv_cache,
                        depth_attention_kv_cache,
                    )
                else:
                    cache_layer_idx = (
                        (layer_idx if loop_share_kv_cache is not None else execution_idx)
                        if double_loop_split
                        else None
                    )
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        loop_idx=loop_idx,
                        loop_cache_layer_idx=cache_layer_idx,
                        mhc_loop_idx=mhc_loop_idx,
                        loop_share_kv_cache=loop_share_kv_cache,
                        depth_attention_kv_cache=depth_attention_kv_cache,
                    )
                hidden_states = layer_outputs[0]
                if use_cache:
                    current_loop_next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    current_loop_all_self_attns += (layer_outputs[1],)
            if self.config.enable_hyper_connection:
                hidden_states = self._contract_hyper_connection_streams(hidden_states)
            if not getattr(self.config, "skip_loop_final_norm", False):
                hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                current_loop_all_hidden_states += (hidden_states,)
            last_loop_all_hidden_states = current_loop_all_hidden_states
            last_loop_all_self_attns = current_loop_all_self_attns
            last_loop_next_decoder_cache = current_loop_next_decoder_cache
        if getattr(self.config, "skip_loop_final_norm", False):
            hidden_states = self.norm(hidden_states)
            if output_hidden_states and last_loop_all_hidden_states is not None:
                last_loop_all_hidden_states = last_loop_all_hidden_states[:-1] + (hidden_states,)
        next_cache = last_loop_next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                (
                    v
                    for v in [hidden_states, next_cache, last_loop_all_hidden_states, last_loop_all_self_attns]
                    if v is not None
                )
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=last_loop_all_hidden_states,
            attentions=last_loop_all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        return create_causal_mask(
            config=self.config,
            inputs_embeds=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            allow_is_causal_skip=not output_attentions,
        )


class NanbeigeForCausalLM(NanbeigePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight",
        "model.ngram_embeddings.word_embeddings.weight": "model.embed_tokens.weight",
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = NanbeigeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, indices, :])
        logits = logits.float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, *args, **kwargs):
        generation_config = kwargs.get("generation_config", args[1] if len(args) > 1 else self.generation_config)
        if kwargs.get("custom_generate") is not None:
            raise ValueError("Nanbeige uses package-owned generation; custom_generate is not supported.")
        if kwargs.get("use_cache", generation_config.use_cache):
            if kwargs.get("cache_implementation", generation_config.cache_implementation) is not None:
                raise ValueError("Nanbeige generation requires the default DynamicCache.")
            if kwargs.get("past_key_values") is None:
                kwargs["past_key_values"] = (
                    NgramCache(self.config) if self.model.ngram_embeddings is not None else DynamicCache()
                )
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        **kwargs,
    ):
        if next_sequence_length is None and past_key_values is not None:
            past_length = self.model._get_cache_seq_length(past_key_values)
            next_sequence_length = (
                input_ids.shape[-1] - past_length if input_ids.shape[-1] > past_length else input_ids.shape[-1]
            )
        prepared = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        tensor = prepared.get("input_ids")
        sequence_length = tensor.shape[-1] if tensor is not None else prepared["inputs_embeds"].shape[-2]
        if prepared.get("cache_position") is not None:
            prepared["cache_position"] = prepared["cache_position"][-sequence_length:]
        return prepared
