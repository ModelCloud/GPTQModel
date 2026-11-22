# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# The math is ported from DeepSeek-V4.1-Flash ``inference/model.py`` at
# 314113f8a7dadb388215396ba5fbee2c610a04d6, Copyright (c) 2023 DeepSeek, MIT License
# (see LICENSE in this directory). Module names follow the GPT-QModel V4.1 runtime
# contract (``gptqmodel/utils/v41_source.py::source_layer_key`` maps them to the
# released checkpoint keys).
"""DeepSeek V4.1 text decoder and dSpark draft stages on the Transformers 5.23 ABI.

Every equation follows the reference ``inference/model.py``:

* latent attention with a grouped low-rank output projection, interleaved partial
  RoPE (inverse-rotated on the output) and per-head attention sinks;
* a sliding window of raw KV on every layer, plus -- on layers with
  ``compress_ratio > 0`` -- compressed KV produced by the layer's KV source
  (softmax-gated pooling of ``compress_ratio`` tokens) and selected per query by
  the Lightning indexer of the layer's index source, optionally pre-filtered by
  the candidate source's block selection;
* manifold-constrained hyper-connections (``hc_mult`` residual streams, Sinkhorn
  ``comb``); the ``pre`` mix a sublayer computes collapses the *next* sublayer's
  input, exactly as the reference block chain;
* top-k MoE with ``sqrtsoftplus`` scores, a selection-only correction bias (a
  separate one for image tokens), normalized and scaled weights, clamped SwiGLU
  experts and one shared expert;
* Engram (n-gram PLE) injection into the residual streams before a layer;
* the dSpark draft stages (``mtp.*``), their Markov and confidence heads.

Deliberate differences from the reference, all outside the model math:

* weights are dense (BF16/FP32). The reference's FP8/FP4 GEMMs and the in-place
  activation fake-quantization of its kernels (``act_quant``/``fp4_act_quant`` on
  KV, indexer keys/queries and attention outputs) are not simulated; the native
  helpers in ``gptqmodel/utils/v41_native.py`` run those kernels when supplied;
* attention is a pure PyTorch gather/softmax equal to ``kernel.sparse_attn``;
* caches are a 5.23 ``Cache`` (``DeepseekV41Cache``) instead of fixed-size
  module buffers, so prefill, decode and multi-token continuation all use the
  per-query causal visibility the reference uses for prefill and decode;
* padding is rejected: the reference has no padded-batch semantics for the
  compressor, indexer or Engram hash.

The vision tower, aligner and image-span embeddings are not ported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Unpack

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicSlidingWindowLayer
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel, slice_logits_to_keep
from transformers.utils import ModelOutput, TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .configuration_deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig


# --------------------------------------------------------------------------------------------
# normalization, rotary embedding, sparse attention primitives
# --------------------------------------------------------------------------------------------
class DeepseekV41RMSNorm(nn.Module):
    """FP32 statistics; the weight multiplies the FP32 value before the single rounding."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * value).to(dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class DeepseekV41UnweightedRMSNorm(nn.Module):
    """The weight-free RMS statistic of the hyper-connection input."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + self.eps)


def reference_inv_freq(dim: int, params: dict) -> torch.Tensor:
    """Reference ``precompute_freqs_cis`` frequencies (YaRN without magnitude correction)."""

    base = params["rope_theta"]
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if params["rope_type"] == "yarn":
        original = params["original_max_position_embeddings"]
        factor = params["factor"]

        def corrected_dim(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(params.get("beta_fast", 32))), 0)
        high = min(math.ceil(corrected_dim(params.get("beta_slow", 1))), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


class DeepseekV41RotaryEmbedding(nn.Module):
    """Per-family (``main``/``compress``) interleaved RoPE coefficients.

    Returns FP32 ``(cos, sin)`` of shape ``[..., qk_rope_head_dim // 2]``: one angle
    per adjacent channel pair, as the reference's complex ``freqs_cis``.
    """

    layer_types = ("main", "compress")

    def __init__(self, config: DeepseekV41TextConfig, device=None):
        super().__init__()
        self.config = config
        self.rope_dim = config.qk_rope_head_dim
        self.rope_type = {}
        for layer_type in self.layer_types:
            params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = params["rope_type"]
            inv_freq = reference_inv_freq(self.rope_dim, params).to(device)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            # The reference applies no YaRN attention factor.
            setattr(self, f"{layer_type}_attention_scaling", 1.0)

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type=None):
        if layer_type not in self.layer_types:
            raise ValueError(f"unknown DeepSeek V4.1 rope family: {layer_type}")
        inv = getattr(self, f"{layer_type}_inv_freq").to(device=x.device, dtype=torch.float32)
        angles = position_ids.to(x.device).float().unsqueeze(-1) * inv
        coeff = torch.polar(torch.ones_like(angles), angles)
        factor = getattr(self, f"{layer_type}_attention_scaling")
        return coeff.real * factor, coeff.imag * factor


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels as complex pairs; one final rounding."""

    width = cos.shape[-1] * 2
    pairs = torch.view_as_complex(x[..., -width:].float().reshape(*x.shape[:-1], -1, 2))
    coeff = torch.complex(cos.float(), -sin.float() if inverse else sin.float())
    if x.ndim == 4:
        coeff = coeff.unsqueeze(2)
    rotated = torch.view_as_real(pairs * coeff).flatten(-2).to(x.dtype)
    return torch.cat([x[..., :-width], rotated], dim=-1)


def sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, sinks: torch.Tensor, indices: torch.Tensor, scaling: float
) -> torch.Tensor:
    """``kernel.sparse_attn``: each query attends to ``kv[indices]`` (``-1`` = none) plus a sink.

    q: ``[B, M, H, D]``, kv: ``[B, N, D]``, sinks: ``[H]``, indices: ``[B, M, K]``. The sink
    joins the softmax denominator only; a query with no valid index returns zeros.
    """

    batch, length, heads, dim = q.shape
    width = indices.shape[-1]
    out = torch.empty_like(q)
    if width == 0:
        return out.zero_()
    rows = torch.arange(batch, device=kv.device).view(batch, 1, 1)
    # Bound the gathered FP32 keys to ~64 MiB per chunk.
    chunk = max(1, min(length, (64 << 20) // max(1, batch * width * dim * 4)))
    sink = sinks.float().view(1, 1, heads, 1)
    for start in range(0, length, chunk):
        index = indices[:, start : start + chunk].to(device=kv.device, dtype=torch.long)
        valid = index >= 0
        keys = kv[rows, index.clamp_min(0)].float()
        scores = torch.einsum("bmhd,bmkd->bmhk", q[:, start : start + chunk].float(), keys) * scaling
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        logits = torch.cat([scores, sink.expand(*scores.shape[:-1], 1)], dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., :width]
        out[:, start : start + chunk] = torch.einsum("bmhk,bmkd->bmhd", probs, keys).to(q.dtype)
    return out


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the ``topk_blocks`` best blocks per query.

    Unreachable positions are already ``-inf`` in ``logits``. The block holding a
    query's newest compressed position is always kept. Returns a mask shaped like
    ``logits``.
    """

    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def dspark_joint_ring(main_kv, anchors, window_size):
    """Gather independent reference-order rings from one already-projected prefix."""

    slots = torch.arange(window_size, device=main_kv.device)
    history = anchors[:, None] - ((anchors[:, None] - slots) % window_size)
    valid = history >= 0
    window = main_kv[0, history.clamp_min(0)].masked_fill(~valid[..., None], 0)
    indices = slots[None].expand(anchors.numel(), -1).masked_fill(~valid, -1)
    return window, indices


# --------------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------------
class DeepseekV41CacheLayer(DynamicSlidingWindowLayer):
    """One decoder layer's state.

    ``keys`` holds the last ``sliding_window - 1`` window KV rows (``[B, 1, T, D]``,
    ``values`` aliases it). A KV source additionally keeps its compressed history
    (RoPE applied), the raw FP32 KV/score rows of an incomplete compression group
    and, as an indexer key owner, its index keys.
    """

    def __init__(self, sliding_window: int, **kwargs):
        super().__init__(sliding_window=sliding_window)
        self.compressed_kv: torch.Tensor | None = None
        self.index_keys: torch.Tensor | None = None
        self.buffer_kv: torch.Tensor | None = None
        self.buffer_score: torch.Tensor | None = None

    def update_window(self, kv: torch.Tensor) -> torch.Tensor:
        """Append ``[B, L, D]`` window KV; return the visible rows ``[B, past + L, D]``."""

        rows = kv.unsqueeze(1)
        if not self.is_initialized:
            self.lazy_initialization(rows, rows)
        self.cumulative_length += kv.shape[1]
        full = torch.cat([self.keys, rows], dim=-2)
        keep = self.sliding_window - 1
        self.keys = full[:, :, full.shape[-2] - keep :] if keep > 0 else full[:, :, :0]
        self.values = self.keys
        return full[:, 0]

    @property
    def entry_count(self) -> int:
        return 0 if self.compressed_kv is None else self.compressed_kv.shape[1]

    def update_compressed(self, latent: torch.Tensor | None) -> torch.Tensor | None:
        if latent is not None and latent.shape[1]:
            self.compressed_kv = latent if self.compressed_kv is None else torch.cat([self.compressed_kv, latent], 1)
        return self.compressed_kv

    def update_index_keys(self, keys: torch.Tensor | None) -> torch.Tensor | None:
        if keys is not None and keys.shape[1]:
            self.index_keys = keys if self.index_keys is None else torch.cat([self.index_keys, keys], 1)
        return self.index_keys

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        self.values = self.keys
        for name in ("compressed_kv", "index_keys", "buffer_kv", "buffer_score"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, beam_idx.to(value.device)))

    def reset(self) -> None:
        super().reset()
        self.compressed_kv = self.index_keys = self.buffer_kv = self.buffer_score = None

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("DeepSeek V4.1 compressor state cannot be rolled back")


class DeepseekV41Cache(Cache):
    """Per-layer window/compressor/indexer state plus the trailing Engram n-gram ids."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(layers=[DeepseekV41CacheLayer(config.sliding_window) for _ in range(config.num_hidden_layers)])
        self.engram_history: torch.Tensor | None = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.engram_history is not None:
            self.engram_history = self.engram_history.index_select(0, beam_idx.to(self.engram_history.device))

    def reset(self) -> None:
        super().reset()
        self.engram_history = None


# --------------------------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------------------------
class DeepseekV41GroupedLinear(nn.Linear):
    """Block-diagonal output projection: group ``g`` of ``[..., G, in]`` uses rows ``g*R:(g+1)*R``."""

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(self.n_groups, -1, x.shape[-1])
        return torch.einsum("...gd,grd->...gr", x, weight)


class DeepseekV41Compressor(nn.Module):
    """Pools every ``compress_ratio`` tokens into one KV latent (before RoPE).

    Ratio 1 is a plain normalized projection. Above ratio 1 the projections and
    the softmax pooling run in FP32, as in the reference. Returns
    ``(latent, group_positions, counts, lengths)``: the newly completed latents
    ``[B, G, D]`` (``None`` when no group completed), the position of each group's
    first token, the total compressed entries per row and, per query, the number
    of tokens seen so far.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.compress_ratio > 1 else None
        )
        self.kv_norm = DeepseekV41RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states, cache_layer=None, position_ids=None, padding_mask=None):
        if padding_mask is not None:
            raise ValueError("DeepSeek V4.1 compression has no padded-batch semantics")
        if position_ids is None:
            raise ValueError("DeepSeek V4.1 compression needs position_ids")
        batch = hidden_states.shape[0]
        ratio, dtype = self.compress_ratio, hidden_states.dtype
        first_entry = cache_layer.entry_count if cache_layer is not None else 0
        if ratio == 1:
            latent = self.kv_norm(self.kv_proj(hidden_states))
        else:
            x = hidden_states.float()
            kv = F.linear(x, self.kv_proj.weight.float())
            score = F.linear(x, self.gate_proj.weight.float())
            if cache_layer is not None and cache_layer.buffer_kv is not None:
                kv = torch.cat([cache_layer.buffer_kv, kv], dim=1)
                score = torch.cat([cache_layer.buffer_score, score], dim=1)
            usable = kv.shape[1] // ratio * ratio
            if cache_layer is not None:
                cache_layer.buffer_kv, cache_layer.buffer_score = kv[:, usable:], score[:, usable:]
            kv = kv[:, :usable].unflatten(1, (-1, ratio))
            score = score[:, :usable].unflatten(1, (-1, ratio))
            latent = (kv * score.softmax(dim=2)).sum(dim=2)
            latent = self.kv_norm(latent.to(dtype))
        groups = latent.shape[1]
        counts = torch.full((batch,), first_entry + groups, dtype=torch.long, device=hidden_states.device)
        lengths = position_ids.to(hidden_states.device) + 1
        if not groups:
            return None, None, counts, lengths
        group_positions = (first_entry + torch.arange(groups, device=hidden_states.device)) * ratio
        return latent, group_positions.expand(batch, -1), counts, lengths


class DeepseekV41Indexer(nn.Module):
    """Lightning indexer: keeps the ``index_topk`` best compressed positions per query.

    Publishes ``shared["topk_idx"]`` (compressed entry ids or ``-1``) and, on the
    candidate source, ``shared["candidates"]``. An index-key owner (a KV source)
    also publishes ``shared["index_k"]`` (``[B, 1, N, index_head_dim]``).
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.heads_scaling = self.num_heads**-0.5
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.owns_k:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    def forward(self, hidden_states, q_residual, latent, group_positions, position_ids, cache_layer, shared):
        batch, length, _ = hidden_states.shape
        if self.owns_k:
            keys = None
            if latent is not None:
                keys = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary_emb(keys, group_positions, "compress")
                keys = apply_rotary_emb(keys, cos, sin)
            if cache_layer is not None:
                keys = cache_layer.update_index_keys(keys)
            shared["index_k"] = None if keys is None else keys.unsqueeze(1)
        keys = shared.get("index_k")
        if keys is None or keys.shape[2] == 0:
            shared["topk_idx"] = None
            if self.is_candidate_source:
                shared["candidates"] = None
            return
        keys = keys[:, 0].to(hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids, "compress")
        query = self.q_b_proj(q_residual).view(batch, length, self.num_heads, self.head_dim)
        query = apply_rotary_emb(query, cos, sin)
        weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.heads_scaling)
        lengths = shared["compress_lens"].to(keys.device).unsqueeze(-1)
        scores = torch.einsum("bshd,btd->bsht", query, keys)
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        entries = torch.arange(keys.shape[1], device=keys.device)
        scores = scores.masked_fill(entries >= lengths, -torch.inf)
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                scores, lengths, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            candidates = shared.get("candidates")
            if candidates is not None:
                scores = scores.masked_fill(~candidates.to(scores.device), -torch.inf)
        top_k = min(self.index_topk, keys.shape[1])
        indices = scores.topk(top_k, dim=-1, sorted=False).indices.sort(dim=-1).values
        shared["topk_idx"] = torch.where(indices < lengths, indices, -1)


class DeepseekV41Attention(nn.Module):
    """Sliding-window latent attention, plus selected compressed KV when ``compress_ratio > 0``.

    ``forward(hidden_states, shared, position_embeddings, ...)`` returns ``(output, None)``.
    ``shared`` carries the compressed KV / index keys / selections from the layers that
    produce them to the layers that reuse them, within one forward.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.rope_layer_type = "compress" if self.compress_ratio else "main"

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV41RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_a_proj = DeepseekV41GroupedLinear(
            self.num_heads * self.head_dim // self.o_groups, self.o_groups * self.o_lora_rank, self.o_groups
        )
        self.o_b_proj = nn.Linear(self.o_groups * self.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        self.compress_rotary = DeepseekV41RotaryEmbedding(config) if self.compress_ratio else None

    def _project_output(self, output, cos, sin):
        batch, length = output.shape[:2]
        output = apply_rotary_emb(output, cos, sin, inverse=True)
        grouped = output.reshape(batch, length, self.o_groups, -1)
        return self.o_b_proj(self.o_a_proj(grouped).flatten(2))

    def forward(
        self,
        hidden_states,
        shared=None,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        padding_mask=None,
        **kwargs,
    ):
        if padding_mask is not None:
            raise ValueError("DeepSeek V4.1 attention has no padded-batch semantics")
        shared = {} if shared is None else shared
        batch, length, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_layer_type]
        qr = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(qr).view(batch, length, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, cos, sin)
        kv = apply_rotary_emb(self.kv_norm(self.kv_proj(hidden_states)), cos, sin)

        cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
        window = cache_layer.update_window(kv) if cache_layer is not None else kv
        past = window.shape[1] - length
        width = min(self.sliding_window, window.shape[1])
        end = past + torch.arange(length, device=kv.device).unsqueeze(-1)
        indices = (end - self.sliding_window + 1).clamp_min(0) + torch.arange(width, device=kv.device)
        indices = indices.expand(batch, -1, -1)
        valid = indices <= end
        if attention_mask is not None:
            mask = attention_mask.expand(batch, -1, -1, -1)[:, 0, -length:]
            allowed = mask.gather(-1, indices.clamp_max(mask.shape[-1] - 1))
            valid = valid & (allowed if allowed.dtype == torch.bool else allowed == 0)
        indices = indices.masked_fill(~valid, -1)
        bank = window

        if self.compress_ratio:
            latent = group_positions = None
            if self.is_kv_source:
                latent, group_positions, counts, lengths = self.compressor(
                    hidden_states, cache_layer, position_ids, None
                )
                shared["group_counts"] = counts
                shared["compress_lens"] = lengths // self.compress_ratio
            if self.is_index_source:
                self.indexer(hidden_states, qr, latent, group_positions, position_ids, cache_layer, shared)
            if self.is_kv_source:
                rotated = None
                if latent is not None:
                    group_cos, group_sin = self.compress_rotary(latent, group_positions, "compress")
                    rotated = apply_rotary_emb(latent, group_cos, group_sin)
                if cache_layer is not None:
                    rotated = cache_layer.update_compressed(rotated)
                shared["compress_kv"] = None if rotated is None else rotated.unsqueeze(1)
            compressed, selected = shared.get("compress_kv"), shared.get("topk_idx")
            if compressed is not None and selected is not None and selected.shape[-1]:
                selected = selected.to(kv.device)
                extra = torch.where(selected >= 0, selected + bank.shape[1], -1)
                indices = torch.cat([indices, extra], dim=-1)
                bank = torch.cat([bank, compressed[:, 0].to(kv.device)], dim=1)

        output = sparse_attention(q, bank, self.sinks, indices, self.scaling)
        return self._project_output(output, cos, sin), None


class DeepseekV41DSparkAttention(DeepseekV41Attention):
    """dSpark draft attention over an explicit main-history window and all drafts.

    ``main_x`` holds the projected main features at absolute positions ``0..P``; the
    drafts sit at ``P + 1 ..``. Equivalent to the reference ``DSparkAttention`` after
    its prefill seeded the window ring with ``main_x[:, :P]`` and the decode step
    added position ``P``. With ``anchor_positions`` (one prompt, ``main_x`` batch 1)
    every batch row is an independent anchor ``P = anchor`` over the shared prefix.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.compress_ratio:
            raise ValueError("dSpark stages attend over a sliding window only")
        self.compress_rotary = DeepseekV41RotaryEmbedding(config)

    def forward(self, hidden_states, shared=None, *, main_x, anchor_positions=None, **kwargs):
        if kwargs.get("past_key_values") is not None:
            raise ValueError("dSpark replay requires explicit history and no decode cache")
        batch, drafts, _ = hidden_states.shape
        joint = anchor_positions is not None
        if main_x.ndim != 3 or main_x.shape[0] != (1 if joint else batch) or main_x.shape[1] < 2:
            raise ValueError("dSpark needs matching batch and at least two main-history positions")
        length, win = main_x.shape[1], self.sliding_window
        if joint and (
            anchor_positions.shape != (batch,)
            or anchor_positions.dtype != torch.long
            or anchor_positions.min() < 1
            or anchor_positions.max() >= length
            or not torch.all(anchor_positions[1:] > anchor_positions[:-1])
        ):
            raise ValueError("invalid joint dSpark anchor positions")
        positions = torch.arange(0 if joint else max(0, length - win), length, device=hidden_states.device)[None]
        main = main_x if joint else main_x[:, -win:]
        cos, sin = self.compress_rotary(main, position_ids=positions, layer_type="main")
        main_kv = apply_rotary_emb(self.kv_norm(self.kv_proj(main)), cos, sin)
        if joint:
            anchors = anchor_positions.to(hidden_states.device)
            window, main_indices = dspark_joint_ring(main_kv, anchors, win)
            draft_positions = anchors[:, None] + 1 + torch.arange(drafts, device=hidden_states.device)
        else:
            window = main_kv.new_zeros(batch, win, self.head_dim)
            window.index_copy_(1, positions[0] % win, main_kv)
            draft_positions = torch.arange(length, length + drafts, device=hidden_states.device)[None]
        cos, sin = self.compress_rotary(hidden_states, position_ids=draft_positions, layer_type="main")
        qr = self.q_a_norm(self.q_a_proj(hidden_states))
        q = apply_rotary_emb(self.q_b_proj(qr).view(batch, drafts, self.num_heads, self.head_dim), cos, sin)
        kv = apply_rotary_emb(self.kv_norm(self.kv_proj(hidden_states)), cos, sin)
        bank = torch.cat([window, kv], dim=1)
        if joint:
            draft_indices = (win + torch.arange(drafts, device=q.device))[None].expand(batch, -1)
            indices = torch.cat([main_indices, draft_indices], dim=-1)[:, None].expand(-1, drafts, -1)
        else:
            indices = torch.cat(
                [torch.arange(min(win, length), device=q.device), win + torch.arange(drafts, device=q.device)]
            )
            indices = indices.view(1, 1, -1).expand(batch, drafts, -1)
        output = sparse_attention(q, bank, self.sinks, indices, self.scaling)
        return self._project_output(output, cos, sin), None


# --------------------------------------------------------------------------------------------
# hyper-connections, MoE, Engram
# --------------------------------------------------------------------------------------------
class DeepseekV41HyperConnection(nn.Module):
    """Computes the ``pre``/``post``/``comb`` mixes of ``hc_mult`` streams (reference ``hc_mixes``)."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV41UnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor):
        hc, eps = self.hc_mult, self.hc_eps
        flat = hidden_states.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * self.input_norm(flat)
        scale, base = self.scale.float(), self.base.float()
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
        comb = comb.unflatten(-1, (hc, hc))
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb


class DeepseekV41TopKRouter(nn.Module):
    """Returns ``(logits, weights, indices)``; the correction bias selects experts only."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        self.register_buffer("e_score_correction_bias_vl", torch.zeros(self.num_experts, dtype=torch.float32))

    def score_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "softmax":
            return logits.softmax(dim=-1)
        if self.scoring_func == "sigmoid":
            return logits.sigmoid()
        return F.softplus(logits).sqrt()

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        bias = self.e_score_correction_bias
        if image_mask is not None:
            bias = torch.where(image_mask.reshape(-1, 1), self.e_score_correction_bias_vl, bias)
        indices = (scores + bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV41Experts(nn.Module):
    """Routed experts as 3D tensors; returns the FP32 sum in expert order (reference ``MoE``)."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        counts = torch.bincount(top_k_index.flatten(), minlength=self.num_experts).tolist()
        for expert, count in enumerate(counts):
            if not count:
                continue
            token, slot = torch.where(top_k_index == expert)
            gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).float().chunk(2, dim=-1)
            if self.limit > 0:
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            activated = F.silu(gate) * up * top_k_weights[token, slot].unsqueeze(-1)
            value = F.linear(activated.to(hidden_states.dtype), self.down_proj[expert])
            result.index_add_(0, token, value.float())
        return result


class DeepseekV41MLP(nn.Module):
    """The shared expert: clamped SwiGLU (up on both sides, gate from above)."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, hidden_states):
        gate, up = self.gate_proj(hidden_states).float(), self.up_proj(hidden_states).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        return self.down_proj((F.silu(gate) * up).to(hidden_states.dtype))


class DeepseekV41SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV41Experts(config)
        self.shared_experts = DeepseekV41MLP(config)

    def forward(self, hidden_states, image_mask=None):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)
        _, weights, indices = self.gate(flat, image_mask)
        routed = self.experts(flat, indices, weights)
        return (routed + self.shared_experts(flat)).to(hidden_states.dtype).view(shape)


class DeepseekV41EngramEmbedding(nn.Module):
    """FP8 Engram table with one E8M0 scale per 32 values; rows dequantize on lookup (FP32)."""

    block_size = 32

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim % self.block_size:
            raise ValueError("Engram rows are scaled in blocks of 32")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.ones(num_embeddings, embedding_dim // self.block_size, dtype=torch.float8_e8m0fnu),
            requires_grad=False,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        values = self.weight[ids].float().unflatten(-1, (-1, self.block_size))
        return (values * self.scale[ids].float().unsqueeze(-1)).flatten(-2)


class DeepseekV41Engram(nn.Module):
    """Adds the n-gram value into every stream, gated by a normalized stream/key match."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.kv_proj = nn.Linear(
            hash_cols * config.engram_head_dim, config.hidden_size * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(self, hidden_states, engram_rows, token_mask=None):
        kv = self.kv_proj(engram_rows.to(device=hidden_states.device, dtype=hidden_states.dtype).flatten(-2))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = hidden_states.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)


# --------------------------------------------------------------------------------------------
# decoder layers
# --------------------------------------------------------------------------------------------
class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    """``forward(hidden, pre_mix, engram_rows, token_mask, *, shared, **attention_kwargs)``.

    ``hidden`` is ``[B, S, hc_mult, D]`` and ``pre_mix`` collapses it into this
    block's attention input. Returns ``(hidden, ffn_pre)``: the FFN's ``pre`` mix
    collapses the next block's input (or the final output). ``token_mask`` is
    False for image tokens (no Engram, image routing bias).
    """

    attention_class = DeepseekV41Attention

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.self_attn = self.attention_class(config, layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(config)
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)
        self.engram = DeepseekV41Engram(config) if layer_idx in config.engram_layer_ids else None

    @staticmethod
    def hc_collapse(hidden_states: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(dim=2).to(hidden_states.dtype)

    @staticmethod
    def hc_expand(x, residual, post, comb):
        mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        return (post.unsqueeze(-1) * x.unsqueeze(-2) + mixed).to(x.dtype)

    def forward(
        self,
        hidden_states,
        pre_mix,
        engram_rows=None,
        token_mask=None,
        *,
        shared=None,
        stream_capture=None,
        **kwargs,
    ):
        shared = {} if shared is None else shared
        if self.engram is not None:
            if engram_rows is None:
                raise ValueError(f"V4.1 layer {self.layer_idx} needs its Engram rows")
            hidden_states = self.engram(hidden_states, engram_rows, token_mask)
        if stream_capture is not None:
            # dSpark reads the unweighted stream mean at its target layers' input.
            stream_capture.append(hidden_states.mean(dim=2))
        residual = hidden_states
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_states)
        x = self.input_layernorm(self.hc_collapse(hidden_states, pre_mix))
        x, _ = self.self_attn(x, shared, **kwargs)
        hidden_states = self.hc_expand(x, residual, attn_post, attn_comb)

        residual = hidden_states
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_states)
        x = self.post_attention_layernorm(self.hc_collapse(hidden_states, attn_pre))
        x = self.mlp(x, None if token_mask is None else ~token_mask)
        hidden_states = self.hc_expand(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41MarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.head = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids):
        embed = self.embed(token_ids)
        return F.linear(embed.float(), self.head.weight.float()), embed


class DeepseekV41ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, hidden, markov_embed):
        return F.linear(torch.cat([hidden, markov_embed], dim=-1).float(), self.proj.weight.float()).squeeze(-1)


class DeepseekV41DSparkLayer(DeepseekV41DecoderLayer):
    """A dSpark stage (``mtp.<stage>``); stage 0 owns the main-feature input, the last the heads."""

    attention_class = DeepseekV41DSparkAttention

    def __init__(self, config: DeepseekV41TextConfig, stage: int, num_targets: int):
        super().__init__(config, stage)
        if stage == 0:
            self.main_proj = nn.Linear(config.hidden_size * num_targets, config.hidden_size, bias=False)
            self.main_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        if stage == config.num_hidden_layers - 1:
            self.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DeepseekV41MarkovHead(config.vocab_size, config.dspark_markov_rank)
            self.confidence_head = DeepseekV41ConfidenceHead(config.hidden_size + config.dspark_markov_rank)


def sample(logits: torch.Tensor, temperature: float = 1.0, generator=None) -> torch.Tensor:
    """Reference Gumbel-max sampling; ``temperature == 0`` is greedy."""

    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    return probs.div_(noise).argmax(dim=-1)


@dataclass
class DeepseekV41DSparkOutput(ModelOutput):
    output_ids: torch.LongTensor | None = None
    logits: torch.FloatTensor | None = None
    confidence: torch.FloatTensor | None = None


class DeepseekV41DSpark(nn.Module):
    """The dSpark draft head (``mtp.*``): ``num_nextn_predict_layers`` stages sharing the
    main model's token embedding and output head.

    ``forward(main_hidden, input_ids, embed_tokens, lm_head)`` drafts ``dspark_block_size``
    tokens after the known token ``input_ids`` (``[B]``). ``main_hidden`` holds the
    main model's ``dspark_hidden`` for positions ``0..P`` (all history), the known
    token sits at ``P + 1``.
    """

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.config = config.dspark_config()
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.hc_mult = config.hc_mult
        targets = len(config.dspark_target_layer_ids)
        self.layers = nn.ModuleList(
            [DeepseekV41DSparkLayer(self.config, stage, targets) for stage in range(self.config.num_hidden_layers)]
        )

    def prepare(self, main_hidden, input_ids, embed_tokens):
        first = self.layers[0]
        main_x = first.main_norm(first.main_proj(main_hidden))
        draft_ids = input_ids.new_full((input_ids.shape[0], self.block_size), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = embed_tokens(draft_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return hidden, main_x

    def forward(self, main_hidden, input_ids, embed_tokens, lm_head, temperature: float = 0.0, generator=None):
        hidden, main_x = self.prepare(main_hidden, input_ids, embed_tokens)
        pre_mix = hidden.new_zeros(*hidden.shape[:3], dtype=torch.float32)
        pre_mix[..., 0] = 1
        for layer in self.layers:
            hidden, pre_mix = layer(hidden, pre_mix, None, None, shared={}, main_x=main_x)
        last = self.layers[-1]
        x = last.hc_collapse(hidden, pre_mix)
        logits = F.linear(last.norm(x).float(), lm_head.weight.float())
        output_ids = input_ids.new_empty(input_ids.shape[0], self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for index in range(self.block_size):
            bias, markov_embed = last.markov_head(output_ids[:, index])
            logits[:, index] += bias
            markov_embeds.append(markov_embed)
            output_ids[:, index + 1] = sample(logits[:, index], temperature, generator)
        confidence = last.confidence_head(x, torch.stack(markov_embeds, dim=1))
        return DeepseekV41DSparkOutput(output_ids=output_ids, logits=logits, confidence=confidence)


# --------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------
@dataclass
class DeepseekV41ModelOutputWithPast(MoeModelOutputWithPast):
    """``dspark_hidden``: target-layer stream means, concatenated (``output_dspark_hidden=True``)."""

    dspark_hidden: torch.FloatTensor | None = None


@dataclass
class DeepseekV41CausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    dspark_hidden: torch.FloatTensor | None = None


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config: DeepseekV41TextConfig
    config_class = DeepseekV41TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules: ClassVar[list[str]] = ["DeepseekV41DecoderLayer"]
    _skip_keys_device_placement: ClassVar[list[str]] = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    # Compressor and window state is not rewindable (no assisted/contrastive decoding).
    _is_stateful = True
    _can_record_outputs: ClassVar[dict] = {
        "router_logits": OutputRecorder(DeepseekV41TopKRouter, index=0),
        "hidden_states": DeepseekV41DecoderLayer,
    }
    _keep_in_fp32_modules_strict: ClassVar[list[str]] = ["attn_hc", "ffn_hc", "sinks", "e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp\.",
        r"^vision\.",
        r"^aligner\.",
        r"^image_(start|end|newline)$",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, DeepseekV41EngramEmbedding):
            init.zeros_(module.weight)
            init.ones_(module.scale)
            return
        super()._init_weights(module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, DeepseekV41HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV41TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
            init.zeros_(module.e_score_correction_bias_vl)
        elif isinstance(module, DeepseekV41Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV41Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV41Engram):
            init.ones_(module.q_weight)
            init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41RotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq = reference_inv_freq(module.rope_dim, module.config.rope_parameters[layer_type])
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


class DeepseekV41Model(DeepseekV41PreTrainedModel):
    """Embedding -> ``hc_mult`` streams (+ Engram) -> decoder layers -> collapse -> norm."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV41DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        self.engram_tables = nn.ModuleDict(
            {
                str(layer): DeepseekV41EngramEmbedding(rows, config.engram_head_dim)
                for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings)
            }
        )
        self.engram_hash = None
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def bind_engram_hash(self, hash_state):
        """Attach the tokenizer-derived ``DeepseekV41NgramHash`` (required with Engram layers)."""

        if tuple(hash_state.layout.layer_ids) != tuple(self.config.engram_layer_ids):
            raise ValueError("hash-state Engram layer layout differs from the model config")
        self.engram_hash = hash_state

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
        output_dspark_hidden: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DeepseekV41ModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        if attention_mask is not None and (attention_mask.ndim != 2 or not bool(attention_mask.all())):
            raise ValueError("DeepSeek V4.1 has no padded-batch semantics; pass unpadded sequences")
        if past_key_values is not None and not isinstance(past_key_values, DeepseekV41Cache):
            raise TypeError("DeepSeek V4.1 requires its own DeepseekV41Cache")
        if use_cache and past_key_values is None:
            past_key_values = DeepseekV41Cache(self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, length = inputs_embeds.shape[:2]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (torch.arange(length, device=inputs_embeds.device) + past).unsqueeze(0).expand(batch, -1)

        rows = {}
        if len(self.engram_tables):
            if input_ids is None:
                raise ValueError("Engram layers hash input_ids; inputs_embeds alone is not enough")
            if self.engram_hash is None:
                raise ValueError("bind the tokenizer-derived n-gram hash state with bind_engram_hash() first")
            history = past_key_values.engram_history if past_key_values is not None else None
            hashes = self.engram_hash(input_ids, history, None)
            if past_key_values is not None:
                past_key_values.engram_history = self.engram_hash.next_history(history, input_ids)
            for index, layer in enumerate(self.config.engram_layer_ids):
                rows[layer] = self.engram_tables[str(layer)](hashes[:, :, index])

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = hidden_states.new_zeros(batch, length, self.config.hc_mult, dtype=torch.float32)
        pre_mix[..., 0] = 1
        position_embeddings = {
            kind: self.rotary_emb(inputs_embeds, position_ids, kind) for kind in DeepseekV41RotaryEmbedding.layer_types
        }
        shared = {}
        targets = [] if output_dspark_hidden else None
        target_ids = set(self.config.dspark_target_layer_ids)
        for layer in self.layers:
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                rows.get(layer.layer_idx),
                None,
                shared=shared,
                stream_capture=targets if targets is not None and layer.layer_idx in target_ids else None,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=None,
                padding_mask=None,
                past_key_values=past_key_values,
            )
        hidden_states = self.norm(DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix))
        return DeepseekV41ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            dspark_hidden=torch.cat(targets, dim=-1) if targets else None,
        )


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    """Text decoder with the FP32 output head. Accepts the released composite config."""

    config: DeepseekV41Config
    config_class = DeepseekV41Config
    _tied_weights_keys = None

    def __init__(self, config: DeepseekV41Config | DeepseekV41TextConfig):
        super().__init__(config)
        text_config = config.get_text_config()
        self.model = DeepseekV41Model(text_config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

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
    ) -> DeepseekV41CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        if isinstance(self.lm_head, nn.Linear):
            # The reference head computes in FP32.
            logits = F.linear(hidden_states.float(), self.lm_head.weight.float())
        else:
            logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        return DeepseekV41CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            router_logits=outputs.router_logits,
            dspark_hidden=outputs.dspark_hidden,
        )


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41Cache",
    "DeepseekV41CacheLayer",
    "DeepseekV41Compressor",
    "DeepseekV41DSpark",
    "DeepseekV41DSparkAttention",
    "DeepseekV41DSparkLayer",
    "DeepseekV41DecoderLayer",
    "DeepseekV41Engram",
    "DeepseekV41EngramEmbedding",
    "DeepseekV41Experts",
    "DeepseekV41ForCausalLM",
    "DeepseekV41GroupedLinear",
    "DeepseekV41HyperConnection",
    "DeepseekV41Indexer",
    "DeepseekV41MLP",
    "DeepseekV41Model",
    "DeepseekV41PreTrainedModel",
    "DeepseekV41RMSNorm",
    "DeepseekV41RotaryEmbedding",
    "DeepseekV41SparseMoeBlock",
    "DeepseekV41TopKRouter",
    "apply_rotary_emb",
    "dspark_joint_ring",
    "reference_inv_freq",
    "select_candidate_blocks",
    "sparse_attention",
]
