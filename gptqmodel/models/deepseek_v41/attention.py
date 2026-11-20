# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window and compressed sparse attention for DeepSeek V4.1."""

import math
from dataclasses import dataclass, field, replace

import torch
from torch import nn
from torch.nn import functional as F

from .layers import V41PackedLinear, V41RMSNorm, fp4_roundtrip, fp8_roundtrip


@dataclass
class _GrowingSequence:
    """Append K/V along the sequence dimension without copying the prefix every step."""

    storage: torch.Tensor | None = None
    length: int = 0

    def append(self, value):
        end = self.length + value.size(1)
        if self.storage is None:
            self.storage = value
        elif end > self.storage.size(1):
            shape = (value.size(0), max(end, 2 * self.length), *value.shape[2:])
            storage = value.new_empty(shape)
            storage[:, :self.length].copy_(self.storage[:, :self.length])
            self.storage = storage
        if value.size(1) and self.storage is not value:
            self.storage[:, self.length:end].copy_(value)
        self.length = end
        return self.storage[:, :end]

    def reorder(self, indices):
        if self.storage is not None:
            # Only the occupied prefix is initialized.
            self.storage = self.storage[:, :self.length].index_select(0, indices.to(self.storage.device))


@dataclass
class _LayerHistory:
    window: torch.Tensor | None = None
    pool_tail: torch.Tensor | None = None
    gate_tail: torch.Tensor | None = None
    compressed: _GrowingSequence = field(default_factory=_GrowingSequence)
    index_keys: _GrowingSequence = field(default_factory=_GrowingSequence)

    def reorder(self, indices):
        for name in ("window", "pool_tail", "gate_tail"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, indices.to(value.device)))
        self.compressed.reorder(indices)
        self.index_keys.reorder(indices)


class DeepSeekV41FlashCache:
    """Each request owns its window, incomplete pooling groups and compressed banks."""

    def __init__(self, num_layers):
        self.layers = [_LayerHistory() for _ in range(num_layers)]
        self.length = 0
        self.batch_size = None
        self.engram_tail = None
        self.draft_windows = ()
        self.draft_length = 0

    def stage(self):
        # Appends can use spare capacity, but a failed forward must leave the
        # previous visible prefix and all window/pooling state intact.
        return [
            replace(layer, compressed=replace(layer.compressed), index_keys=replace(layer.index_keys))
            for layer in self.layers
        ]

    def get_seq_length(self, layer_idx=0):
        return self.length

    def reorder_cache(self, beam_idx):
        for layer in self.layers:
            layer.reorder(beam_idx)
        if self.engram_tail is not None:
            self.engram_tail = self.engram_tail.index_select(0, beam_idx.to(self.engram_tail.device))
        self.draft_windows = tuple(window.index_select(0, beam_idx.to(window.device)) for window in self.draft_windows)
        self.batch_size = beam_idx.numel()


@dataclass
class V41AttentionState:
    """Source outputs passed between decoder layers during one forward."""

    kv: torch.Tensor | None = None
    keys: torch.Tensor | None = None
    selected: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


class V41Rotary:
    def __init__(self, config, *, compressed):
        self.width = config.qk_rope_head_dim
        self.theta = config.compress_rope_theta if compressed else config.rope_theta
        rope = config.rope_parameters or {}
        self.scaling = rope if compressed and rope.get("rope_type") == "yarn" else None

    def __call__(self, positions):
        power = torch.arange(0, self.width, 2, dtype=torch.float32, device=positions.device) / self.width
        frequency = self.theta ** -power
        if self.scaling:
            original = self.scaling["original_max_position_embeddings"]
            log_base = self.width / (2 * math.log(self.theta))
            fast = self.scaling["beta_fast"]
            slow = self.scaling["beta_slow"]
            low = max(0, math.floor(log_base * math.log(original / (2 * math.pi * fast))))
            high = min(self.width - 1, math.ceil(log_base * math.log(original / (2 * math.pi * slow))))
            ramp = ((power * (self.width / 2) - low) / max(high - low, 1e-3)).clamp(0, 1)
            frequency = frequency * (1 - ramp + ramp / self.scaling["factor"])
        angles = positions.float().unsqueeze(-1) * frequency
        return angles.cos(), angles.sin()


def _rotate(x, coefficients, *, inverse=False):
    cos, sin = coefficients
    width = cos.shape[-1] * 2
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    if inverse:
        sin = -sin
    real, imag = x[..., -width:].float().unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    return torch.cat((x[..., :-width], rotated.flatten(-2).to(x.dtype)), dim=-1)


class V41Compressor(nn.Module):
    def __init__(self, config, dtype, ratio):
        super().__init__()
        self.ratio = ratio
        self.norm = V41RMSNorm(config.head_dim, config.rms_norm_eps, dtype)
        projection_dtype = torch.float32 if ratio > 1 else dtype
        self.wkv = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=projection_dtype)
        self.wgate = (
            nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=projection_dtype)
            if ratio > 1 else None
        )

    def forward(self, hidden, history):
        if self.wgate is None:
            return self.norm(self.wkv(hidden))

        x = hidden.float()
        values, gates = self.wkv(x), self.wgate(x)
        if history.pool_tail is not None:
            values = torch.cat((history.pool_tail, values), dim=1)
            gates = torch.cat((history.gate_tail, gates), dim=1)
        groups, remainder = divmod(values.size(1), self.ratio)
        stop = groups * self.ratio
        history.pool_tail = values[:, stop:].clone() if remainder else None
        history.gate_tail = gates[:, stop:].clone() if remainder else None
        values = values[:, :stop].unflatten(1, (groups, self.ratio))
        gates = gates[:, :stop].unflatten(1, (groups, self.ratio))
        return self.norm((values * gates.softmax(dim=2)).sum(dim=2).to(hidden.dtype))


def _rank_entries(scores, limit):
    # Fix the order of equal scores to the key position. The selection for a
    # prefix must be independent of the number of masked keys after it.
    ranked = scores.argsort(dim=-1, descending=True, stable=True)[..., :limit]
    return ranked.masked_fill(scores.gather(-1, ranked) == -torch.inf, -1).to(torch.int32)


def _candidate_blocks(scores, visible, block_size, topk):
    # A partially filled newest block is always eligible.
    count = scores.size(-1)
    by_block = F.pad(scores, (0, -count % block_size), value=-torch.inf)
    by_block = by_block.unflatten(-1, (-1, block_size)).amax(-1)
    newest = (visible - 1) // block_size
    by_block.masked_fill_(torch.arange(by_block.size(-1), device=scores.device) == newest, torch.inf)
    return _rank_entries(by_block, topk)


class V41Indexer(nn.Module):
    def __init__(self, config, dtype, layer_id, ratio, *, owns_keys):
        super().__init__()
        self.ratio = ratio
        self.heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.block_size = max(1, config.candidate_block_size)
        self.topk_blocks = config.candidate_topk_blocks
        source = config.candidate_source_layer_id
        self.produces_candidates = layer_id == source
        self.uses_candidates = source is not None and layer_id > source

        self.wq_b = V41PackedLinear(config.q_lora_rank, self.heads * self.head_dim)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False, dtype=dtype)
        if owns_keys:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False, dtype=dtype)
            self.k_norm = V41RMSNorm(self.head_dim, config.rms_norm_eps, dtype)

    def forward(self, hidden, query_residual, keys, positions, coefficients, candidates):
        if keys.size(1) == 0:
            return torch.empty((*hidden.shape[:2], 0), dtype=torch.int32, device=hidden.device), candidates

        query = self.wq_b(query_residual).unflatten(-1, (self.heads, self.head_dim))
        query = fp4_roundtrip(_rotate(query, coefficients))
        weights = self.weights_proj(hidden) * ((self.heads * self.head_dim) ** -0.5)
        entries = torch.arange(keys.size(1), device=hidden.device)
        block_ids = entries // self.block_size
        block_count = (keys.size(1) + self.block_size - 1) // self.block_size
        # Keep only top-k indices between chunks and layers. Score tensors are temporary.
        chunk = max(1, min(32, 2**20 // (hidden.size(0) * self.heads * keys.size(1))))
        selected, new_candidates = [], []
        for start in range(0, hidden.size(1), chunk):
            stop = start + chunk
            visible = ((positions[start:stop] + 1) // self.ratio).unsqueeze(-1)
            scores = torch.einsum("bqhd,bkd->bqhk", query[:, start:stop], keys)
            scores = (scores.relu_() * weights[:, start:stop].unsqueeze(-1)).sum(dim=2)
            scores.masked_fill_(entries >= visible, -torch.inf)
            if self.produces_candidates:
                new_candidates.append(_candidate_blocks(scores, visible, self.block_size, self.topk_blocks))
            if self.uses_candidates:
                picks = candidates[:, start:stop].long()
                kept = scores.new_zeros((*scores.shape[:2], block_count + 1), dtype=torch.bool)
                kept.scatter_(-1, picks.masked_fill(picks < 0, block_count), True)
                scores.masked_fill_(~kept.index_select(-1, block_ids), -torch.inf)
            selected.append(_rank_entries(scores, self.topk))

        if new_candidates:
            candidates = torch.cat(new_candidates, dim=1)
        return torch.cat(selected, dim=1), candidates


class V41Attention(nn.Module):
    def __init__(self, config, dtype, layer_id, schedule):
        super().__init__()
        ratio, kv_source, index_source = schedule
        self.ratio = ratio
        self.window = config.sliding_window
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.groups = config.o_groups
        self.rotary = V41Rotary(config, compressed=bool(ratio))
        self.attn_sink = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))

        width, rank = config.hidden_size, config.q_lora_rank
        self.wq_a = V41PackedLinear(width, rank)
        self.q_norm = V41RMSNorm(rank, config.rms_norm_eps, dtype)
        self.wq_b = V41PackedLinear(rank, self.heads * self.head_dim)
        self.wkv = V41PackedLinear(width, self.head_dim)
        self.kv_norm = V41RMSNorm(self.head_dim, config.rms_norm_eps, dtype)
        output_rank = self.groups * config.o_lora_rank
        self.wo_a = V41PackedLinear(self.heads * self.head_dim // self.groups, output_rank, groups=self.groups)
        self.wo_b = V41PackedLinear(output_rank, width)

        self.compressor = V41Compressor(config, dtype, ratio) if layer_id == kv_source else None
        self.indexer = (
            V41Indexer(config, dtype, layer_id, ratio, owns_keys=layer_id == kv_source)
            if layer_id == index_source else None
        )

    def _pool(self, hidden, positions, history, state):
        latents = self.compressor(hidden, history)
        group_starts = (
            torch.arange(latents.size(1), device=hidden.device) * self.ratio
            + (positions[0] // self.ratio) * self.ratio
        )
        coefficients = self.rotary(group_starts)
        keys = self.indexer.k_norm(self.indexer.wk(latents))
        state.keys = history.index_keys.append(fp4_roundtrip(_rotate(keys, coefficients)))
        values = fp4_roundtrip(_rotate(latents, coefficients), compressed_kv=True)
        state.kv = history.compressed.append(values)

    def _attend(self, query, window, history_length, state):
        compressed = state.kv
        selected = state.selected
        width = min(window.size(1), self.window)
        offsets = torch.arange(1 - width, 1, device=window.device)
        outputs = []
        for start in range(0, query.size(1), 32):
            stop = min(start + 32, query.size(1))
            positions = torch.arange(start, stop, device=window.device) + history_length
            indices = positions[:, None] + offsets
            values = window[:, indices.clamp_min(0)]
            valid = (indices >= 0).expand(query.size(0), -1, -1)
            if self.ratio and compressed.size(1) > 0:
                indices = selected[:, start:stop].long()
                batch = torch.arange(query.size(0), device=window.device)[:, None, None]
                sparse = compressed[batch, indices.clamp_min(0)]
                values = torch.cat((values, sparse), dim=2)
                valid = torch.cat((valid, indices >= 0), dim=-1)
            scores = torch.einsum("bqhd,bqkd->bqhk", query[:, start:stop].float(), values.float())
            scores = scores * (self.head_dim ** -0.5)
            scores.masked_fill_(~valid.unsqueeze(-2), -torch.inf)
            sink = self.attn_sink.expand(*scores.shape[:2], self.heads).unsqueeze(-1)
            probabilities = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
            outputs.append(torch.einsum("bqhk,bqkd->bqhd", probabilities, values.float()).to(query.dtype))
        return torch.cat(outputs, dim=1)

    def forward(self, hidden, positions, history, state):
        coefficients = self.rotary(positions)
        residual = self.q_norm(self.wq_a(hidden))
        query = self.wq_b(residual).unflatten(-1, (self.heads, self.head_dim))
        query = _rotate(query, coefficients)
        kv = fp8_roundtrip(_rotate(self.kv_norm(self.wkv(hidden)), coefficients))

        previous = history.window
        history_length = 0 if previous is None else previous.size(1)
        window = kv if previous is None else torch.cat((previous, kv), dim=1)
        kept = min(window.size(1), self.window - 1)
        history.window = window[:, -kept:].clone() if kept else None

        if self.compressor is not None:
            self._pool(hidden, positions, history, state)
        if self.ratio:
            state.kv = state.kv.to(hidden.device)
            if self.indexer is not None:
                state.keys = state.keys.to(hidden.device)
                candidates = None if state.candidates is None else state.candidates.to(hidden.device)
                state.selected, state.candidates = self.indexer(
                    hidden, residual, state.keys, positions, coefficients, candidates
                )
            else:
                state.selected = state.selected.to(hidden.device)
        attended = self._attend(query, window, history_length, state)
        return self._project_output(attended, coefficients)

    def _project_output(self, attended, coefficients):
        grouped = _rotate(attended, coefficients, inverse=True).reshape(*attended.shape[:2], self.groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(-2))


class V41DraftAttention(V41Attention):
    """A draft block attends to accepted KV and jointly to its draft positions."""

    def accepted_kv(self, main, positions, previous):
        kv = fp8_roundtrip(_rotate(self.kv_norm(self.wkv(main)), self.rotary(positions)))
        if previous is not None:
            kv = torch.cat((previous, kv), dim=1)
        return kv[:, -self.window:].clone()

    def forward(self, hidden, positions, accepted, state):
        coefficients = self.rotary(positions)
        residual = self.q_norm(self.wq_a(hidden))
        query = _rotate(self.wq_b(residual).unflatten(-1, (self.heads, self.head_dim)), coefficients)
        draft = fp8_roundtrip(_rotate(self.kv_norm(self.wkv(hidden)), coefficients))
        bank = torch.cat((accepted, draft), dim=1).float()
        scores = torch.einsum("bqhd,bkd->bqhk", query.float(), bank) * self.head_dim**-0.5
        sink = self.attn_sink.expand(*scores.shape[:2], self.heads).unsqueeze(-1)
        probabilities = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
        attended = torch.einsum("bqhk,bkd->bqhd", probabilities, bank).to(hidden.dtype)
        return self._project_output(attended, coefficients)
