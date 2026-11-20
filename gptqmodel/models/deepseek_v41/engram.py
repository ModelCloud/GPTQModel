# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-defined n-gram hashes and mapped Engram rows."""

import numpy as np
import torch
from sympy import nextprime
from tokenizers import Regex, normalizers
from torch import nn

from ...utils.ple_mmap import MappedPLETable
from .layers import V41PackedLinear


def _compressed_vocabulary(tokenizer):
    # Keep a single space distinct from whitespace removed by Strip.
    sentinel = "\ue000"
    normalize = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex("^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    groups = {}
    token_map = []
    for token in range(len(tokenizer)):
        text = backend.decode([token], skip_special_tokens=False)
        key = backend.id_to_token(token) if "\ufffd" in text else normalize.normalize_str(text) or text
        token_map.append(groups.setdefault(key, len(groups)))
    return token_map, len(groups)


def engram_layers(config):
    ids = tuple(config.engram_layer_ids)
    rows = tuple(config.engram_num_embeddings)
    if len(ids) != len(rows) or len(set(ids)) != len(ids) or any(
        not 0 <= layer < config.num_hidden_layers or size <= 0 for layer, size in zip(ids, rows)
    ):
        raise ValueError("Invalid DeepSeek V4.1 Engram layer or table sizes")
    if ids and (
        config.engram_max_ngram_size < 2 or config.engram_vocab_size < 2
        or config.engram_n_heads < 1 or config.engram_head_dim <= 0 or config.engram_head_dim % 32
        or config.engram_compressed_vocab_size < 1
    ):
        raise ValueError("Invalid DeepSeek V4.1 Engram geometry")
    return dict(zip(ids, rows))


class V41EngramHash(nn.Module):
    """Hash compressed token IDs with a bounded prefix carried by the request."""

    def __init__(self, config, tokenizer):
        super().__init__()
        ids, rows = tuple(config.engram_layer_ids), tuple(config.engram_num_embeddings)
        token_map, count = _compressed_vocabulary(tokenizer)
        if count != config.engram_compressed_vocab_size:
            raise ValueError(f"DeepSeek V4.1 Engram expected {config.engram_compressed_vocab_size} compressed tokens, got {count}")
        if not 0 <= config.engram_pad_token_id < len(token_map):
            raise ValueError("DeepSeek V4.1 Engram padding token is outside the tokenizer vocabulary")
        self.history_size = config.engram_max_ngram_size - 1
        self.pad_id = token_map[config.engram_pad_token_id]
        self.layer_indices = {index: slot for slot, index in enumerate(ids)}
        # Every head starts at the same lower bound. Consuming primes in
        # checkpoint order makes the next free prime monotonically increase.
        prime = config.engram_vocab_size - 1
        moduli = []
        for size in rows:
            buckets = []
            for _ in range(self.history_size * config.engram_n_heads):
                prime = int(nextprime(prime))
                buckets.append(prime)
            if sum(buckets) != size:
                raise ValueError("DeepSeek V4.1 Engram hash buckets differ from the table size")
            moduli.append(buckets)
        moduli = torch.tensor(moduli, dtype=torch.int64)
        bound = max(1, (torch.iinfo(torch.int64).max // count) // 2)
        multipliers = [
            np.random.default_rng(10007 * index).integers(bound, size=self.history_size + 1, dtype=np.int64) * 2 + 1
            for index in ids
        ]
        self.register_buffer("token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False)
        self.register_buffer("multipliers", torch.from_numpy(np.stack(multipliers)), persistent=False)
        self.register_buffer("offsets", moduli.cumsum(-1) - moduli, persistent=False)
        self.register_buffer("moduli", moduli.unflatten(-1, (self.history_size, config.engram_n_heads)), persistent=False)

    def prepare(self, input_ids, tail, image_mask):
        tokens = self.token_map.index_select(0, input_ids.reshape(-1)).view_as(input_ids)
        if image_mask is not None:
            tokens = tokens.masked_fill(image_mask, -1)
        if tail is None:
            tail = tokens.new_full((tokens.size(0), self.history_size), -1)
        context = torch.cat((tail, tokens), dim=1)
        return context, context[:, -self.history_size:].clone()

    def forward(self, context, layer_index):
        slot = self.layer_indices[layer_index]
        length = context.size(1) - self.history_size
        multiplier = self.multipliers[slot]
        current = context[:, self.history_size:]
        blocked = current < 0
        code = current.masked_fill(blocked, self.pad_id) * multiplier[0]
        rows = []
        for order, modulus in enumerate(self.moduli[slot], start=1):
            previous = context[:, self.history_size - order:self.history_size - order + length]
            blocked = blocked | (previous < 0)
            code = code ^ (previous.masked_fill(blocked, self.pad_id) * multiplier[order])
            rows.append(code.unsqueeze(-1).remainder(modulus))
        return torch.cat(rows, dim=-1) + self.offsets[slot]


class V41EngramEmbedding(nn.Module):
    """Decode gathered rows while keeping the full table in its source shards."""

    def __init__(self, rows, width):
        super().__init__()
        self.rows = rows
        self.width = width
        self.table = None

    def bind(self, weight_path, scale_path, prefix):
        table = MappedPLETable(weight_path, prefix, scale_path=scale_path)
        if (table.rows, table.width) != (self.rows, self.width):
            table.close()
            raise ValueError(f"Invalid DeepSeek V4.1 Engram table shape for {prefix}")
        self.table = table

    def forward(self, ids, device, dtype):
        table = self.table
        if table is None:
            raise RuntimeError("DeepSeek V4.1 Engram table must be bound before execution")
        values = torch.empty((ids.numel(), self.width), dtype=dtype, device=device)
        try:
            for start, indices in enumerate(ids.reshape(-1).cpu().split(table.max_gather_rows)):
                rows, scales = table.gather(indices.tolist())
                row = torch.frombuffer(bytearray(rows), dtype=torch.float8_e4m3fn).reshape(-1, self.width // 32, 32)
                scale = torch.frombuffer(bytearray(scales), dtype=torch.float8_e8m0fnu).reshape(-1, self.width // 32, 1)
                first = start * table.max_gather_rows
                values[first:first + len(indices)].copy_((row.float() * scale.float()).flatten(-2).to(dtype))
        finally:
            table.release()
        return values.view(*ids.shape, self.width)


class V41Engram(nn.Module):
    def __init__(self, config, rows, dtype):
        super().__init__()
        width, streams = config.hidden_size, config.hc_mult
        self.eps = config.rms_norm_eps
        self.embed = V41EngramEmbedding(rows, config.engram_head_dim)
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = V41PackedLinear(columns * config.engram_head_dim, (streams + 1) * width)
        self.q_weight = nn.Parameter(torch.empty(streams, width, dtype=dtype))
        self.k_weight = nn.Parameter(torch.empty(streams, width, dtype=dtype))
        self.chunk_size = max(1, (1 << 22) // max(columns * config.engram_head_dim, (streams + 1) * width))

    def forward(self, hidden, ids, image_mask):
        tokens = hidden.flatten(0, 1)
        indices = ids.flatten(0, 1)
        mask = None if image_mask is None else image_mask.flatten()
        output = torch.empty_like(tokens)
        for start in range(0, tokens.size(0), self.chunk_size):
            rows = slice(start, start + self.chunk_size)
            output[rows] = self._add_rows(tokens[rows], indices[rows], None if mask is None else mask[rows])
        return output.reshape_as(hidden)

    def _add_rows(self, hidden, ids, image_mask):
        streams, width = hidden.shape[-2:]
        rows = self.embed(ids, hidden.device, hidden.dtype).flatten(-2)
        keys, value = self.wkv(rows).split((streams * width, width), dim=-1)
        x = hidden.float()
        keys = keys.float().unflatten(-1, (streams, width))
        weight = self.q_weight.float() * self.k_weight.float()
        norm_x = torch.rsqrt(x.square().mean(-1) + self.eps)
        norm_k = torch.rsqrt(keys.square().mean(-1) + self.eps)
        score = (x * weight * keys).sum(-1) * (norm_x * norm_k) * width**-0.5
        gate = torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score).sigmoid()
        if image_mask is not None:
            gate = gate.masked_fill(image_mask.unsqueeze(-1), 0)
        return (x + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
