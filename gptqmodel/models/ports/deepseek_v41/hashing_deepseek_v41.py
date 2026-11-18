# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Adapted from DeepSeek-V4.1-Flash ``inference/engram.py`` at
# 314113f8a7dadb388215396ba5fbee2c610a04d6, Copyright (c) 2023 DeepSeek, MIT License
# (see LICENSE in this directory).
"""Engram n-gram hashing for DeepSeek V4.1.

The hash ids select rows of the per-layer Engram (PLE) tables. They depend on a
compressed token map derived from the tokenizer; GPT-QModel never builds a
tokenizer implicitly, so the caller supplies the map (``build_compressed_token_map``
reproduces the reference derivation from an already loaded fast tokenizer).

Differences from the reference module, none of which changes a hash id:

* primes are found with a deterministic Miller-Rabin test instead of ``sympy``;
* instead of a ``[max_batch, max_seq]`` token cache, the caller passes the
  compressed ids of at most ``max_ngram_size - 1`` preceding tokens (``history``),
  which is all the hash ever reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


_MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(value: int) -> bool:
    """Deterministic for every 64-bit integer."""

    if value < 2:
        return False
    for prime in _MILLER_RABIN_BASES:
        if value % prime == 0:
            return value == prime
    odd, rounds = value - 1, 0
    while odd % 2 == 0:
        odd //= 2
        rounds += 1
    for base in _MILLER_RABIN_BASES:
        x = pow(base, odd, value)
        if x in (1, value - 1):
            continue
        for _ in range(rounds - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""

    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Reference token normalization: ids that normalize alike share one compressed id."""

    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids, max_ngram_size: int, tokenizer_vocab_size: int) -> torch.Tensor:
    """One odd multiplier per (layer, lookback), from a per-layer RNG."""

    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Prime-sized, disjoint bucket ranges of every (layer, n-gram size, head)."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config) -> EngramLayout | None:
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = config.engram_max_ngram_size, config.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=config.engram_head_dim,
        )

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


class DeepseekV41NgramHash(nn.Module):
    """Maps every position to the Engram row ids of the n-grams ending there.

    ``forward(input_ids, history=None, token_mask=None)`` returns
    ``[batch, seq, n_engram_layers, n_hash_cols]``. ``history`` holds the
    compressed ids of the up to ``max_ngram_size - 1`` tokens that precede
    ``input_ids`` (``DEAD`` for image tokens); ``token_mask`` is False for
    tokens that take no part in an n-gram.
    """

    DEAD = -1

    def __init__(self, config, token_map):
        super().__init__()
        layout = EngramLayout.from_config(config)
        if layout is None:
            raise ValueError("config declares no Engram layers")
        token_map = torch.as_tensor(token_map, dtype=torch.long)
        if token_map.ndim != 1 or token_map.numel() != config.vocab_size:
            raise ValueError("the compressed token map needs one entry per vocabulary id")
        # Every hash multiplier derives from the compressed vocab size.
        vocab_size = config.engram_compressed_vocab_size
        if int(token_map.min()) < 0 or int(token_map.max()) >= vocab_size:
            raise ValueError(f"compressed token ids must lie in [0, {vocab_size})")
        self.layout = layout
        self.pad_id = int(token_map[config.engram_pad_token_id])
        flat = [[prime for per_ngram in layer for prime in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        for sizes, rows in zip(flat, layout.num_embeddings):
            if sum(sizes) != rows:
                raise ValueError("Engram prime layout does not match engram_num_embeddings")
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size),
            persistent=False,
        )
        self.register_buffer("token_map", token_map.clone(), persistent=False)

    def compress(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        return compressed

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        history: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seqlen = input_ids.shape
        compressed = self.compress(input_ids, token_mask)
        if history is None:
            history = compressed.new_empty(batch, 0)
        if history.shape[0] != batch or history.shape[1] > self.layout.max_ngram_size - 1:
            raise ValueError("n-gram history must be [batch, <= max_ngram_size - 1]")
        sequence = torch.cat([history.to(compressed.device), compressed], dim=1)
        positions = history.shape[1] + torch.arange(seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = sequence.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for index in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def next_history(self, history: torch.Tensor | None, input_ids, token_mask=None) -> torch.Tensor:
        """Compressed ids of the trailing ``max_ngram_size - 1`` tokens, for the next call."""

        compressed = self.compress(input_ids, token_mask)
        if history is not None:
            compressed = torch.cat([history.to(compressed.device), compressed], dim=1)
        return compressed[:, -(self.layout.max_ngram_size - 1) :] if self.layout.max_ngram_size > 1 else compressed[:, :0]


__all__ = [
    "DeepseekV41NgramHash",
    "EngramLayout",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "find_next_prime",
    "is_prime",
]
