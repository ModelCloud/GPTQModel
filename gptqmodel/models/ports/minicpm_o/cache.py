# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Layer-API helpers for MiniCPM-o streaming code on Transformers 5.23 ``DynamicCache``.

Upstream streaming code edits KV tensors directly (sliding windows, RoPE reindexing, round
dropping). These helpers express those edits through ``Cache.layers`` instead of tuple caches.
"""

from transformers.cache_utils import DynamicCache


def cache_length(cache) -> int:
    """Sequence length held by ``cache`` (0 for ``None``)."""
    return 0 if cache is None else cache.get_seq_length()


def cache_pairs(cache):
    """``[(keys, values), ...]`` for every initialized layer of ``cache``."""
    return [(layer.keys, layer.values) for layer in cache.layers if layer.is_initialized]


def cache_from_pairs(pairs) -> DynamicCache:
    """A new ``DynamicCache`` holding ``pairs`` of ``(keys, values)`` per layer."""
    cache = DynamicCache()
    for index, (keys, values) in enumerate(pairs):
        cache.update(keys, values, index)
    return cache


def set_layer_states(cache, layer_idx: int, keys, values) -> None:
    """Replace the cached tensors of one initialized layer in place."""
    layer = cache.layers[layer_idx]
    layer.keys, layer.values = keys, values
