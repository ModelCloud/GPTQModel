# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Instance-local accessors for upstream streaming code on Transformers 5 caches."""

from collections.abc import MutableSequence

from transformers.cache_utils import DynamicCache as NativeDynamicCache


class _TensorView(MutableSequence):
    def __init__(self, cache, field):
        self.cache = cache
        self.field = field

    def __len__(self):
        return len(self.cache.layers)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [getattr(layer, self.field) for layer in self.cache.layers[index]]
        return getattr(self.cache.layers[index], self.field)

    def __setitem__(self, index, value):
        layer = self.cache.layers[index]
        setattr(layer, self.field, value)

    def __delitem__(self, index):
        raise TypeError("Remove cache tokens with crop, not individual layers.")

    def insert(self, index, value):
        raise TypeError("Initialize cache layers with update.")


class DynamicCache(NativeDynamicCache):
    @property
    def key_cache(self):
        return _TensorView(self, "keys")

    @property
    def value_cache(self):
        return _TensorView(self, "values")

    @property
    def seen_tokens(self):
        return self.get_seq_length()

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)

    @classmethod
    def from_legacy_cache(cls, values=None):
        if isinstance(values, cls):
            return values
        if isinstance(values, NativeDynamicCache):
            values = ((layer.keys, layer.values) for layer in values.layers if layer.is_initialized)
        cache = cls()
        for index, (keys, vals) in enumerate(values or ()):
            cache.update(keys, vals, index)
        return cache

    def __getitem__(self, index):
        layer = self.layers[index]
        return layer.keys, layer.values

    def __iter__(self):
        return iter(self.to_legacy_cache())
