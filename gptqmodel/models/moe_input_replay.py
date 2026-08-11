# SPDX-License-Identifier: Apache-2.0
"""Bounded exact-input replay between dependent routed MoE subsets."""

from __future__ import annotations

import threading
from typing import Optional

import torch

from .base import (
    MODULE_TREE_FLAG_DOWN,
    MODULE_TREE_FLAG_GATE,
    MODULE_TREE_FLAG_ROUTED,
    MODULE_TREE_FLAG_UP,
)


def routed_projection_roles(subset) -> frozenset[str]:
    """Return explicit routed projection roles declared by the module tree."""

    roles = set()
    for named_module in (subset or {}).values():
        flags = frozenset(getattr(named_module, "state", {}).get("module_tree_flags", ()))
        if MODULE_TREE_FLAG_ROUTED not in flags:
            continue
        roles.update(flags & {MODULE_TREE_FLAG_GATE, MODULE_TREE_FLAG_UP, MODULE_TREE_FLAG_DOWN})
    return frozenset(roles)


class RoutedMoEInputReplayAttachment:
    """Retain exact gate/up inputs for the immediately following down subset."""

    def __init__(self, *, maximum_retained_bytes: int = 12 * 1024**3):
        self.maximum_retained_bytes = int(maximum_retained_bytes)
        self._lock = threading.Lock()
        self._producer_active = False
        self._consumer_active = False
        self._expected_batches = 0
        self._inputs: dict[int, torch.Tensor] = {}
        self._storage_bytes: dict[tuple[object, ...], int] = {}
        self._retained_bytes = 0

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[object, ...]:
        storage = tensor.untyped_storage()
        return (tensor.device.type, tensor.device.index, storage.data_ptr())

    def prepare_subset(self, subset, batch_count: int) -> None:
        roles = routed_projection_roles(subset)
        batch_count = int(batch_count)
        with self._lock:
            self._producer_active = roles == {MODULE_TREE_FLAG_GATE, MODULE_TREE_FLAG_UP}
            self._consumer_active = (
                roles == {MODULE_TREE_FLAG_DOWN}
                and batch_count > 0
                and batch_count == self._expected_batches
                and len(self._inputs) == self._expected_batches
                and set(self._inputs) == set(range(self._expected_batches))
            )
            if self._producer_active:
                self._inputs.clear()
                self._storage_bytes.clear()
                self._retained_bytes = 0
                self._expected_batches = batch_count
            elif not self._consumer_active:
                self._clear_locked()

    def retain(self, batch_index: Optional[int], hidden_states: torch.Tensor) -> None:
        if batch_index is None:
            return
        value = hidden_states.detach()
        with self._lock:
            if not self._producer_active or batch_index in self._inputs:
                return
            key = self._storage_key(value)
            storage_bytes = 0 if key in self._storage_bytes else value.untyped_storage().nbytes()
            requested = self._retained_bytes + storage_bytes
            if requested > self.maximum_retained_bytes:
                self._clear_locked()
                return
            self._storage_bytes[key] = storage_bytes
            self._retained_bytes = requested
            self._inputs[int(batch_index)] = value

    def take(self, batch_index: int) -> Optional[torch.Tensor]:
        with self._lock:
            if not self._consumer_active:
                return None
            value = self._inputs.pop(int(batch_index), None)
            if not self._inputs:
                self._clear_locked()
            return value

    def _clear_locked(self) -> None:
        self._producer_active = False
        self._consumer_active = False
        self._expected_batches = 0
        self._inputs.clear()
        self._storage_bytes.clear()
        self._retained_bytes = 0

    def abort(self) -> None:
        with self._lock:
            self._clear_locked()
