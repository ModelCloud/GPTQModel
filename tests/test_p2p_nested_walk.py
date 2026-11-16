# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for the nested P2P tensor walk and rebuild.

``p2p_move_nested`` must find every tensor leaf before it can decide whether the
peer-copy fast path applies. The walk used to stop at mappings and sequences, so
tensors inside dataclasses and Transformers-style cache objects were invisible,
and the rebuild used ``type(item)(generator)``, which raises for namedtuples.
"""

import dataclasses
from collections import namedtuple

import torch

from gptqmodel.utils import p2p as p2p_mod


_Point = namedtuple("_Point", ["hidden", "scale"])


@dataclasses.dataclass
class _Payload:
    hidden: torch.Tensor
    mask: torch.Tensor


class _FakeCache:
    """Minimal stand-in for ``DynamicCache``/Hymba's hybrid cache."""

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key_cache = [key]
        self.value_cache = [value]


def _collect(item):
    collected = []
    p2p_mod._p2p_walk_tensors(item, collected)
    return collected


def _replacements(*tensors):
    return {
        id(tensor): torch.full_like(tensor, float(index))
        for index, tensor in enumerate(tensors)
    }


def test_walk_and_rebuild_preserve_namedtuple_type():
    first = torch.zeros(2)
    second = torch.ones(2)
    item = {"point": _Point(first, second)}

    assert _collect(item) == [first, second]

    rebuilt = p2p_mod._p2p_rebuild(item, _replacements(first, second))
    point = rebuilt["point"]
    assert isinstance(point, _Point)
    assert point.hidden.tolist() == [0.0, 0.0]
    assert point.scale.tolist() == [1.0, 1.0]


def test_walk_and_rebuild_reach_dataclass_fields():
    payload = _Payload(torch.zeros(2), torch.ones(2))

    assert _collect(payload) == [payload.hidden, payload.mask]

    rebuilt = p2p_mod._p2p_rebuild(
        payload, _replacements(payload.hidden, payload.mask)
    )
    assert isinstance(rebuilt, _Payload)
    assert rebuilt.hidden.tolist() == [0.0, 0.0]
    assert rebuilt.mask.tolist() == [1.0, 1.0]


def test_walk_and_rebuild_reach_cache_key_value_lists():
    cache = _FakeCache(torch.zeros(2), torch.ones(2))
    item = {"cache": cache}

    assert _collect(item) == [cache.key_cache[0], cache.value_cache[0]]

    rebuilt = p2p_mod._p2p_rebuild(
        item, _replacements(cache.key_cache[0], cache.value_cache[0])
    )
    # Caches are mutable and shared; the rebuild updates them in place so the
    # caller's reference stays valid.
    assert rebuilt["cache"] is cache
    assert cache.key_cache[0].tolist() == [0.0, 0.0]
    assert cache.value_cache[0].tolist() == [1.0, 1.0]
