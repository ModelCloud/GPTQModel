# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The default layer-wise map must not leave the capability group (#4734).

#4714 kept the *kernel-selection* target inside the first device's
compute-capability group, but the default resolution still returned that group
as a scalar ``torch.device`` when the group held one card. The layer-wise map
builder treats a scalar as an abstract family, so ``_layerwise_device_count``
fell back to ``torch.cuda.device_count()`` and spread decoder layers over
**every** visible GPU — including the other generation. On a 4090 + 5090 host
with one card per capability that meant the mixed-capability guard passed
(it only saw ``cuda:0``) while placement still spanned both generations.

The default placement therefore has to carry the resolved group explicitly,
exactly as it already does when the group holds two or more cards.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel import BACKEND
from gptqmodel.models.loader import (
    _explicit_layerwise_devices,
    _layerwise_device_count,
    _layerwise_device_strings,
    _resolve_default_layerwise_device,
)
from gptqmodel.utils.importer import _selector_devices


pytestmark = [pytest.mark.cpu]


def _gpu_capabilities(monkeypatch, capabilities):
    """Describe a host whose GPUs have ``capabilities`` in ordinal order."""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.get_device_capability",
        lambda device=None: capabilities[torch.device(device).index],
    )
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.device_count",
        lambda: len(capabilities),
    )


def _default_map_targets(monkeypatch, capabilities):
    """Resolve the default device and return what the map builder would use."""

    _gpu_capabilities(monkeypatch, capabilities)
    resolved = _resolve_default_layerwise_device(None, BACKEND.AUTO)
    count = _layerwise_device_count(resolved)
    return resolved, count, _layerwise_device_strings(resolved, count)


def test_one_gpu_per_capability_maps_only_the_first_group(monkeypatch):
    """4090 + 5090: the group is one card, and the map must stay on it."""

    resolved, count, targets = _default_map_targets(monkeypatch, [(8, 9), (12, 0)])

    # Before the fix this was ``cuda:0`` (a scalar), so ``count`` was 2 and the
    # map builder spread layers over ``cuda:0`` and ``cuda:1``.
    assert _selector_devices(resolved) == (torch.device("cuda:0"),)
    assert count == 1
    assert targets == ["cuda:0"]
    assert _explicit_layerwise_devices(resolved) == ["cuda:0"]


def test_a_group_with_two_devices_maps_to_exactly_those(monkeypatch):
    """[8.9, 12.0, 8.9]: skip the other generation, keep the group's ordinals."""

    resolved, count, targets = _default_map_targets(
        monkeypatch, [(8, 9), (12, 0), (8, 9)]
    )

    assert count == 2
    assert targets == ["cuda:0", "cuda:2"]


def test_a_homogeneous_host_still_uses_every_gpu(monkeypatch):
    """The fix must not collapse a single-generation multi-GPU host."""

    resolved, count, targets = _default_map_targets(monkeypatch, [(8, 9), (8, 9)])

    assert count == 2
    assert targets == ["cuda:0", "cuda:1"]


def test_a_single_visible_gpu_still_maps_to_one_device(monkeypatch):
    resolved, count, targets = _default_map_targets(monkeypatch, [(12, 1)])

    assert _selector_devices(resolved) == (torch.device("cuda:0"),)
    assert count == 1
    assert targets == ["cuda:0"]


def test_an_explicit_sequence_keeps_its_own_span(monkeypatch):
    """An explicit request is a caller choice and must not be narrowed."""

    _gpu_capabilities(monkeypatch, [(8, 9), (8, 9)])
    resolved = _resolve_default_layerwise_device(
        (torch.device("cuda:0"), torch.device("cuda:1")), BACKEND.AUTO
    )

    assert _selector_devices(resolved) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )
    assert _layerwise_device_count(resolved) == 2


def test_an_explicit_mixed_span_is_still_refused(monkeypatch):
    """Narrowing is only for the default; an explicit mixed span fails closed."""

    _gpu_capabilities(monkeypatch, [(8, 9), (12, 0)])

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        _resolve_default_layerwise_device(
            (torch.device("cuda:0"), torch.device("cuda:1")), BACKEND.AUTO
        )
