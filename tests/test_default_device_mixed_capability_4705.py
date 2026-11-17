# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The default placement must not span mixed GPU generations (#4705).

With no ``device``, ``device_map`` or ``max_memory`` the loader expanded the
CUDA family to every visible GPU, and kernel selection then refused a map whose
devices have different compute capabilities. On a host with an RTX 4090 next to
an RTX 5090 the plainest call — ``GPTQModel.load(path)`` — always raised, even
for a 1B model that fits on one card. ``max_memory`` budgets were fixed in
#4637; this pins the default path.

The default placement now stays inside the first device's capability group.
Explicit placement arguments still get the full span and the (now actionable)
refusal.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel import BACKEND
from gptqmodel.models.loader import _resolve_default_layerwise_device
from gptqmodel.utils.importer import (
    _selector_devices,
    _validate_selector_device_families,
)


pytestmark = [pytest.mark.cpu]


def _gpu_capabilities(monkeypatch, capabilities):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.get_device_capability",
        lambda device=None: capabilities[torch.device(device).index],
    )
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.device_count",
        lambda: len(capabilities),
    )


def _default_targets():
    """The exact chain ``from_quantized`` runs before kernel selection."""

    targets = _selector_devices(_resolve_default_layerwise_device(None, BACKEND.AUTO))
    _validate_selector_device_families(targets)
    return targets


def test_default_load_stays_inside_one_capability_group(monkeypatch):
    _gpu_capabilities(monkeypatch, [(12, 0), (8, 9)])

    # Before the fix this raised "cuda:0=12.0, cuda:1=8.9".
    assert _default_targets() == (torch.device("cuda:0"),)


def test_default_load_still_uses_every_gpu_of_one_generation(monkeypatch):
    _gpu_capabilities(monkeypatch, [(8, 9), (8, 9)])

    assert _default_targets() == (torch.device("cuda:0"), torch.device("cuda:1"))


def test_a_single_visible_gpu_is_unaffected(monkeypatch):
    _gpu_capabilities(monkeypatch, [(8, 9)])

    assert _default_targets() == (torch.device("cuda:0"),)


def test_explicit_mixed_placement_is_refused_with_the_remedy(monkeypatch):
    """An explicit span is a user choice, so it keeps failing closed."""

    _gpu_capabilities(monkeypatch, [(12, 0), (8, 9)])

    with pytest.raises(ValueError, match="different GPU compute capabilities") as error:
        _resolve_default_layerwise_device(
            (torch.device("cuda:0"), torch.device("cuda:1")), BACKEND.AUTO
        )

    assert "CUDA_VISIBLE_DEVICES" in str(error.value)


def test_explicit_single_device_is_unaffected(monkeypatch):
    _gpu_capabilities(monkeypatch, [(12, 0), (8, 9)])

    assert _selector_devices(
        _resolve_default_layerwise_device(torch.device("cuda:1"), BACKEND.AUTO)
    ) == (torch.device("cuda:1"),)


def test_mixed_capability_refusal_names_a_way_forward(monkeypatch):
    _gpu_capabilities(monkeypatch, [(12, 0), (8, 9)])

    with pytest.raises(ValueError, match="different GPU compute capabilities") as error:
        _validate_selector_device_families(
            (torch.device("cuda:0"), torch.device("cuda:1"))
        )

    message = str(error.value)
    assert "cuda:0=12.0" in message
    assert "cuda:1=8.9" in message
    assert "CUDA_VISIBLE_DEVICES" in message
    assert "max_memory" in message
