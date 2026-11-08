# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Placement contract for the default layer-wise device map.

``device=("cuda:1", "cuda:2")`` means *those two GPUs*. The map builder used to
ignore the sequence and span every visible accelerator, so asking for a subset
silently used all of them. These tests pin the helper contract without needing
accelerators: the visible-device count is monkeypatched.
"""

import torch

from gptqmodel.models.loader import (
    _explicit_layerwise_devices,
    _explicit_requested_devices,
    _layerwise_device_count,
    _layerwise_device_strings,
)


def _fake_visible(monkeypatch, n):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_explicit_devices_accepts_concrete_ordinals():
    assert _explicit_layerwise_devices(("cuda:1", "cuda:2")) == ["cuda:1", "cuda:2"]
    assert _explicit_layerwise_devices(["cuda:0", "cuda:3"]) == ["cuda:0", "cuda:3"]
    assert _explicit_layerwise_devices(
        (torch.device("cuda:2"), torch.device("cuda:1"))
    ) == ["cuda:2", "cuda:1"]


def test_duplicate_explicit_devices_do_not_expand_to_every_gpu(monkeypatch):
    _fake_visible(monkeypatch, 4)
    requested = ("cuda:1", "cuda:1")

    assert _explicit_requested_devices(requested) == [torch.device("cuda:1")]
    assert _layerwise_device_count(requested) == 1
    assert _layerwise_device_strings(requested, 1) == ["cuda:1"]


def test_explicit_devices_rejects_non_ordinal_selectors():
    """An abstract family, a bare name, CPU, or a single entry keeps the old path."""
    assert _explicit_layerwise_devices("cuda") is None
    assert _explicit_layerwise_devices("cuda:1") is None
    assert _explicit_layerwise_devices(("cuda", "cuda:1")) is None
    assert _explicit_layerwise_devices(("cuda:1", "cpu")) is None
    # a single-element sequence is not a placement request
    assert _explicit_layerwise_devices(("cuda:1",)) is None


def test_requested_devices_are_concrete_torch_devices():
    requested = _explicit_requested_devices(("cuda:1", "cuda:2"))
    assert requested == [torch.device("cuda:1"), torch.device("cuda:2")]
    assert all(isinstance(target, torch.device) for target in requested)


def test_count_honours_requested_subset(monkeypatch):
    _fake_visible(monkeypatch, 4)
    assert _layerwise_device_count(("cuda:1", "cuda:2")) == 2
    assert _layerwise_device_count(["cuda:0", "cuda:1", "cuda:2", "cuda:3"]) == 4
    # abstract family still spans every visible device
    assert _layerwise_device_count("cuda") == 4


def test_strings_render_requested_subset(monkeypatch):
    _fake_visible(monkeypatch, 4)
    assert _layerwise_device_strings(("cuda:1", "cuda:2"), 2) == ["cuda:1", "cuda:2"]
    assert _layerwise_device_strings(["cuda:2", "cuda:3"], 2) == ["cuda:2", "cuda:3"]
    # family selector keeps the 0..n-1 rendering
    assert _layerwise_device_strings("cuda", 4) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]
