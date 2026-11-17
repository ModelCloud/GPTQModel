# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Kernel selection must use the devices a ``max_memory`` budget names (#4632).

``max_memory`` is the authoritative placement list (#4082), but kernel
selection expanded the CUDA family to every visible GPU. On a mixed-generation
host (RTX 5090 sm_120 + RTX 4090 sm_89) the capability check then rejected
budgets that never touched the other card, including GPU+CPU offload budgets
with the portable Torch backend.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel import BACKEND
from gptqmodel.models.loader import (
    _resolve_default_layerwise_device,
    _resolve_max_memory_device,
)
from gptqmodel.utils.importer import (
    _selector_devices,
    _validate_selector_device_families,
)


def _mixed_capabilities(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.get_device_capability",
        lambda device=None: (12, 0) if torch.device(device).index == 0 else (8, 9),
    )
    monkeypatch.setattr("gptqmodel.utils.importer.torch.cuda.device_count", lambda: 2)


def _resolved_kernel_devices(max_memory):
    """The exact chain ``from_quantized`` runs before kernel selection."""

    device = _resolve_max_memory_device(max_memory, None)
    resolved = _resolve_default_layerwise_device(device, BACKEND.AUTO)
    targets = _selector_devices(resolved)
    _validate_selector_device_families(targets)
    return targets


@pytest.mark.parametrize(
    "max_memory, expected",
    [
        ({1: "20GiB"}, (torch.device("cuda:1"),)),
        ({0: "20GiB"}, (torch.device("cuda:0"),)),
        ({0: "600MiB", "cpu": "20GiB"}, (torch.device("cuda:0"),)),
        ({"cuda:1": "20GiB"}, (torch.device("cuda:1"),)),
    ],
)
def test_single_gpu_budget_never_validates_the_other_card(
    monkeypatch, max_memory, expected
):
    _mixed_capabilities(monkeypatch)

    assert _resolve_max_memory_device(max_memory, None) == expected
    # Before the fix this raised "cuda:0=12.0, cuda:1=8.9".
    assert _resolved_kernel_devices(max_memory) == expected


def test_a_budget_that_names_both_cards_still_rejects_mixed_capabilities(monkeypatch):
    _mixed_capabilities(monkeypatch)

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        _resolved_kernel_devices({1: "20GiB", 0: "20GiB"})


def test_cpu_only_budget_still_selects_cpu_kernels():
    from gptqmodel.models._const import DEVICE

    assert _resolve_max_memory_device({"cpu": "20GiB"}, None) == DEVICE.CPU


def test_budget_device_family_must_match_an_explicit_device(monkeypatch):
    _mixed_capabilities(monkeypatch)

    with pytest.raises(ValueError, match="does not fund it"):
        _resolve_max_memory_device({0: "20GiB"}, "cuda:1")

    assert _resolve_max_memory_device({1: "20GiB"}, "cuda:1") == (
        torch.device("cuda:1"),
    )


def test_cpu_only_budget_rejects_an_accelerator_device():
    with pytest.raises(ValueError, match="no accelerator entry"):
        _resolve_max_memory_device({"cpu": "20GiB"}, "cuda:1")
