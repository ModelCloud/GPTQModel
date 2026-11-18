# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5084: the HF bridge must not validate a keyword map too early.

Transformers calls the quantizer's ``preprocess_model`` *before* it resolves
``device_map`` from ``max_memory``. The bridge therefore saw only the keyword
``"auto"``, expanded it to every visible GPU, and kernel selection refused the
load when those GPUs spanned compute-capability groups -- even when the budget
named one card. ``GPTQModel.load`` already resolved ``max_memory`` first
(#4632/#4747), so only the plain-Transformers path failed.

These nodes drive the exact chain a keyword map takes through the bridge and
pin both sides of the contract: a single-card budget loads, and a resolved map
that really spans generations is still refused, now with a remedy that can
work on this path.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.utils.importer import (
    _selector_devices,
    hf_normalize_device_device_map,
    hf_select_quant_linear,
    normalize_device_device_map,
)


pytestmark = [pytest.mark.cpu]

# Selector validation ignores the CUDA capability tuple on ROCm (the HIP tuple
# is not an NVIDIA SM version), so a fake capability pair cannot group there.
requires_nvidia_capability = pytest.mark.skipif(
    torch.version.hip is not None,
    reason="ROCm ignores the CUDA capability tuple used by these fakes",
)


def _gpu_capabilities(monkeypatch, capabilities):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.get_device_capability",
        lambda device=None: capabilities[torch.device(device).index],
    )
    monkeypatch.setattr(
        "gptqmodel.utils.importer.torch.cuda.device_count",
        lambda: len(capabilities),
    )


@requires_nvidia_capability
@pytest.mark.parametrize("keyword", ["auto", "balanced", "sequential", "balanced_low_0"])
def test_keyword_map_is_not_rejected_before_max_memory(monkeypatch, keyword):
    """`preprocess_model` runs first, so the keyword map cannot be refused yet.

    Before the fix this raised "cuda:0=8.9, cuda:1=12.0" for every keyword,
    which is what made `from_pretrained(device_map="auto", max_memory={1: ...})`
    fail on a mixed-generation host.
    """

    _gpu_capabilities(monkeypatch, [(8, 9), (12, 0)])

    assert hf_normalize_device_device_map(None, keyword) == DEVICE.CUDA


@requires_nvidia_capability
def test_keyword_map_keeps_every_visible_device_selectable(monkeypatch):
    """The unresolved map must select a kernel valid on *every* visible card.

    A concrete expansion would pick a kernel for `cuda:0`; the budget may name
    `cuda:1`, whose capability group can need a different kernel. The family
    selector validates all visible devices instead.
    """

    _gpu_capabilities(monkeypatch, [(8, 9), (12, 0)])

    # The family enum (not an expanded ordinal) is the unresolved placement.
    assert hf_normalize_device_device_map(None, "auto") == DEVICE.CUDA

    # Kernel selection must succeed rather than raising the capability error.
    assert hf_select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        checkpoint_format="gptq",
        device_map="auto",
    ) is not None


@requires_nvidia_capability
def test_a_homogeneous_host_keeps_the_previous_expansion(monkeypatch):
    """Uniform capability groups keep the concrete expansion, so nothing changes.

    Deferring to the family on a uniform host would silently relax kernel
    selection there, so the historical behaviour is preserved exactly.
    """

    _gpu_capabilities(monkeypatch, [(8, 9), (8, 9)])

    assert _selector_devices(hf_normalize_device_device_map(None, "auto")) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )


@requires_nvidia_capability
def test_an_explicit_mixed_map_is_still_refused(monkeypatch):
    """An explicit user choice keeps failing closed before any weight loads."""

    _gpu_capabilities(monkeypatch, [(8, 9), (12, 0)])

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        hf_normalize_device_device_map(None, {"": "cuda:0", "model.layers.1": "cuda:1"})

    with pytest.raises(ValueError, match="different GPU compute capabilities"):
        normalize_device_device_map(None, {"": 0, "lm_head": 1})
