# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch

from gptqmodel.utils import looper_helpers as lh


def _stacked_linear(depth: int) -> torch.nn.Module:
    layers = [torch.nn.Linear(2, 2, bias=False) for _ in range(depth)]
    return torch.nn.Sequential(*layers)


def test_should_force_deepcopy_when_threshold_exceeded(monkeypatch):
    module = _stacked_linear(4)
    monkeypatch.setattr(lh, "REPLICATE_MODULE_THRESHOLD", 3, raising=False)

    should_force, module_count = lh.should_force_deepcopy_for_module(module)

    assert should_force
    assert module_count is not None
    assert module_count >= 3


def test_should_force_deepcopy_disabled_when_threshold_non_positive(monkeypatch):
    module = _stacked_linear(4)
    monkeypatch.setattr(lh, "REPLICATE_MODULE_THRESHOLD", 0, raising=False)

    should_force, module_count = lh.should_force_deepcopy_for_module(module)

    assert should_force is False
    assert module_count is None


def test_should_force_deepcopy_returns_count_below_threshold(monkeypatch):
    module = _stacked_linear(2)
    monkeypatch.setattr(lh, "REPLICATE_MODULE_THRESHOLD", 512, raising=False)

    should_force, module_count = lh.should_force_deepcopy_for_module(module)

    assert should_force is False
    assert module_count == sum(1 for _ in module.modules())
