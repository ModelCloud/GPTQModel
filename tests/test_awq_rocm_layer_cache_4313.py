# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4313: AWQ releases its allocator cache per layer on ROCm.

AWQ's clip search allocates a ~256 MiB scratch buffer per clip-eligible module.
On ROCm the default ``max_split_size_mb:256`` prevents the caching allocator
from reusing a freed 256 MiB block for the next module's slightly smaller
request, so reserved VRAM grew by roughly 1 GiB per decoder layer. ``apply_clip``
is the once-per-layer boundary where those scratch tensors have already been
released, so it must hand the unused cache back to the driver exactly once.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from torch import nn

from gptqmodel.quantization.awq.quantize import scale


pytestmark = [pytest.mark.cpu]


def _module_with_two_linears() -> nn.Sequential:
    module = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    with torch.no_grad():
        module[0].weight.fill_(2.0)
        module[1].weight.fill_(-2.0)
    return module


def _clip_list() -> list[tuple[str, torch.Tensor]]:
    return [("0", torch.ones(8, 1, 1)), ("1", torch.ones(8, 1, 1))]


def test_apply_clip_releases_rocm_cache_once_per_layer(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    # Two clipped modules still belong to one layer, so the release runs once.
    scale.apply_clip(module, _clip_list())

    assert calls == ["empty"]
    # The clamp still ran (the objective must not change).
    assert torch.all(module[0].weight <= 1.0)
    assert torch.all(module[1].weight >= -1.0)


def test_apply_clip_releases_cache_on_resolved_quant_device(monkeypatch):
    module = _module_with_two_linears()
    calls: list[torch.device] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        scale,
        "_resolve_module_exec_device",
        lambda _module: torch.device("cuda", 1),
    )
    # The device is only recorded here; do not require a real GPU to move to it.
    monkeypatch.setattr(scale, "move_to", lambda _module, device=None: None)

    def _record_device(device):
        calls.append(torch.device(device))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.cuda, "device", _record_device)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    scale.apply_clip(module, _clip_list())

    assert calls == [torch.device("cuda", 1)]


def test_apply_clip_does_not_release_cache_on_cuda(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    scale.apply_clip(module, _clip_list())

    assert calls == []


def test_apply_clip_skips_release_without_cuda(monkeypatch):
    module = _module_with_two_linears()
    calls: list[str] = []

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

    scale.apply_clip(module, _clip_list())

    assert calls == []


def test_apply_clip_never_fails_on_allocator_error(monkeypatch):
    module = _module_with_two_linears()

    monkeypatch.setattr(scale, "running_on_rocm", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: contextlib.nullcontext())

    def _boom():
        raise RuntimeError("allocator is unavailable")

    monkeypatch.setattr(torch.cuda, "empty_cache", _boom)

    # Cleanup is best-effort: a failure must not abort quantization.
    scale.apply_clip(module, _clip_list())
    assert torch.all(module[0].weight <= 1.0)
