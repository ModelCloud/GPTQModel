# SPDX-License-Identifier: Apache-2.0
"""Allocator telemetry must follow the active accelerator, never torch.cuda."""

import torch

from gptqmodel.utils import torch as gm_torch


def test_cpu_reports_no_allocator_stats():
    """A device without an accelerator allocator stays empty instead of raising."""

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}


def test_cpu_has_no_allocator_stats_callable():
    assert gm_torch.resolve_memory_stats_callable("cpu") is None


def test_missing_allocator_reports_empty_stats(monkeypatch):
    """When the device type has no usable allocator the query returns an empty mapping."""

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: None)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}


def test_stats_come_from_the_requested_device_allocator(monkeypatch):
    """The active accelerator's allocator is queried with the requested device."""

    queried = []

    def fake_allocator(device):
        queried.append(device)
        return {"active_bytes.all.peak": 7}

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: fake_allocator)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {"active_bytes.all.peak": 7}
    assert queried == [torch.device("cpu")]


def test_allocator_failures_do_not_break_telemetry(monkeypatch):
    """Telemetry is best effort: a failing allocator query yields an empty mapping."""

    def failing_allocator(device):
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(gm_torch, "resolve_memory_stats_callable", lambda device_type: failing_allocator)

    assert gm_torch.memory_stats_for_device(torch.device("cpu")) == {}
