# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import pytest

from scripts import gpu_idle_preflight


GPU_UUID = "GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855"
INVENTORY = f"1, 00000000:2B:00.0, {GPU_UUID}, PG506-232, 0, 0\n"


def _completed(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def _install_queries(monkeypatch, *, inventory: str = INVENTORY, processes: str = "") -> list[list[str]]:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu" in command:
            return _completed(inventory)
        if "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in command:
            return _completed(processes)
        raise AssertionError(command)

    monkeypatch.setattr(gpu_idle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(gpu_idle_preflight.time, "sleep", lambda _: None)
    return calls


def test_bootstrap_accepts_three_idle_uuid_pinned_samples(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    calls = _install_queries(monkeypatch)

    state = gpu_idle_preflight.bootstrap_gpu_idle_preflight(
        ["--idle-samples", "3", "--idle-interval-seconds", "0"]
    )

    assert state is not None
    assert state.physical_id == 1
    assert state.pci_bus_id == "00000000:2B:00.0"
    assert len(state.samples) == 3
    assert all(command[1:3] == ["-i", GPU_UUID] for command in calls)


def test_bootstrap_rejects_foreign_compute_process(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    _install_queries(monkeypatch, processes=f"{GPU_UUID}, 1234, python, 1024\n")

    with pytest.raises(RuntimeError, match="foreign_processes"):
        gpu_idle_preflight.bootstrap_gpu_idle_preflight(["--idle-interval-seconds", "0"])


def test_recheck_accepts_only_current_process_and_driver_baseline(monkeypatch):
    own_pid = os.getpid()
    inventory = f"1, 00000000:2B:00.0, {GPU_UUID}, PG506-232, 4108, 9\n"
    processes = f"{GPU_UUID}, {own_pid}, python, 4096\n"
    _install_queries(monkeypatch, inventory=inventory, processes=processes)
    initial = {
        "physical_id": 1,
        "pci_bus_id": "00000000:2B:00.0",
        "uuid": GPU_UUID,
        "name": "PG506-232",
        "memory_used_mib": 0,
        "utilization_pct": 0,
        "compute_processes": (),
    }
    state = gpu_idle_preflight.GPUIdlePreflightState(
        physical_id=1,
        pci_bus_id="00000000:2B:00.0",
        uuid=GPU_UUID,
        name="PG506-232",
        config=gpu_idle_preflight.GPUIdlePreflightConfig(3, 0.0, 16),
        samples=(initial, initial, initial),
    )

    snapshot = gpu_idle_preflight.recheck_gpu_exclusivity(state)

    assert snapshot["memory_used_mib"] == 4108


def test_recheck_rejects_unattributed_memory(monkeypatch):
    own_pid = os.getpid()
    inventory = f"1, 00000000:2B:00.0, {GPU_UUID}, PG506-232, 4200, 0\n"
    processes = f"{GPU_UUID}, {own_pid}, python, 4096\n"
    _install_queries(monkeypatch, inventory=inventory, processes=processes)
    initial = {
        "physical_id": 1,
        "pci_bus_id": "00000000:2B:00.0",
        "uuid": GPU_UUID,
        "name": "PG506-232",
        "memory_used_mib": 0,
        "utilization_pct": 0,
        "compute_processes": (),
    }
    state = gpu_idle_preflight.GPUIdlePreflightState(
        physical_id=1,
        pci_bus_id="00000000:2B:00.0",
        uuid=GPU_UUID,
        name="PG506-232",
        config=gpu_idle_preflight.GPUIdlePreflightConfig(3, 0.0, 16),
        samples=(initial, initial, initial),
    )

    with pytest.raises(RuntimeError, match="unexplained memory"):
        gpu_idle_preflight.recheck_gpu_exclusivity(state)


def test_preflight_requires_at_least_three_samples():
    args = SimpleNamespace(
        idle_samples=2,
        idle_interval_seconds=0.0,
        idle_max_driver_memory_mib=16,
    )

    with pytest.raises(ValueError, match="at least 3"):
        gpu_idle_preflight._config_from_args(args)


def test_bootstrap_help_does_not_query_gpu(monkeypatch):
    monkeypatch.setattr(
        gpu_idle_preflight.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("help must not query nvidia-smi"),
    )

    assert gpu_idle_preflight.bootstrap_gpu_idle_preflight(["--help"]) is None


def test_recheck_rejects_foreign_process_after_warmup(monkeypatch):
    own_pid = os.getpid()
    processes = (
        f"{GPU_UUID}, {own_pid}, python, 4096\n"
        f"{GPU_UUID}, 4321, python, 1024\n"
    )
    _install_queries(
        monkeypatch,
        inventory=f"1, 00000000:2B:00.0, {GPU_UUID}, PG506-232, 5132, 15\n",
        processes=processes,
    )
    base = gpu_idle_preflight.GPUIdlePreflightState(
        physical_id=1,
        pci_bus_id="00000000:2B:00.0",
        uuid=GPU_UUID,
        name="PG506-232",
        config=gpu_idle_preflight.GPUIdlePreflightConfig(3, 0.0, 16),
        samples=(),
    )
    state = replace(base, samples=({}, {}, {}))

    with pytest.raises(RuntimeError, match="foreign_processes"):
        gpu_idle_preflight.recheck_gpu_exclusivity(state)
