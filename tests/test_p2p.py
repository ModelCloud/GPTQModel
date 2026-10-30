# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
from models.model_test import ModelTest

from gptqmodel.utils import p2p


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    ndev = torch.cuda.device_count()
    if ndev < 2:
        print(f"Only {ndev} CUDA device(s) visible, need >= 2")
        return

    print("Devices:")
    for d in range(ndev):
        props = torch.cuda.get_device_properties(d)
        print(f"  cuda:{d} {props.name} {props.total_memory/1024**3:.1f} GiB CC {props.major}.{props.minor}")

    print("\nP2P capability (rows=src, cols=dst):")
    for i in range(ndev):
        row = []
        for j in range(ndev):
            if i == j:
                row.append("  - ")
                continue
            row.append("yes" if torch.cuda.can_device_access_peer(i, j) else " no")
        print(f"{i:>2}: " + " ".join(f"{r:>3}" for r in row))


def _fake_topology(monkeypatch, capability, *, ndev=2, bandwidth=None, enabled=None):
    """Install a fake multi-device CUDA/HIP runtime for the P2P helpers."""
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: ndev)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda src, dst: bool(capability.get((src, dst), False)),
    )

    calls = []

    def fake_enable(src, dst):
        calls.append((src, dst))
        if enabled is None:
            return True, "fake"
        return enabled(src, dst)

    monkeypatch.setattr(p2p, "_enable_one", fake_enable)

    def fake_bandwidth(src, dst):
        if bandwidth is None:
            return 100.0, 50.0
        return bandwidth(src, dst)

    monkeypatch.setattr(p2p, "_probe_bandwidth", fake_bandwidth)
    return calls


def test_peer_matrix_symmetric_and_reflexive(monkeypatch):
    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    matrix = p2p.peer_matrix([0, 1])
    assert len(matrix) == 2
    # Reflexive: every device can address itself.
    assert matrix[0][0] is True and matrix[1][1] is True
    # Symmetric topology is reported as such.
    assert matrix[0][1] is matrix[1][0] is True

    # Directed asymmetry must not be silently symmetrised.
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda s, d: (s, d) == (0, 1))
    directed = p2p.peer_matrix([0, 1])
    assert directed[0][1] is True and directed[1][0] is False


def test_enable_peer_access_idempotent(monkeypatch):
    calls = _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    first = p2p.enable_peer_access([0, 1])
    assert sorted(calls) == [(0, 1), (1, 0)]
    assert first.state(0, 1) == p2p.STATE_ENABLED
    assert first.state(1, 0) == p2p.STATE_ENABLED

    second = p2p.enable_peer_access([0, 1])
    assert second is first
    # Cached: no second enablement call per direction.
    assert sorted(calls) == [(0, 1), (1, 0)]


def test_p2p_env_disable(monkeypatch):
    calls = _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    monkeypatch.setenv("GPTQMODEL_P2P", "0")
    status = p2p.enable_peer_access([0, 1])
    assert calls == []
    assert status.state(0, 1) == p2p.STATE_UNSUPPORTED
    assert status.can_copy(0, 1) is False


def test_p2p_bandwidth_gate(monkeypatch):
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (1.0, 0.5) if (src, dst) == (0, 1) else (100.0, 50.0),
    )
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    status = p2p.enable_peer_access([0, 1])
    # Capability true but slow: degraded, and direct copy is refused.
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.state(1, 0) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is False
    assert status.can_copy(1, 0) is True
    # One degraded direction forces the whole status off the direct path.
    assert status.direct_copy_allowed() is False


def test_p2p_slower_than_host_staging_is_degraded(monkeypatch):
    # The absolute floor alone cannot catch this: a direct copy can clear any low
    # GB/s threshold while still being slower than the host-staged fallback.
    # Measured on 2x RTX 4090 (0.998x) and 4x R9700 (0.954x-0.99x):
    # direct ran no faster than pinned-host staging.
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (40.0, 45.0),
    )
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.bandwidth_gbps[(0, 1)] == 40.0
    assert status.host_staged_gbps[(0, 1)] == 45.0
    assert status.can_copy(0, 1) is False
    assert status.direct_copy_allowed() is False


def test_p2p_ratio_gate_can_be_relaxed_by_env(monkeypatch):
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (40.0, 45.0))
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_RATIO", "0.8")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED


def test_p2p_unmeasurable_direct_link_is_degraded(monkeypatch):
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (None, 50.0))
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.can_copy(0, 1) is False


def test_multi_gpu_direct_copy(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA/HIP devices required")

    # The mocked cases above cache synthetic statuses for the same device tuple;
    # this test needs the real probe, so drop them (and any probe bandwidth).
    p2p.clear_p2p_cache()
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("device 0 cannot access peer device 1")

    status = p2p.enable_peer_access([0, 1])
    if not status.can_copy(0, 1):
        pytest.skip(f"peer link 0->1 not usable: {status.summary()}")

    src = torch.arange(4096, device="cuda:0", dtype=torch.int32)
    dst = torch.zeros_like(src, device="cuda:1")
    p2p.copy_tensor_to(dst, src, status)
    torch.cuda.synchronize(1)
    assert torch.equal(dst.cpu(), src.cpu())


class Test(ModelTest):
    def test(self):
        main()
