# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

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
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    status = p2p.enable_peer_access([0, 1])
    # Capability true but slow: degraded, and direct copy is refused.
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.state(1, 0) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is False
    assert status.can_copy(1, 0) is True
    # One degraded direction forces the whole status off the direct path.
    assert status.direct_copy_allowed() is False


def test_p2p_parity_with_host_staging_stays_enabled(monkeypatch):
    # Measured on this host: enabling peer access does not add bandwidth (both
    # rigs are PCIe-attached, no NVLink/Infinity Fabric). Direct/pinned-host
    # ratios are 0.92-1.00, i.e. parity. Parity must NOT be downgraded to a CPU
    # bounce: ``degraded`` is reserved for a link materially slower than staging.
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (45.0, 45.0),
    )
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is True
    assert status.direct_copy_allowed() is True


def test_p2p_materially_slower_than_host_staging_is_degraded(monkeypatch):
    # A link far below the fallback (e.g. an emulated/degraded peer path) is
    # still rejected, so the ratio gate is not a no-op.
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (10.0, 45.0),
    )
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.bandwidth_gbps[(0, 1)] == 10.0
    assert status.host_staged_gbps[(0, 1)] == 45.0
    assert status.can_copy(0, 1) is False
    assert status.direct_copy_allowed() is False


def test_p2p_ratio_gate_can_be_tightened_by_env(monkeypatch):
    # A stricter 1.1x requirement (direct must strictly beat staging) rejects a
    # parity link; the default does not.
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (45.0, 45.0))
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_RATIO", "1.1")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED


def test_copy_tensor_to_never_bounces_through_host(monkeypatch):
    """A degraded verdict must not be turned into an explicit CPU relay.

    Production transfers go through ``dst.copy_(src)``, where the driver picks
    the peer or staged path. An explicit ``src.to("cpu")`` here would be the
    ``gpu1 -> CPU -> gpu2`` relay multi-GPU quantization must avoid.
    """
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (1.0, 100.0))
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED

    to_cpu_calls = []
    original_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        if args and args[0] == "cpu":
            to_cpu_calls.append(args[0])
        return original_to(self, *args, **kwargs)

    dst = torch.zeros(8)
    src = torch.arange(8)
    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    p2p.copy_tensor_to(dst, src, status)
    assert to_cpu_calls == []
    assert torch.equal(dst, src)


def test_p2p_unmeasurable_direct_link_is_degraded(monkeypatch):
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (None, 50.0))
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
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


def test_live_peer_matrix_invariants():
    """The live peer matrix must be reflexive and match the runtime's own view.

    Unlike the mocked cases this reads the real device set, so it fails if the
    probe, index normalization, or direction handling regresses on hardware.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA/HIP devices required")

    p2p.clear_p2p_cache()
    indices = (0, 1)
    matrix = p2p.peer_matrix(indices)
    for row, src in enumerate(indices):
        for col, dst in enumerate(indices):
            if src == dst:
                assert matrix[row][col] is True
            else:
                assert matrix[row][col] is bool(torch.cuda.can_device_access_peer(src, dst))


def test_enable_for_devices_wrapper(monkeypatch):
    # ``enable_for_devices`` is the entry point the quantization loops call.
    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    status = p2p.enable_for_devices([0, 1])
    assert status.devices == (0, 1)
    assert status.state(0, 1) == p2p.STATE_ENABLED


def test_p2p_probe_is_resilient_to_a_noisy_sample(monkeypatch):
    # A single unlucky bandwidth sample must not demote a healthy link: the probe
    # measures each direction twice and keeps the best sample. Regression for a
    # real run where one direction of a pair flipped to ``degraded`` seconds after
    # measuring enabled, purely from sampling noise under memory pressure.
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda s, d: True)
    monkeypatch.setattr(p2p, "_enable_one", lambda s, d: (True, "fake"))

    calls = {"n": 0}

    def flaky(src, dst, bytes_moved, iterations=5):
        # First sample for each direction is a transient dip, the second recovers.
        calls["n"] += 1
        return 10.0 if calls["n"] % 2 == 1 else 54.0

    monkeypatch.setattr(p2p, "_measure_gbps", flaky)
    monkeypatch.setattr(p2p, "_measure_host_staged_gbps", lambda *a, **k: 55.0)
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")

    direct, staged = p2p._probe_bandwidth(0, 1)
    assert direct == 54.0
    assert staged == 55.0


def test_p2p_forced_mode_ignores_relative_gate(monkeypatch):
    # ``GPTQMODEL_P2P=1`` asks for the direct path explicitly; a link slower than
    # staging must not be vetoed by the relative comparison (the absolute floor
    # still applies).
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (20.0, 45.0))
    monkeypatch.setenv("GPTQMODEL_P2P", "1")
    monkeypatch.setenv("GPTQMODEL_P2P_MIN_GBPS", "5")
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE", "1")
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is True


def test_p2p_probe_is_opt_in(monkeypatch):
    """Enabling must not move probe payload unless diagnostics were requested.

    PyTorch already enables peer access in its own copy path, so measuring every
    directed pair at startup only adds latency; the probe is therefore opt-in.
    """
    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})

    def fail(*args, **kwargs):
        raise AssertionError("bandwidth probe must not run without GPTQMODEL_P2P_PROBE=1")

    monkeypatch.setattr(p2p, "_probe_bandwidth", fail)
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.state(1, 0) == p2p.STATE_ENABLED
    assert status.bandwidth_gbps == {}


def test_indexless_string_and_device_mean_current_device(monkeypatch):
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert p2p._normalize_indices(["cuda", "cuda:1"]) == (2, 1)
    assert p2p._normalize_indices([torch.device("cuda"), torch.device("cuda:3")]) == (2, 3)
    assert p2p._normalize_indices(["cpu", 1]) == (1,)


def test_probe_mib_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE_MIB", "not-a-number")
    assert p2p._probe_mib() == 64
    monkeypatch.setenv("GPTQMODEL_P2P_PROBE_MIB", "8")
    assert p2p._probe_mib() == 8


def test_strict_mode_raises_even_when_best_effort_result_is_cached(monkeypatch):
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        enabled=lambda src, dst: (False, "boom") if (src, dst) == (0, 1) else (True, "fake"),
    )
    # Automatic (best-effort) call populates the cache with the failure.
    best_effort = p2p.enable_for_devices([0, 1])
    assert best_effort.state(0, 1) == p2p.STATE_UNSUPPORTED
    assert best_effort.errors[(0, 1)] == "boom"

    with pytest.raises(RuntimeError, match="boom"):
        p2p.enable_peer_access([0, 1], strict=True)
    with pytest.raises(RuntimeError, match="boom"):
        p2p.enable_for_devices([0, 1], strict=True)


def test_strict_mode_reraises_directly(monkeypatch):
    _fake_topology(
        monkeypatch,
        {(0, 1): True},
        enabled=lambda src, dst: (False, "no link"),
    )
    with pytest.raises(RuntimeError, match="no link"):
        p2p.enable_for_devices([0, 1], strict=True)


def test_host_staged_probe_orders_the_two_legs(monkeypatch):
    """The D2H leg must complete before the H2D leg reads the staging buffer."""
    p2p.clear_p2p_cache()
    events = []

    class FakeTensor:
        def __init__(self, device):
            self.device = device

        def copy_(self, other, non_blocking=False):
            events.append(("copy", self.device, getattr(other, "device", "cpu")))
            return self

    monkeypatch.setattr(p2p.torch, "empty", lambda *a, **k: FakeTensor(k.get("device") or "cpu"))
    monkeypatch.setattr(p2p.torch.cuda, "synchronize", lambda dev: events.append(("sync", dev)))
    monkeypatch.setattr(p2p, "_timed_gbps", lambda step, *a, **k: (step(), 100.0)[1])

    p2p._measure_host_staged_gbps(0, 1, 1024)
    kinds = [e[0] for e in events]
    assert kinds == ["copy", "sync", "copy"], events
    assert events[1][1] == 0
    assert events[2] == ("copy", "cuda:1", "cpu")


def test_device_map_targets_are_normalized():
    """String, integer and torch.device map targets must all be honoured."""
    device_map = {"a": "cuda:0", "b": 1, "c": torch.device("cuda", 1), "d": "cpu", "e": None}
    assert p2p.accelerator_devices_from_device_map(device_map) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
    assert p2p.accelerator_devices_from_device_map({"a": 0, "b": 1}) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
