# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import subprocess
import sys
import textwrap
import threading

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.utils import p2p


_REAL_RAW_PEER_COPY_UNSAFE = p2p._raw_peer_copy_unsafe
_REAL_ROCM_PEER_LINK_UNUSABLE = p2p._rocm_peer_link_unusable
_REAL_PYTORCH_OWNS_ENABLEMENT = p2p._pytorch_owns_peer_enablement


@pytest.fixture(autouse=True)
def _isolate_p2p_config():
    """Keep the process-wide P2P policy from leaking between tests."""

    previous = p2p.get_p2p_config()
    p2p.configure_p2p(None)
    p2p.clear_p2p_cache()
    try:
        yield
    finally:
        p2p.configure_p2p(previous)
        p2p.clear_p2p_cache()


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
    # Synthetic topology tests do not run real copies. Live tests keep all
    # allocator guards, including raw-copy suppression, in place.
    monkeypatch.setattr(p2p, "_raw_peer_copy_unsafe", lambda: False)
    monkeypatch.setattr(p2p, "_rocm_peer_link_unusable", lambda: False)
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

    def fake_validation(src, dst, **_kwargs):
        direct, staged = fake_bandwidth(src, dst)
        return True, direct, staged, ""

    monkeypatch.setattr(p2p, "_validate_peer_copy_pair", fake_validation)
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


def test_policy_change_invalidates_a_cached_verdict(monkeypatch):
    """A cached classification must not survive a stricter policy (#3947)."""

    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (10.0, 1.0),
    )

    p2p.configure_p2p({"probe": True, "min_gbps": 5.0})
    loose = p2p.enable_peer_access([0, 1])
    assert loose.state(0, 1) == p2p.STATE_ENABLED

    # Same measured 10 GB/s link, stricter gate: it is not fast enough now.
    p2p.configure_p2p({"probe": True, "min_gbps": 20.0})
    strict = p2p.enable_peer_access([0, 1])

    assert strict is not loose
    assert strict.state(0, 1) == p2p.STATE_DEGRADED


def test_policy_change_during_a_probe_does_not_cache_a_stale_verdict(monkeypatch):
    """A probe that finishes after reconfiguration must not repopulate the cache.

    ``configure_p2p`` swaps the policy and clears the cache. A probe that was
    already running under the previous policy used to finish afterwards and
    reinsert its verdict, so a stricter ``min_gbps`` gate was silently bypassed
    for that device pair. The interleaving below is deterministic: thread A is
    parked inside the bandwidth probe while thread B installs the new policy.
    """

    started = threading.Event()
    release = threading.Event()
    probes = {"count": 0}

    def slow_bandwidth(src, dst):
        probes["count"] += 1
        if probes["count"] == 1:
            started.set()
            # Bounded so a stuck probe fails the test instead of hanging it.
            assert release.wait(10), "the reconfiguration never released the probe"
        return 10.0, 1.0

    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=slow_bandwidth,
    )

    p2p.configure_p2p({"probe": True, "min_gbps": 5.0})

    verdicts = {}

    def probe_under_loose_policy():
        verdicts["loose"] = p2p.enable_peer_access([0, 1])

    worker = threading.Thread(target=probe_under_loose_policy, name="p2p-probe")
    worker.start()
    assert started.wait(10), "the probe never started"

    # Thread B: 10 GB/s is below this new 20 GB/s gate.
    p2p.configure_p2p({"probe": True, "min_gbps": 20.0})
    release.set()
    worker.join(10)
    assert not worker.is_alive(), "the probe thread hung"

    # The in-flight verdict is still returned to its own caller, which measured
    # under the policy it started with...
    assert verdicts["loose"].state(0, 1) == p2p.STATE_ENABLED

    # ...but it is not reused under the policy that replaced it.
    current = p2p.enable_peer_access([0, 1])
    assert current is not verdicts["loose"]
    assert current.state(0, 1) == p2p.STATE_DEGRADED


def test_status_cache_is_keyed_by_policy_generation(monkeypatch):
    """The cache key must include the policy generation, not only the devices."""

    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (10.0, 1.0),
    )

    p2p.configure_p2p({"probe": True, "min_gbps": 5.0})
    p2p.enable_peer_access([0, 1])
    first_generation = p2p._policy_snapshot()[0]
    assert any(key[0] == first_generation for key in p2p._STATUS_CACHE)

    p2p.configure_p2p({"probe": True, "min_gbps": 20.0})
    second_generation = p2p._policy_snapshot()[0]
    assert second_generation > first_generation
    # The clear removed the old entries, so nothing can be answered from them.
    assert not any(key[0] == first_generation for key in p2p._STATUS_CACHE)


def test_p2p_config_disable(monkeypatch):
    calls = _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    p2p.configure_p2p(p2p.P2PConfig(mode="off"))
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
    p2p.configure_p2p(p2p.P2PConfig(min_gbps=5, probe=True))
    status = p2p.enable_peer_access([0, 1])
    # Capability true but slow: degraded, and direct copy is refused.
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.state(1, 0) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is False
    assert status.can_copy(1, 0) is True


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
    p2p.configure_p2p(p2p.P2PConfig(min_gbps=5, probe=True))
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is True


def test_p2p_materially_slower_than_host_staging_is_degraded(monkeypatch):
    # A link far below the fallback (e.g. an emulated/degraded peer path) is
    # still rejected, so the ratio gate is not a no-op.
    _fake_topology(
        monkeypatch,
        {(0, 1): True, (1, 0): True},
        bandwidth=lambda src, dst: (10.0, 45.0),
    )
    p2p.configure_p2p(p2p.P2PConfig(min_gbps=5, probe=True))
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.bandwidth_gbps[(0, 1)] == 10.0
    assert status.host_staged_gbps[(0, 1)] == 45.0
    assert status.can_copy(0, 1) is False


def test_p2p_ratio_gate_can_be_tightened(monkeypatch):
    # A stricter 1.1x requirement (direct must strictly beat staging) rejects a
    # parity link; the default does not.
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (45.0, 45.0))
    p2p.configure_p2p(p2p.P2PConfig(min_gbps=5, probe=True, min_ratio=1.1))
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED


def test_asymmetric_peer_topology_is_per_direction(monkeypatch):
    """Regression: ``0 -> 1`` unsupported while ``1 -> 0`` works.

    The rig that exposed the SIGSEGV reports exactly this matrix. A global
    "some direction works" verdict used to send the unsupported direction down
    ``hipMemcpyPeerAsync``, which crashes the process instead of raising.
    """
    _fake_topology(monkeypatch, {(1, 0): True})
    status = p2p.enable_peer_access([0, 1])

    assert status.can_copy(0, 1) is False
    assert status.can_copy(1, 0) is True
    assert p2p.can_transfer_directly("cuda:0", "cuda:1", status) is False
    assert p2p.can_transfer_directly("cuda:1", "cuda:0", status) is True
    assert not hasattr(status, "direct_copy_allowed")


def test_safe_move_to_stages_unsupported_direction(monkeypatch):
    """The unsupported direction must relay through host memory."""
    staged = []

    def fake_stage(tensor, device, *, dtype=None, copy=False):
        staged.append((tensor.device, torch.device(device)))
        return tensor.clone()

    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *args, **kwargs: False)
    monkeypatch.setattr(p2p, "_stage_through_host", fake_stage)

    src = torch.arange(8)
    moved = p2p.safe_move_to(src, "cuda:1")
    assert staged == [(src.device, torch.device("cuda:1"))]
    assert torch.equal(moved, src)


def test_safe_move_to_uses_direct_path_when_supported(monkeypatch):
    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        p2p,
        "_stage_through_host",
        lambda *args, **kwargs: pytest.fail("healthy link must not be staged"),
    )
    src = torch.arange(8)
    moved = p2p.safe_move_to(src, torch.device("cpu"), copy=True)
    assert torch.equal(moved, src)
    assert moved is not src


@pytest.mark.parametrize("device", [DEVICE.CUDA, DEVICE.ROCM])
def test_safe_moves_with_family_enum_use_current_device(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *a, **k: False)
    staged = []

    def fake_stage(tensor, target, **kwargs):
        staged.append(target)
        return tensor.clone()

    monkeypatch.setattr(p2p, "_stage_through_host", fake_stage)
    src = torch.arange(8)
    assert torch.equal(p2p.safe_move_to(src, device), src)
    assert staged == [torch.device("cuda:1")]

    module = torch.nn.Linear(2, 2)
    targets = []
    monkeypatch.setattr(module, "to", lambda **kwargs: targets.append(kwargs["device"]) or module)
    assert p2p.safe_module_to(module, device) is module
    assert targets[-1] == torch.device("cuda:1")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA/HIP devices")
@pytest.mark.parametrize("device", [DEVICE.CUDA, DEVICE.ROCM])
def test_family_enum_moves_on_nonzero_current_device(device):
    src = torch.arange(8)
    module = torch.nn.Linear(8, 2)
    with torch.no_grad():
        module.weight.copy_(torch.arange(16).reshape(2, 8) / 16)
        module.bias.copy_(torch.arange(2) / 8)
    expected = module(src.float()).detach()

    with torch.cuda.device(1):
        moved = p2p.safe_move_to(src, device)
        p2p.safe_module_to(module, device)
        assert moved.device == torch.device("cuda:1")
        assert module.weight.device == torch.device("cuda:1")
        assert torch.equal(moved.cpu(), src)
        torch.testing.assert_close(module(moved.float()).cpu(), expected)
        # Explicit ordinals must still override the current-device context.
        assert p2p.safe_move_to(src, "cuda:0").device == torch.device("cuda:0")


def test_asymmetric_pairs_lists_only_the_broken_direction(monkeypatch):
    _fake_topology(monkeypatch, {(1, 0): True})
    status = p2p.enable_peer_access([0, 1])
    assert status.asymmetric_pairs() == [(0, 1)]


def test_forced_peer_access_is_not_trusted_when_capability_denies(monkeypatch):
    """``P2PConfig(mode='on')`` can make enablement report success on a denied pair.

    That is not evidence of a usable link: a direct copy there still crashes the
    ROCm rig, so the direction must stay host-staged even when forced.
    """
    _fake_topology(monkeypatch, {(1, 0): True})
    p2p.configure_p2p(p2p.P2PConfig(mode="on"))

    status = p2p.enable_peer_access([0, 1])

    assert status.state(0, 1) == p2p.STATE_UNVERIFIED
    assert status.can_copy(0, 1) is False
    assert status.unverified_pairs() == [(0, 1)]
    assert p2p.can_transfer_directly("cuda:0", "cuda:1", status) is False
    assert p2p.can_transfer_directly("cuda:1", "cuda:0", status) is True


def test_forced_unverified_pair_is_never_probed(monkeypatch):
    """The probe performs a direct copy, so it must not run on a denied pair."""
    probed = []

    def fake_bandwidth(src, dst):
        probed.append((src, dst))
        return 100.0, 50.0

    _fake_topology(monkeypatch, {(1, 0): True}, bandwidth=fake_bandwidth)
    p2p.configure_p2p(p2p.P2PConfig(mode="on", probe=True))

    status = p2p.enable_peer_access([0, 1])

    assert (0, 1) not in probed
    assert (1, 0) in probed
    assert status.state(0, 1) == p2p.STATE_UNVERIFIED


def test_forced_unverified_pair_counts_as_asymmetric(monkeypatch):
    """A forced-on but capability-denied direction is still unusable.

    ``clone_module_for_devices`` skips NCCL replicate on ``asymmetric_pairs()``;
    the forced state must therefore be reported there too, otherwise a rig that
    denies ``0 -> 1`` would still attempt the collective path.
    """
    _fake_topology(monkeypatch, {(1, 0): True})
    p2p.configure_p2p(p2p.P2PConfig(mode="on"))

    status = p2p.enable_peer_access([0, 1])

    assert status.state(0, 1) == p2p.STATE_UNVERIFIED
    assert status.asymmetric_pairs() == [(0, 1)]


def test_can_transfer_directly_falls_back_to_capability(monkeypatch):
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(p2p, "_can_access_peer", lambda src, dst: (src, dst) == (1, 0))

    assert p2p.can_transfer_directly("cuda:0", "cuda:1") is False
    assert p2p.can_transfer_directly("cuda:1", "cuda:0") is True
    # Transfers that do not cross two accelerators are always safe.
    assert p2p.can_transfer_directly("cpu", "cuda:1") is True


def test_safe_module_to_stages_whole_module_on_unsupported_direction(monkeypatch):
    class FakeTensor:
        def __init__(self, device):
            self.device = device

    module = torch.nn.Linear(2, 2)
    monkeypatch.setattr(module, "parameters", lambda: iter([FakeTensor(torch.device("cuda:0"))]))
    monkeypatch.setattr(module, "buffers", lambda: iter([]))
    calls = []
    monkeypatch.setattr(module, "to", lambda *a, **k: calls.append((a, k)) or module)
    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *a, **k: False)

    p2p.safe_module_to(module, torch.device("cuda:1"))
    assert [call[1]["device"] for call in calls] == [
        torch.device("cpu"),
        torch.device("cuda:1"),
    ]


def test_safe_module_to_uses_direct_path_when_supported(monkeypatch):
    class FakeTensor:
        def __init__(self, device):
            self.device = device

    module = torch.nn.Linear(2, 2)
    monkeypatch.setattr(module, "parameters", lambda: iter([FakeTensor(torch.device("cuda:0"))]))
    monkeypatch.setattr(module, "buffers", lambda: iter([]))
    calls = []
    monkeypatch.setattr(module, "to", lambda *a, **k: calls.append((a, k)) or module)
    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *a, **k: True)

    p2p.safe_module_to(module, torch.device("cuda:1"))
    assert len(calls) == 1
    assert calls[0][1]["device"] == torch.device("cuda:1")


def test_copy_tensor_to_stages_unsupported_direction(monkeypatch):
    src = torch.arange(8)
    dst = torch.zeros(8)
    staged = src + 1
    monkeypatch.setattr(p2p, "can_transfer_directly", lambda *a, **k: False)
    monkeypatch.setattr(p2p, "safe_move_to", lambda *a, **k: staged)

    result = p2p.copy_tensor_to(dst, src)
    assert result is dst
    assert torch.equal(dst, staged)


def test_hooked_linear_routes_cross_device_input_through_safe_move(monkeypatch):
    """Regression: the forward replay crash was ``input.to(weight.device)``.

    ``HookedLinear`` must never call ``Tensor.to`` directly across devices; the
    asymmetric ROCm rig segfaulted there inside ``hipMemcpyPeerAsync``.
    """
    from gptqmodel.nn_modules import hooked_linear

    class FakeData:
        # The forward reads ``weight.data.device`` to pick the compute device,
        # so ``data`` must itself report the accelerator; a plain CPU tensor
        # would make the layer believe no cross-device move is needed.
        device = torch.device("cuda:1")

    class FakeWeight:
        dtype = torch.float32
        is_meta = False
        device = torch.device("cuda:1")
        data = FakeData()

    layer = hooked_linear.HookedLinear.from_linear(torch.nn.Linear(4, 4))
    # Replace the registered parameter without going through nn.Module's
    # Parameter type check; the forward only reads ``.data.device``.
    del layer._parameters["weight"]
    object.__setattr__(layer, "weight", FakeWeight())
    moved = []
    monkeypatch.setattr(
        hooked_linear,
        "safe_move_to",
        lambda tensor, device, **kwargs: moved.append(torch.device(device)) or tensor,
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda inp, weight, bias=None: inp,
    )

    output = layer(torch.zeros(2, 4))
    assert moved == [torch.device("cuda:1")]
    assert output.device == torch.device("cpu")


@pytest.mark.parametrize("broken_state", [p2p.STATE_UNSUPPORTED, p2p.STATE_UNVERIFIED])
def test_clone_skips_nccl_replicate_on_asymmetric_topology(monkeypatch, broken_state):
    from gptqmodel.utils import looper_helpers

    status = p2p.PeerStatus(
        devices=(0, 1),
        pairs={(0, 1): broken_state, (1, 0): p2p.STATE_ENABLED},
    )
    monkeypatch.setattr(looper_helpers, "enable_for_devices", lambda devices: status)
    p2p.configure_p2p(p2p.P2PConfig(use_torch_replicate=True))
    monkeypatch.setattr(looper_helpers, "_REPLICATE_DISABLED", False)
    # Exercise the replicate decision even when the test interpreter has the GIL
    # disabled (where clone_module_for_devices intentionally skips replicate).
    monkeypatch.setattr(looper_helpers, "has_gil_disabled", lambda: False)
    monkeypatch.setattr(
        looper_helpers,
        "torch_replicate",
        lambda *a, **k: pytest.fail("replicate must be skipped"),
    )
    monkeypatch.setattr(looper_helpers, "safe_module_to", lambda *a, **k: None)
    monkeypatch.setattr(looper_helpers, "rehome_module_to_device", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    clones = looper_helpers.clone_module_for_devices(
        torch.nn.Linear(2, 2),
        [torch.device("cuda", 0), torch.device("cuda", 1)],
    )
    assert set(clones) == {torch.device("cuda", 0), torch.device("cuda", 1)}


def test_clone_disables_replicate_after_nccl_failure(monkeypatch):
    from gptqmodel.utils import looper_helpers

    status = p2p.PeerStatus(
        devices=(0, 1),
        pairs={(0, 1): p2p.STATE_ENABLED, (1, 0): p2p.STATE_ENABLED},
    )
    monkeypatch.setattr(looper_helpers, "enable_for_devices", lambda devices: status)
    p2p.configure_p2p(p2p.P2PConfig(use_torch_replicate=True))
    monkeypatch.setattr(looper_helpers, "_REPLICATE_DISABLED", False)
    monkeypatch.setattr(looper_helpers, "has_gil_disabled", lambda: False)
    monkeypatch.setattr(
        looper_helpers,
        "torch_replicate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NCCL Error 2: unhandled system error")),
    )
    monkeypatch.setattr(looper_helpers, "safe_module_to", lambda *a, **k: None)
    monkeypatch.setattr(looper_helpers, "rehome_module_to_device", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    clones = looper_helpers.clone_module_for_devices(
        torch.nn.Linear(2, 2),
        [torch.device("cuda", 0), torch.device("cuda", 1)],
    )
    assert set(clones) == {torch.device("cuda", 0), torch.device("cuda", 1)}
    assert looper_helpers._REPLICATE_DISABLED is True


def test_p2p_unmeasurable_direct_link_is_degraded(monkeypatch):
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (None, 50.0))
    p2p.configure_p2p(p2p.P2PConfig(probe=True))
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.can_copy(0, 1) is False


def test_multi_gpu_direct_copy(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA/HIP devices required")
    if _REAL_ROCM_PEER_LINK_UNUSABLE():
        pytest.skip("cross-device ROCm copies are unreliable on this allocator")

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
    p2p.configure_p2p(p2p.P2PConfig(min_gbps=5))

    direct, staged = p2p._probe_bandwidth(0, 1)
    assert direct == 54.0
    assert staged == 55.0


def test_p2p_forced_mode_ignores_relative_gate(monkeypatch):
    # ``P2PConfig(mode='on')`` asks for the direct path explicitly; a link slower
    # than staging must not be vetoed by the relative comparison (the absolute
    # floor still applies).
    _fake_topology(monkeypatch, {(0, 1): True}, bandwidth=lambda src, dst: (20.0, 45.0))
    p2p.configure_p2p(p2p.P2PConfig(mode="on", min_gbps=5, probe=True))
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
        raise AssertionError("bandwidth probe must not run unless P2PConfig.probe is set")

    monkeypatch.setattr(p2p, "_probe_bandwidth", fail)
    status = p2p.enable_peer_access([0, 1])
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.state(1, 0) == p2p.STATE_ENABLED
    # ``auto`` still runs its small correctness/benefit validation; the full
    # 64 MiB diagnostic probe remains opt-in and must not run here.
    assert set(status.bandwidth_gbps) == {(0, 1), (1, 0)}


def test_indexless_string_and_device_mean_current_device(monkeypatch):
    p2p.clear_p2p_cache()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert p2p._normalize_indices(["cuda", "cuda:1"]) == (2, 1)
    assert p2p._normalize_indices([torch.device("cuda"), torch.device("cuda:3")]) == (2, 3)
    assert p2p._normalize_indices(["cpu", 1]) == (1,)


def test_probe_mib_invalid_value_is_rejected():
    with pytest.raises(ValueError):
        p2p.P2PConfig(probe_mib="not-a-number")
    p2p.configure_p2p(p2p.P2PConfig(probe_mib=8))
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


def test_p2p_auto_validates_the_runtime_peer_copy_without_probe(monkeypatch):
    """``auto`` must exercise the real copy path before trusting a pair."""

    validated = []
    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})

    def fake_validation(src, dst, **_kwargs):
        validated.append((src, dst))
        return True, 100.0, 50.0, ""

    monkeypatch.setattr(p2p, "_validate_peer_copy_pair", fake_validation)
    status = p2p.enable_peer_access([0, 1])

    assert sorted(validated) == [(0, 1), (1, 0)]
    assert status.state(0, 1) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is True


def test_p2p_auto_disables_a_pair_whose_peer_copy_is_broken(monkeypatch):
    """A capability bit or rc=0 is not enough: mismatched bytes disable the pair."""

    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    monkeypatch.setattr(
        p2p,
        "_validate_peer_copy_pair",
        lambda *a, **k: (False, None, None, "runtime peer copy returned mismatched bytes"),
    )

    status = p2p.enable_peer_access([0, 1])

    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.can_copy(0, 1) is False
    assert p2p.can_transfer_directly("cuda:0", "cuda:1", status) is False


def test_p2p_auto_disables_a_pair_that_is_not_faster_than_staging(monkeypatch):
    """``auto`` keeps only links that beat the fallback."""

    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    monkeypatch.setattr(
        p2p,
        "_validate_peer_copy_pair",
        lambda *a, **k: (True, 10.0, 50.0, ""),
    )

    status = p2p.enable_peer_access([0, 1])

    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.can_copy(0, 1) is False
    assert status.bandwidth_gbps[(0, 1)] == 10.0
    assert status.host_staged_gbps[(0, 1)] == 50.0


def test_p2p_validates_only_after_every_direction_is_enabled(monkeypatch):
    """A validation copy needs the reverse direction enabled to read the source."""

    calls = _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    seen = []

    def fake_validation(src, dst, **_kwargs):
        seen.append((src, dst, set(calls)))
        return True, 100.0, 50.0, ""

    monkeypatch.setattr(p2p, "_validate_peer_copy_pair", fake_validation)
    p2p.enable_peer_access([0, 1])

    assert len(seen) == 2
    for _src, _dst, enabled_at_validation in seen:
        assert (0, 1) in enabled_at_validation
        assert (1, 0) in enabled_at_validation


def test_p2p_validation_rejects_a_noop_copy(monkeypatch):
    """A successful return code is not enough: random content must match."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")

    monkeypatch.setattr(p2p, "_PEER_VALIDATION_CACHE", {})
    # Simulate a runtime copy that returns success without moving bytes. Patch
    # the helper the probe actually drives, which is ``Tensor.copy_`` whenever
    # the raw runtime call is withheld (ROCm with a pre-enable-free allocator).
    monkeypatch.setattr(p2p, "_copy_payloads_like_movers", lambda s, t, d: True)

    usable, direct, staged, reason = p2p._validate_peer_copy_pair(0, 1, probe_mib=1)

    assert usable is False
    assert direct is None and staged is None
    assert "mismatched bytes" in reason


def _activate_rocm_non_native_guard(monkeypatch):
    """Force the non-native allocator condition on any test host."""

    monkeypatch.setattr(p2p, "_raw_peer_copy_unsafe", lambda: True)
    monkeypatch.setattr(p2p, "_rocm_peer_link_unusable", lambda: True)
    p2p.clear_p2p_cache()


def _activate_pytorch_enablement(monkeypatch, *, ndev=2, capability=None):
    """Force the "PyTorch owns enablement" condition on any test host.

    These are policy-only tests: the allocator and raw-copy predicates are
    replaced, so the link verdict must not depend on how many GPUs the host
    actually has. Install a synthetic peer topology alongside the policy stubs.
    """

    monkeypatch.setattr(p2p, "_raw_peer_copy_unsafe", lambda: True)
    monkeypatch.setattr(p2p, "_rocm_peer_link_unusable", lambda: False)
    monkeypatch.setattr(p2p, "_pytorch_owns_peer_enablement", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: ndev)
    peers = capability if capability is not None else {(0, 1): True, (1, 0): True}
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda src, dst: bool(peers.get((src, dst), False)),
    )
    p2p.clear_p2p_cache()


@pytest.mark.parametrize("backend", ["native", "cudaMallocAsync", None])
@pytest.mark.parametrize("hip", [None, "test"])
def test_allocator_guards_use_active_backend(monkeypatch, backend, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(p2p, "_runtime_allocator_backend", lambda: backend)
    owns_enablement = backend in ("native", None)
    unusable = hip is not None and backend != "native"

    assert p2p._allocator_is_native() is (backend == "native")
    assert _REAL_RAW_PEER_COPY_UNSAFE() is (owns_enablement or unusable)
    assert _REAL_ROCM_PEER_LINK_UNUSABLE() is unusable
    assert _REAL_PYTORCH_OWNS_ENABLEMENT() is owns_enablement


@pytest.mark.parametrize("settings", ["", "expandable_segments:False", "max_split_size_mb:256", None])
@pytest.mark.parametrize("hip", [None, "test"])
def test_native_allocator_settings_never_authorize_raw_copy(monkeypatch, settings, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: settings, raising=False)
    monkeypatch.setattr(p2p, "_runtime_allocator_backend", lambda: "native")

    assert _REAL_RAW_PEER_COPY_UNSAFE() is True
    assert _REAL_PYTORCH_OWNS_ENABLEMENT() is True
    assert _REAL_ROCM_PEER_LINK_UNUSABLE() is False


def test_rocm_native_expandable_keeps_p2p_direct(monkeypatch):
    """A native expandable allocator keeps the validated direct link."""

    _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    _activate_pytorch_enablement(monkeypatch)

    def fake_validation(src, dst, **_kwargs):
        return True, 30.0, 20.0, ""

    monkeypatch.setattr(p2p, "_validate_peer_copy_pair", fake_validation)
    monkeypatch.setattr(
        p2p,
        "_measure_host_staged_gbps",
        lambda *a, **k: 20.0,
    )

    status = p2p.enable_peer_access([0, 1])

    # The link stays direct: nothing is degraded and no host relay is inserted.
    for src, dst in ((0, 1), (1, 0)):
        assert status.state(src, dst) == p2p.STATE_ENABLED
    assert status.can_copy(0, 1) is True
    assert status.reason_code == ""
    assert p2p.can_transfer_directly("cuda:0", "cuda:1", status) is True


@pytest.mark.parametrize("hip", [None, "test"])
def test_pytorch_enablement_skips_the_ctypes_call(monkeypatch, hip):
    """Native expandable allocators own enablement on both GPU families."""

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(p2p, "_pytorch_owns_peer_enablement", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(
        p2p,
        "_load_hip",
        lambda: pytest.fail("delegated enablement must not load the HIP runtime"),
    )
    monkeypatch.setattr(p2p, "_load_cuda", lambda: pytest.fail("must not load the CUDA driver"))

    ok, detail = p2p._enable_one(0, 1)

    assert ok is True
    assert detail == p2p.REASON_PYTORCH_ENABLEMENT


def test_pytorch_enablement_keeps_direct_moves(monkeypatch):
    """The withheld raw copy must not demote the link itself."""

    _activate_pytorch_enablement(monkeypatch)

    assert p2p.can_transfer_directly("cuda:0", "cuda:1") is True
    assert p2p.can_transfer_directly("cuda:1", "cuda:0") is True


def test_pytorch_enablement_verdict_follows_the_synthetic_topology(monkeypatch):
    """The verdict must come from the stubbed topology, not the host GPU count."""

    _activate_pytorch_enablement(monkeypatch, capability={})

    assert p2p.can_transfer_directly("cuda:0", "cuda:1") is False
    # A single-device host cannot have this pair, so a same-device query stays safe.
    assert p2p.can_transfer_directly("cuda:0", "cuda:0") is True


def test_pytorch_enablement_withholds_only_the_raw_copy(monkeypatch):
    """The raw ctypes copy is withheld; the link itself is not demoted."""

    _activate_pytorch_enablement(monkeypatch)

    class FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

        def is_contiguous(self):
            return True

        def numel(self):
            return 1

        def element_size(self):
            return 1

    assert p2p._raw_peer_copy_unsafe() is True
    assert p2p._peer_copy_batch_enqueue(
        [FakeTensor("cuda:0")], [FakeTensor("cuda:1")]
    ) is False
    # ...but a direct transfer is still allowed, because ``copy_`` is correct.
    assert p2p.can_transfer_directly("cuda:0", "cuda:1") is True


def test_pytorch_enablement_probe_uses_the_mover_path(monkeypatch):
    """The probe must validate ``Tensor.copy_``, not the withheld raw call."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA/HIP devices required")
    if _REAL_ROCM_PEER_LINK_UNUSABLE() or not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("a usable peer direction is required")
    _activate_pytorch_enablement(monkeypatch)
    monkeypatch.setattr(
        p2p,
        "_peer_copy_batch_enqueue",
        lambda sources, targets: pytest.fail("the raw copy must not be validated"),
    )
    used = []
    real_copy = p2p._copy_payloads_like_movers

    def recording_copy(sources, targets, dst):
        used.append((sources, targets, dst))
        return real_copy(sources, targets, dst)

    monkeypatch.setattr(p2p, "_copy_payloads_like_movers", recording_copy)

    usable, _direct, _staged, reason = p2p._validate_peer_copy_pair(0, 1, probe_mib=1)

    assert usable is True, reason
    # The probe drives the mover helper for both the correctness copy and the
    # timing loop, and never reaches the withheld raw enqueue.
    assert used


def test_rocm_non_native_allocator_fails_closed_without_probing(monkeypatch):
    """A non-native backend breaks even ``copy_``, so nothing may be probed."""

    calls = _fake_topology(monkeypatch, {(0, 1): True, (1, 0): True})
    _activate_rocm_non_native_guard(monkeypatch)
    monkeypatch.setattr(
        p2p,
        "_validate_peer_copy_pair",
        lambda *a, **k: pytest.fail("a non-native allocator must not be probed"),
    )

    status = p2p.enable_peer_access([0, 1])

    assert calls == []
    assert status.state(0, 1) == p2p.STATE_DEGRADED
    assert status.state(1, 0) == p2p.STATE_DEGRADED
    assert status.can_copy(0, 1) is False
    assert status.reason_code == p2p.REASON_ROCM_VMM
    assert "not native" in status.reason


def test_rocm_non_native_guard_overrides_enabled_status(monkeypatch):
    """A cached ``enabled`` verdict must not bypass the allocator guard."""

    _activate_rocm_non_native_guard(monkeypatch)
    status = p2p.PeerStatus(
        devices=(0, 1),
        pairs={(0, 1): p2p.STATE_ENABLED, (1, 0): p2p.STATE_ENABLED},
    )

    assert p2p.can_transfer_directly("cuda:0", "cuda:1", status) is False
    assert p2p.can_transfer_directly("cuda:1", "cuda:0", status) is False
    # Same-device and host transfers do not use a peer link.
    assert p2p.can_transfer_directly("cuda:0", "cuda:0", status) is True
    assert p2p.can_transfer_directly("cpu", "cuda:1", status) is True


def test_rocm_non_native_guard_reports_its_own_warning(monkeypatch):
    """The allocator guard must not blame the motherboard/BIOS topology wording."""

    _activate_rocm_non_native_guard(monkeypatch)
    warnings = []

    class RecordingLog:
        def warning(self, message, *args, **kwargs):
            warnings.append(message % args if args else message)

        def info(self, message, *args, **kwargs):
            pass

        def debug(self, message, *args, **kwargs):
            pass

    monkeypatch.setattr(p2p, "log", RecordingLog())

    assert p2p.can_transfer_directly("cuda:0", "cuda:1") is False
    assert p2p.can_transfer_directly("cuda:1", "cuda:0") is False
    p2p.enable_peer_access([0, 1])

    assert len(warnings) == 1
    assert "not native" in warnings[0]
    # The topology wording is reserved for a real peer-link problem.
    assert "motherboard" not in warnings[0]


def test_peer_copy_batch_rejects_mixed_group_before_probing(monkeypatch):
    """A mixed group is rejected without a real probe copy."""

    class FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

    monkeypatch.setattr(
        p2p,
        "_can_peer_copy",
        lambda *a, **k: pytest.fail("a mixed group must not be probed"),
    )

    assert p2p.peer_copy_batch(
        [FakeTensor("cuda:0"), FakeTensor("cuda:1")],
        [FakeTensor("cuda:1"), FakeTensor("cuda:0")],
    ) is False


@pytest.mark.parametrize("failure_at", [1, 9, 10])
def test_bandwidth_measurement_rejects_failed_copy(monkeypatch, failure_at):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    calls = []

    def step():
        calls.append(None)
        return len(calls) != failure_at

    assert p2p._timed_gbps(step, 0, 1, 1024, iterations=3) is None
    assert len(calls) == failure_at


@pytest.mark.parametrize("hip", [None, "test"])
def test_unknown_backend_delegates_without_loading_runtime(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(p2p, "_runtime_allocator_backend", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(p2p, "_load_hip", lambda: pytest.fail("unknown backend must delegate"))
    monkeypatch.setattr(p2p, "_load_cuda", lambda: pytest.fail("unknown backend must delegate"))

    assert p2p._enable_one(0, 1) == (True, p2p.REASON_PYTORCH_ENABLEMENT)


@pytest.mark.parametrize("hip", [None, "test"])
@pytest.mark.parametrize("allow_nested", [False, True])
def test_rocm_nested_copy_policy_uses_config(monkeypatch, hip, allow_nested):
    monkeypatch.setattr(torch.version, "hip", hip)
    config = p2p.configure_p2p({"use_rocm_nested_peer_copy": allow_nested})

    assert config.to_dict()["use_rocm_nested_peer_copy"] is allow_nested
    assert p2p._rocm_nested_peer_disabled() is (hip is not None and not allow_nested)


def test_rocm_nested_opt_in_preserves_allocator_guard(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    p2p.configure_p2p(p2p.P2PConfig(use_rocm_nested_peer_copy=True))
    monkeypatch.setattr(p2p, "_runtime_allocator_backend", lambda: "native")

    class FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

    monkeypatch.setattr(p2p, "_peer_copy_runtime", lambda: pytest.fail("raw copy is unsafe"))
    assert p2p._rocm_nested_peer_disabled() is False
    assert p2p._peer_copy_batch_enqueue([FakeTensor("cuda:0")], [FakeTensor("cuda:1")]) is False


@pytest.mark.parametrize("setting", [
    "max_split_size_mb:256",
    "",
    "expandable_segments:False",
    "expandable_segments:False,invalid_option:1",
])
def test_native_allocator_updates_keep_pytorch_ownership(setting):
    """Settings updates cannot authorize raw copies of existing VMM tensors."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA/HIP devices required")
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("requires the native allocator")
    if not all(torch.cuda.can_device_access_peer(src, 1 - src) for src in (0, 1)):
        pytest.skip("two usable peer directions required")

    # Allocator settings are process-global. Keep the transition and retained
    # segments in a child so other tests never inherit a modified allocator.
    script = textwrap.dedent("""
        import sys
        import torch
        from gptqmodel.utils import p2p

        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        expected = torch.arange(1024 * 1024, dtype=torch.int32)
        sources = [expected.to(f"cuda:{index}") for index in (0, 1)]
        segments = torch.cuda.memory_snapshot()
        assert all(any(s["device"] == index and s["is_expandable"]
                       for s in segments) for index in (0, 1))
        setting = sys.argv[1]
        try:
            torch._C._accelerator_setAllocatorSettings(setting)
        except (ValueError, RuntimeError):
            assert "invalid_option" in setting
        else:
            assert "invalid_option" not in setting

        assert p2p._pytorch_owns_peer_enablement(), setting
        assert p2p._raw_peer_copy_unsafe(), setting
        p2p.configure_p2p(p2p.P2PConfig(mode="on"))
        status = p2p.enable_peer_access([0, 1])
        for src in (0, 1):
            dst = 1 - src
            assert status.can_copy(src, dst), status.summary()
            target = torch.empty_like(sources[src], device=f"cuda:{dst}")
            assert not p2p.peer_copy_batch([sources[src]], [target])
            p2p.copy_tensor_to(target, sources[src], status)
            assert torch.equal(target.cpu(), expected)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, setting],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
