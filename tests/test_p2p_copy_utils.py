# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GIL-safe peer-copy helpers built on top of ``gptqmodel.utils.p2p``.

The fast path deliberately avoids ``torch.cuda.comm.*``: on free-threaded
``PYTHON_GIL=0`` builds ``broadcast_coalesced`` has hung and returned torn
copies on both ROCm and CUDA. These tests cover the replacement, the module
and nested-input moves that use it, and the one-time degraded warning.
"""

import pytest
import torch

from gptqmodel.utils import p2p as p2p_mod
from gptqmodel.utils.looper_helpers import move_module_to_device, move_nested_to_device
from gptqmodel.utils.p2p import (
    P2P_NOT_APPLICABLE,
    broadcast_module_to_device,
    clear_p2p_cache,
    enable_peer_access,
    p2p_all_enabled,
    p2p_move_nested,
    peer_copy_batch,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")


class _RecordingLog:
    def __init__(self):
        self.warnings = []
        self.infos = []
        self.debugs = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append((message, args))

    def warn(self, message, *args, **kwargs):
        self.warnings.append((message, args))

    def info(self, message, *args, **kwargs):
        self.infos.append((message, args))

    def debug(self, message, *args, **kwargs):
        self.debugs.append((message, args))


@pytest.fixture(autouse=True)
def reset_p2p_state(monkeypatch):
    clear_p2p_cache()
    monkeypatch.setattr(p2p_mod, "_WARNED_DEGRADED", False)
    recorder = _RecordingLog()
    monkeypatch.setattr(p2p_mod, "log", recorder)
    yield recorder
    clear_p2p_cache()


def _require_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")


def _require_p2p(src=0, dst=1):
    _require_two_gpus()
    status = enable_peer_access([src, dst])
    if not status.can_copy(src, dst):
        pytest.skip("P2P is not enabled on this host for %s->%s" % (src, dst))
    return status


def _require_p2p_both_ways(src=0, dst=1):
    status = _require_p2p(src, dst)
    if not status.can_copy(dst, src):
        pytest.skip("P2P is not enabled on this host for %s->%s" % (dst, src))
    return status


def _require_nested_p2p():
    if p2p_mod._ROCM_NESTED_PEER_DISABLED:
        pytest.skip("nested VMM peer copies are disabled on ROCm")


def test_p2p_all_enabled_matches_status():
    _require_p2p_both_ways()
    assert p2p_all_enabled([0, 1]) is True


def test_degraded_p2p_warns_once_with_actionable_reason(monkeypatch, reset_p2p_state):
    _require_two_gpus()
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda src, dst: False)

    first = enable_peer_access([0, 1])
    second = enable_peer_access([0, 1])

    assert first.can_copy(0, 1) is False
    assert second.can_copy(0, 1) is False
    assert len(reset_p2p_state.warnings) == 1

    message, args = reset_p2p_state.warnings[0]
    rendered = message % args
    assert "motherboard compatibility" in rendered
    assert "BIOS compatibility" in rendered
    assert "hardware compatibility" in rendered
    assert "speed may be sub-optimal" in rendered


def test_concurrent_degraded_probe_warns_once(monkeypatch, reset_p2p_state):
    _require_two_gpus()
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda src, dst: False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: enable_peer_access([0, 1]), range(32)))

    assert all(status.can_copy(0, 1) is False for status in statuses)
    assert len(reset_p2p_state.warnings) == 1


def test_peer_copy_batch_roundtrip():
    _require_p2p(0, 1)

    sources = [torch.randn(1024, 1024, device="cuda:0") for _ in range(3)]
    targets = [torch.empty_like(source, device="cuda:1") for source in sources]

    assert peer_copy_batch(sources, targets) is True
    torch.cuda.current_stream(1).synchronize()
    for source, target in zip(sources, targets):
        torch.testing.assert_close(source.cpu(), target.cpu(), rtol=0, atol=0)


def test_peer_copy_batch_rejects_heterogeneous_groups():
    _require_two_gpus()
    first = torch.randn(8, device="cuda:0")
    second = torch.randn(8, device="cuda:1")
    targets = [
        torch.empty_like(first, device="cuda:1"),
        torch.empty_like(first, device="cuda:1"),
    ]
    assert peer_copy_batch([first, second], targets) is False
    assert peer_copy_batch([], []) is False


def test_peer_copy_batch_rejects_non_contiguous():
    _require_p2p(0, 1)

    source = torch.randn(64, 64, device="cuda:0").t()
    target = torch.empty(64, 64, device="cuda:1")
    assert not source.is_contiguous()
    assert peer_copy_batch([source], [target]) is False


def test_move_module_to_device_roundtrip():
    _require_p2p(0, 1)

    module = torch.nn.Linear(16, 8).to("cuda:0")
    expected_weight = module.weight.detach().clone()
    expected_bias = module.bias.detach().clone()

    move_module_to_device(module, torch.device("cuda:1"))

    assert module.weight.device == torch.device("cuda:1")
    assert module.bias.device == torch.device("cuda:1")
    assert module.weight.requires_grad is True
    assert module.bias.requires_grad is True
    torch.testing.assert_close(module.weight, expected_weight.to("cuda:1"))
    torch.testing.assert_close(module.bias, expected_bias.to("cuda:1"))


def test_broadcast_module_to_device_rejects_non_contiguous():
    _require_p2p(0, 1)

    module = torch.nn.Linear(16, 8).to("cuda:0")
    module.weight = torch.nn.Parameter(module.weight.data.t())
    assert not module.weight.is_contiguous()

    # The peer-copy path moves raw storage, so it must refuse instead of
    # silently transposing the weights.
    assert broadcast_module_to_device(module, torch.device("cuda:1")) is False
    assert module.weight.device == torch.device("cuda:0")


def test_module_move_is_visible_from_another_stream():
    _require_p2p(0, 1)

    module = torch.nn.Linear(256, 128).to("cuda:0")
    expected = module.weight.detach().clone()
    assert broadcast_module_to_device(module, torch.device("cuda:1")) is True

    other = torch.cuda.Stream(device=1)
    with torch.cuda.stream(other):
        observed = float(module.weight.sum().item())
    assert observed == pytest.approx(float(expected.sum().item()), rel=1e-5)


def test_p2p_move_nested_is_ordered_on_destination_stream():
    """A read on the destination device must see the copy without a sync.

    HIP binds a stream to the *current* device's context. Issuing the peer copy
    while the source device was current placed the work on the source queue, so
    the destination stream -- and therefore ``Tensor.cpu()`` -- never waited for
    it. The first, smallest leaf was then reliably read as uninitialised
    memory. ``torch.cuda.synchronize(target)`` did not fix it; synchronising the
    source did, which is what pinned the copy to the wrong queue.
    """
    _require_nested_p2p()
    _require_p2p(0, 1)

    for _ in range(5):
        values = {
            "ids": torch.arange(8 * 512, device="cuda:0").reshape(8, 512) % 32000,
            "hidden": [torch.full((256, 256), float(i), device="cuda:0") for i in range(4)],
        }
        moved = p2p_move_nested(values, torch.device("cuda:1"))

        assert moved is not P2P_NOT_APPLICABLE
        # Deliberately no synchronize(): the destination stream is already ordered.
        assert torch.equal(moved["ids"].cpu(), values["ids"].cpu())
        for observed, expected in zip(moved["hidden"], values["hidden"]):
            assert torch.equal(observed.cpu(), expected.cpu())


def _small_leaves(count: int = 5):
    """Leaves in the peer-copy win region: 256 KiB each, comfortably over 1 MiB."""
    return [torch.randn(256, 256, device="cuda:0") for _ in range(count)]


def test_p2p_move_nested_coalesces_multiple_tensors():
    _require_nested_p2p()
    _require_p2p(0, 1)

    values = _small_leaves()
    moved = p2p_move_nested(values, torch.device("cuda:1"))

    assert moved is not P2P_NOT_APPLICABLE
    for observed, expected in zip(moved, values):
        assert observed.device == torch.device("cuda:1")
        torch.testing.assert_close(observed.cpu(), expected.cpu())


def test_p2p_move_nested_skips_non_contiguous():
    _require_p2p(0, 1)

    base = torch.randn(256, 256, device="cuda:0")
    values = _small_leaves(4) + [base.t()]
    assert not values[-1].is_contiguous()
    assert p2p_move_nested(values, torch.device("cuda:1")) is P2P_NOT_APPLICABLE


def test_p2p_move_nested_falls_back_when_rocm_nested_is_disabled(monkeypatch):
    """ROCm disables the small VMM batch shape that can fault the GPU."""

    monkeypatch.setattr(p2p_mod, "_ROCM_NESTED_PEER_DISABLED", True)
    values = _small_leaves()
    assert p2p_move_nested(values, torch.device("cuda:1")) is P2P_NOT_APPLICABLE


def test_p2p_move_nested_skips_autograd():
    _require_p2p(0, 1)

    values = [
        torch.randn(256, 256, device="cuda:0", requires_grad=True) for _ in range(5)
    ]
    assert p2p_move_nested(values, torch.device("cuda:1")) is P2P_NOT_APPLICABLE


def test_p2p_move_nested_skips_large_leaves():
    """Large leaves stay on ``.to()``: the peer path is measurably slower there."""
    _require_p2p(0, 1)

    values = [torch.randn(1024, 512, device="cuda:0") for _ in range(5)]
    assert p2p_move_nested(values, torch.device("cuda:1")) is P2P_NOT_APPLICABLE


def test_move_nested_to_device_cpu_fallback():
    _require_two_gpus()
    values = {"hidden": torch.randn(8, device="cuda:0")}
    moved = move_nested_to_device(values, torch.device("cpu"))
    assert moved["hidden"].device == torch.device("cpu")
    assert p2p_move_nested(values, torch.device("cpu")) is P2P_NOT_APPLICABLE


def test_p2p_moves_are_thread_safe():
    _require_p2p_both_ways()
    _require_nested_p2p()

    from concurrent.futures import ThreadPoolExecutor

    def _work(_index):
        module = torch.nn.Linear(64, 32).to("cuda:0")
        expected_weight = module.weight.detach().clone()
        expected_bias = module.bias.detach().clone()
        move_module_to_device(module, torch.device("cuda:1"))
        assert module.weight.device == torch.device("cuda:1")
        torch.testing.assert_close(module.weight, expected_weight.to("cuda:1"))
        torch.testing.assert_close(module.bias, expected_bias.to("cuda:1"))

        values = _small_leaves()
        moved = p2p_move_nested(values, torch.device("cuda:1"))
        assert moved is not P2P_NOT_APPLICABLE
        for source, result in zip(values, moved):
            torch.testing.assert_close(source.cpu(), result.cpu())

        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        assert 0 <= free_bytes <= total_bytes
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(_work, range(4)))


def test_p2p_fast_path_does_not_use_torch_cuda_comm(monkeypatch):
    """Guard the GIL=0 fix: torch.cuda.comm.* hangs or tears under threads."""

    _require_p2p_both_ways()
    _require_nested_p2p()

    import torch.cuda.comm as cuda_comm

    def _forbidden(*args, **kwargs):
        raise AssertionError("the P2P fast path must not call torch.cuda.comm")

    monkeypatch.setattr(cuda_comm, "broadcast_coalesced", _forbidden)
    monkeypatch.setattr(cuda_comm, "broadcast", _forbidden)
    monkeypatch.setattr(cuda_comm, "reduce_add", _forbidden)

    module = torch.nn.Linear(32, 16).to("cuda:0")
    assert broadcast_module_to_device(module, torch.device("cuda:1")) is True
    assert module.weight.device == torch.device("cuda:1")

    values = _small_leaves()
    moved = p2p_move_nested(values, torch.device("cuda:1"))
    assert moved is not P2P_NOT_APPLICABLE
    for source, result in zip(values, moved):
        torch.testing.assert_close(source.cpu(), result.cpu(), rtol=0, atol=0)
