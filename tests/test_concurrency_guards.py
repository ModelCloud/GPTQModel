# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Process-global quantization guards must be reference-counted.

Concurrent workers share one process-global BLAS backend, one TF32 setting and
one CPU BF16x3 matmul precision. The old save/restore-per-entry guards let the
first caller to finish restore the user's setting while another caller was
still inside, so its remaining work ran on the wrong backend/precision. Issue
#4094 measured 177-210 of 794 tensors differing from a serial run; #4116
measured the BF16x3 guard flipping a concurrent forward from BF16x3 to IEEE
fp32. These tests pin the counting contract on the guards themselves, without
a GPU.
"""

import threading

import pytest
import torch

import gptqmodel.utils.rocm as rocm_utils
import gptqmodel.utils.torch as torch_utils
from gptqmodel.utils import cpu_features


class _Backend:
    def __init__(self, name):
        self.name = name


def _run_two_threads(enter):
    """Enter a guard in two threads and return a release/join helper."""
    entered_a = threading.Event()
    entered_b = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()
    errors = []

    def worker(entered, release):
        try:
            with enter():
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("guard was never released")
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    thread_a = threading.Thread(target=worker, args=(entered_a, release_a))
    thread_b = threading.Thread(target=worker, args=(entered_b, release_b))
    thread_a.start()
    assert entered_a.wait(timeout=10)
    thread_b.start()
    assert entered_b.wait(timeout=10)
    return release_a, release_b, thread_a, thread_b, errors


def test_quant_blas_backend_is_reference_counted(monkeypatch):
    state = {"current": _Backend("cuda.Cublaslt")}
    setter_calls = []

    def setter(value):
        state["current"] = value if isinstance(value, _Backend) else _Backend(value)
        setter_calls.append(state["current"].name)

    monkeypatch.setattr(rocm_utils, "IS_ROCM", True)
    monkeypatch.setattr(rocm_utils, "quantize_tuning", lambda device=None: {"blas_backend": "hipblas"})
    monkeypatch.setattr(rocm_utils, "_preferred_blas_library_getter", lambda: setter)
    monkeypatch.setattr(rocm_utils, "preferred_blas_library", lambda: state["current"])
    monkeypatch.setattr(rocm_utils, "_QUANT_BLAS_GUARD_DEPTH", 0)
    monkeypatch.setattr(rocm_utils, "_QUANT_BLAS_GUARD_PREVIOUS", None)

    release_a, release_b, thread_a, thread_b, errors = _run_two_threads(rocm_utils.quant_blas_backend)
    try:
        assert state["current"].name == "hipblas"
        release_a.set()
        thread_a.join(timeout=10)
        # The first run to finish must not restore hipBLASLt: the second run is
        # still inside the guard.
        assert state["current"].name == "hipblas"
        assert setter_calls == ["hipblas"]
    finally:
        release_b.set()
        thread_b.join(timeout=10)

    assert not errors
    assert state["current"].name == "cuda.Cublaslt"
    assert setter_calls == ["hipblas", "cuda.Cublaslt"]
    assert rocm_utils._QUANT_BLAS_GUARD_DEPTH == 0


def test_quant_blas_backend_nested_entries_restore_once(monkeypatch):
    state = {"current": _Backend("cuda.Cublaslt")}
    setter_calls = []

    def setter(value):
        state["current"] = value if isinstance(value, _Backend) else _Backend(value)
        setter_calls.append(state["current"].name)

    monkeypatch.setattr(rocm_utils, "IS_ROCM", True)
    monkeypatch.setattr(rocm_utils, "quantize_tuning", lambda device=None: {"blas_backend": "hipblas"})
    monkeypatch.setattr(rocm_utils, "_preferred_blas_library_getter", lambda: setter)
    monkeypatch.setattr(rocm_utils, "preferred_blas_library", lambda: state["current"])
    monkeypatch.setattr(rocm_utils, "_QUANT_BLAS_GUARD_DEPTH", 0)
    monkeypatch.setattr(rocm_utils, "_QUANT_BLAS_GUARD_PREVIOUS", None)

    with rocm_utils.quant_blas_backend():
        assert state["current"].name == "hipblas"
        with rocm_utils.quant_blas_backend():
            assert state["current"].name == "hipblas"
        assert state["current"].name == "hipblas"

    assert state["current"].name == "cuda.Cublaslt"
    assert setter_calls == ["hipblas", "cuda.Cublaslt"]


def test_tf32_high_precision_guard_is_reference_counted(monkeypatch):
    state = {"enabled": True, "restores": []}

    def snapshot():
        return state["enabled"]

    def set_enabled(value):
        state["enabled"] = value

    def restore(value):
        state["restores"].append(value)
        state["enabled"] = value

    monkeypatch.setattr(torch_utils, "HAS_CUDA", True)
    monkeypatch.setattr(torch_utils, "_snapshot_tf32_state", snapshot)
    monkeypatch.setattr(torch_utils, "_set_tf32_state", set_enabled)
    monkeypatch.setattr(torch_utils, "_restore_tf32_state", restore)
    monkeypatch.setattr(torch_utils, "_TF32_GUARD_DEPTH", 0)
    monkeypatch.setattr(torch_utils, "_TF32_GUARD_PREVIOUS", None)

    release_a, release_b, thread_a, thread_b, errors = _run_two_threads(
        torch_utils.tf32_high_precision_guard
    )
    try:
        assert state["enabled"] is False
        release_a.set()
        thread_a.join(timeout=10)
        # The first run to finish must not re-enable TF32 for the second run.
        assert state["enabled"] is False
        assert state["restores"] == []
    finally:
        release_b.set()
        thread_b.join(timeout=10)

    assert not errors
    assert state["enabled"] is True
    assert state["restores"] == [True]
    assert torch_utils._TF32_GUARD_DEPTH == 0



def _install_cpu_bf16x3_state(monkeypatch, precision="none"):
    """Point the BF16x3 scope at a fake process-wide precision setting."""

    state = {"precision": precision}
    writes = []

    def getter():
        return state["precision"]

    def setter(value):
        state["precision"] = value
        writes.append(value)

    monkeypatch.setattr(cpu_features, "_cpu_bf16x3_precision", getter)
    monkeypatch.setattr(cpu_features, "_set_cpu_bf16x3_precision", setter)
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_DEPTH", 0)
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_RESTORE", None)
    return state, writes


def test_cpu_bf16x3_scope_is_reference_counted(monkeypatch):
    state, writes = _install_cpu_bf16x3_state(monkeypatch)

    release_a, release_b, thread_a, thread_b, errors = _run_two_threads(
        cpu_features.cpu_bf16x3_scope
    )
    try:
        assert state["precision"] == "bf16"
        release_a.set()
        thread_a.join(timeout=10)
        # The first forward to finish must not restore IEEE while the second
        # forward is still inside the scope: its remaining matmuls would run in
        # plain FP32 instead of BF16x3.
        assert state["precision"] == "bf16"
        assert writes == ["bf16"]
    finally:
        release_b.set()
        thread_b.join(timeout=10)

    assert not errors
    assert state["precision"] == "none"
    assert writes == ["bf16", "none"]
    assert cpu_features._CPU_BF16X3_DEPTH == 0


def test_cpu_bf16x3_scope_is_reentrant_and_restores_once(monkeypatch):
    state, writes = _install_cpu_bf16x3_state(monkeypatch)

    # A plain Lock would deadlock the nested entry below, so the scope must
    # stay re-entrant.
    assert isinstance(cpu_features._CPU_BF16X3_LOCK, type(threading.RLock()))

    with cpu_features.cpu_bf16x3_scope():
        assert state["precision"] == "bf16"
        with cpu_features.cpu_bf16x3_scope():
            assert state["precision"] == "bf16"
        assert state["precision"] == "bf16"

    assert state["precision"] == "none"
    assert writes == ["bf16", "none"]
    assert cpu_features._CPU_BF16X3_DEPTH == 0


def test_cpu_bf16x3_scope_keeps_a_caller_selected_precision(monkeypatch):
    state, writes = _install_cpu_bf16x3_state(monkeypatch, precision="bf16")

    with cpu_features.cpu_bf16x3_scope():
        assert state["precision"] == "bf16"

    assert state["precision"] == "bf16"
    # The caller already asked for BF16x3, so there is nothing to restore.
    assert writes == []
    assert cpu_features._CPU_BF16X3_DEPTH == 0


def test_cpu_bf16x3_guard_has_one_implementation():
    exllamav3_torch = pytest.importorskip("gptqmodel.nn_modules.exllamav3_torch")

    # GGUF CPU, bitsandbytes CPU and the EXL3 torch kernel share one scope; the
    # per-entry duplicate in the EXL3 module is gone.
    assert exllamav3_torch.cpu_bf16x3_scope is cpu_features.cpu_bf16x3_scope
    assert not hasattr(exllamav3_torch, "_cpu_bf16x3_matmul")

def test_cpu_bf16x3_matmul_entry_point_keeps_the_second_caller(monkeypatch):
    """The GGUF CPU / bitsandbytes CPU entry point is the one that regressed."""

    try:
        previous = torch.backends.mkldnn.matmul.fp32_precision
    except AttributeError:
        pytest.skip("torch does not expose oneDNN matmul precision")

    monkeypatch.setattr(cpu_features, "has_cpu_bf16x3_matmul", lambda: True)
    # The shared scope must start idle. The attributes do not exist on the
    # pre-fix tree, where the guard keeps a per-entry local save/restore.
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_DEPTH", 0, raising=False)
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_RESTORE", None, raising=False)

    torch.backends.mkldnn.matmul.fp32_precision = "none"
    try:
        release_a, release_b, thread_a, thread_b, errors = _run_two_threads(
            cpu_features.cpu_bf16x3_matmul
        )
        try:
            assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
            release_a.set()
            thread_a.join(timeout=10)
            # A exited while B is still inside; B must still see BF16x3.
            assert torch.backends.mkldnn.matmul.fp32_precision == "bf16"
        finally:
            release_b.set()
            thread_b.join(timeout=10)

        assert not errors
        assert torch.backends.mkldnn.matmul.fp32_precision == "none"
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = previous


def test_cpu_bf16x3_capability_probe_respects_an_active_scope(monkeypatch):
    """A capability probe must not restore a stale snapshot over a live scope.

    ``has_cpu_bf16x3_precision`` mutates the process-global precision to probe
    it. Pre-fix it did so without the owner lock, so a concurrent probe could
    snapshot IEEE, race a scope entry, and then restore IEEE while the scope
    was still active (issue #4486).
    """

    state, _ = _install_cpu_bf16x3_state(monkeypatch, precision="ieee")
    monkeypatch.setattr(torch.backends.mkldnn, "is_available", lambda: True)

    real_getter = cpu_features._cpu_bf16x3_precision
    read_done = threading.Event()
    release_probe = threading.Event()
    forward_started = threading.Event()
    scope_entered = threading.Event()
    probe_done = threading.Event()
    results = {}

    def gated_getter():
        value = real_getter()
        if threading.current_thread().name == "capability-probe":
            read_done.set()
            assert release_probe.wait(timeout=10)
        return value

    monkeypatch.setattr(cpu_features, "_cpu_bf16x3_precision", gated_getter)
    # The capability is a process constant published as a module global;
    # monkeypatch resets it for this probe and restores it afterwards.
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_PRECISION", None)

    def probe():
        try:
            results["supported"] = cpu_features.has_cpu_bf16x3_precision()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            results["probe_error"] = exc
        finally:
            probe_done.set()

    def forward():
        try:
            assert read_done.wait(timeout=10)
            forward_started.set()
            with cpu_features.cpu_bf16x3_scope():
                results["scope_entered"] = True
                scope_entered.set()
                assert probe_done.wait(timeout=10)
                results["precision_in_scope"] = real_getter()
                results["depth_in_scope"] = cpu_features._CPU_BF16X3_DEPTH
        except Exception as exc:  # noqa: BLE001 - re-raised below
            results["forward_error"] = exc

    probe_thread = threading.Thread(target=probe, name="capability-probe")
    forward_thread = threading.Thread(target=forward, name="forward")
    try:
        probe_thread.start()
        assert read_done.wait(timeout=10)
        forward_thread.start()
        assert forward_started.wait(timeout=10)
        # Pre-fix the forward enters the scope while the probe is paused. With
        # the fix it blocks on the owner lock, so this wait times out.
        scope_entered.wait(timeout=1.0)
        release_probe.set()
        probe_thread.join(timeout=10)
        forward_thread.join(timeout=10)
    finally:
        release_probe.set()

    assert not probe_thread.is_alive()
    assert not forward_thread.is_alive()
    assert "probe_error" not in results, results.get("probe_error")
    assert "forward_error" not in results, results.get("forward_error")
    assert results["supported"] is True
    assert results["precision_in_scope"] == "bf16"
    assert results["depth_in_scope"] == 1
    assert state["precision"] == "ieee"
    assert cpu_features._CPU_BF16X3_DEPTH == 0
