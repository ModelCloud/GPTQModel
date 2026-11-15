# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Process-global quantization guards must be reference-counted.

Two concurrent ``quantize()`` calls share one process-global BLAS backend and
one TF32 setting. The old save/restore-per-entry guards let the first run to
finish restore the user's setting while the other run was still quantizing, so
the later layers used hipBLASLt/TF32. Issue #4094 measured 177-210 of 794
tensors differing from a serial run. These tests pin the counting contract on
the guards themselves, without a GPU.
"""

import threading

import gptqmodel.utils.rocm as rocm_utils
import gptqmodel.utils.torch as torch_utils


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
        except BaseException as exc:  # noqa: BLE001 - re-raised below
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
    monkeypatch.setattr(rocm_utils, "_BLAS_SCOPE_DEPTH", 0)
    monkeypatch.setattr(rocm_utils, "_BLAS_SCOPE_RESTORE", None)

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
    assert rocm_utils._BLAS_SCOPE_DEPTH == 0


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
