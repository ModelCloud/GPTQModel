# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The TF32/BF16x3/num_threads switches may not straddle a dynamo trace.

dynamo's ``preserve_global_state`` snapshots the process-global torch state
when a trace starts and writes it back when the trace ends. A quantization
guard that only counted its own entries could be overwritten mid-window by a
trace in another thread (#5390): the user's ``tf32`` was restored while
``quantize()`` was still inside the guard, so the checkpoint silently differed
from a serial run. The depth-0 switches now also take the model-global-state
writer, which excludes in-flight compiled calls and their traces.
"""

import threading
from contextlib import contextmanager

import torch

import gptqmodel.utils.threadx as threadx
import gptqmodel.utils.torch as torch_utils
from gptqmodel.utils import cpu_features


def _install_tf32_state(monkeypatch, precision="tf32"):
    state = {"precision": precision, "writes": []}
    lock = torch_utils.model_global_state_lock()

    def snapshot():
        return state["precision"]

    def set_enabled(value):
        state["writes"].append(("set", value, lock.holds_writer()))
        state["precision"] = "ieee" if value is False else "tf32"

    def restore(value):
        state["writes"].append(("restore", value, lock.holds_writer()))
        state["precision"] = value

    monkeypatch.setattr(torch_utils, "HAS_CUDA", True)
    monkeypatch.setattr(torch_utils, "_snapshot_tf32_state", snapshot)
    monkeypatch.setattr(torch_utils, "_set_tf32_state", set_enabled)
    monkeypatch.setattr(torch_utils, "_restore_tf32_state", restore)
    monkeypatch.setattr(torch_utils, "_TF32_GUARD_DEPTH", 0)
    monkeypatch.setattr(torch_utils, "_TF32_GUARD_PREVIOUS", None)
    return state, lock


def test_tf32_guard_takes_the_writer_for_the_switches_only(monkeypatch):
    state, lock = _install_tf32_state(monkeypatch)

    with torch_utils.tf32_high_precision_guard():
        assert state["precision"] == "ieee"
        # Only the switch takes the writer, not the whole quantization.
        assert not lock.holds_writer()

    assert state["precision"] == "tf32"
    assert state["writes"] == [
        ("set", False, True),
        ("restore", "tf32", True),
    ]


def test_tf32_guard_waits_out_an_in_flight_trace(monkeypatch):
    """The switch must run after a concurrent trace writes its snapshot back."""

    state, lock = _install_tf32_state(monkeypatch)
    tracing = threading.Event()
    release_trace = threading.Event()
    guard_entered = threading.Event()
    release_guard = threading.Event()
    errors = []

    def trace():
        try:
            # A compiled call holds the shared side for the whole call,
            # including the trace that dynamo runs inside it.
            with lock.reader():
                captured = state["precision"]
                tracing.set()
                if not release_trace.wait(timeout=10):
                    raise TimeoutError("trace was never released")
                # dynamo's preserve_global_state writes the snapshot back.
                state["precision"] = captured
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    def quantize():
        try:
            with torch_utils.tf32_high_precision_guard():
                guard_entered.set()
                if not release_guard.wait(timeout=10):
                    raise TimeoutError("guard was never released")
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    trace_thread = threading.Thread(target=trace)
    trace_thread.start()
    assert tracing.wait(timeout=10)

    guard_thread = threading.Thread(target=quantize)
    guard_thread.start()

    # The guard may not apply its override while the trace is in flight: the
    # trace's write-back would restore "tf32" for the rest of the guard.
    assert not guard_entered.wait(timeout=0.5)
    assert state["precision"] == "tf32"

    release_trace.set()
    trace_thread.join(timeout=10)
    assert guard_entered.wait(timeout=10)
    # The switch ran after the trace wrote its snapshot back, so the guard's
    # "ieee" is the last write and the trace cannot undo it.
    assert state["precision"] == "ieee"

    release_guard.set()
    guard_thread.join(timeout=10)

    assert not errors
    assert state["precision"] == "tf32"
    assert torch_utils._TF32_GUARD_DEPTH == 0


def test_cpu_bf16x3_scope_takes_the_writer_for_the_switches(monkeypatch):
    state = {"precision": "none", "writes": []}
    lock = torch_utils.model_global_state_lock()

    def getter():
        return state["precision"]

    def setter(value):
        state["writes"].append(lock.holds_writer())
        state["precision"] = value

    monkeypatch.setattr(cpu_features, "_cpu_bf16x3_precision", getter)
    monkeypatch.setattr(cpu_features, "_set_cpu_bf16x3_precision", setter)
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_DEPTH", 0)
    monkeypatch.setattr(cpu_features, "_CPU_BF16X3_RESTORE", None)

    with cpu_features.cpu_bf16x3_scope():
        with cpu_features.cpu_bf16x3_scope():
            pass
        assert state["precision"] == "bf16"

    assert state["precision"] == "none"
    assert state["writes"] == [True, True]


def test_worker_intraop_install_takes_the_writer(monkeypatch):
    calls = []
    current = {"threads": 8}

    @contextmanager
    def recording_guard():
        calls.append(threading.current_thread().name)
        yield

    def get_threads():
        return current["threads"]

    def set_threads(value):
        current["threads"] = value

    monkeypatch.setattr(torch_utils, "model_global_state_write_guard", recording_guard)
    monkeypatch.setattr(torch, "get_num_threads", get_threads)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)

    threadx.install_worker_intraop_threads(2)

    # The transient worker budget and its restore both went through the
    # global-state writer, and the restore ran from the helper thread. The
    # writer itself is covered by the TF32/BF16x3 tests above; this pins the
    # two call sites.
    assert current["threads"] == 8
    assert calls == [threading.current_thread().name, "threadx-intraop-restore"]


def test_write_guard_skips_upgrade_inside_a_compiled_call():
    lock = torch_utils.model_global_state_lock()

    with lock.reader():
        # A thread inside a compiled call holds the reader; taking the writer
        # would deadlock against its own hold, so the switch skips it.
        with torch_utils.model_global_state_write_guard():
            assert not lock.holds_writer()
        assert lock.holds_reader()

    assert not lock.holds_reader()
