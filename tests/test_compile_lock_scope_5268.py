# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compiled forwards must not serialize against each other or a load's I/O.

#5251 fixed #5209 by making every compiled call hold the process-wide
``_MODEL_LOAD_LOCK`` for its whole duration. That serialized compiled forwards
across threads and models and stalled them for the full duration of any load or
save (#5268). The fix keeps compiled calls as *readers* of a process-global
reader/writer lock and gives the writer only to the transformers construction
window that saves, sets and restores ``torch.set_default_dtype``.

These tests pin the two new contracts:

* two warm compiled forwards run concurrently instead of queueing behind each
  other, and
* a compiled forward runs while another thread holds ``_MODEL_LOAD_LOCK`` for a
  load's weight I/O; only the construction writer may block it.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from gptqmodel.models import loader as loader_module
from gptqmodel.utils.torch import dynamo_traced_frames, torch_compile


def _require_dynamo():
    return pytest.importorskip("torch._dynamo")


_OVERLAP_DELAY = 0.2


def _overlap_body(value):
    # Fixed, module-level delay: the body must not read mutable test state, or
    # dynamo re-traces and the second thread takes the ``is_compiling()``
    # inline path instead of the warm compiled path under test.
    time.sleep(_OVERLAP_DELAY)
    return value * 2


def test_two_compiled_forwards_overlap():
    """Two warm compiled calls must run concurrently, not one after the other.

    Pre-fix each call holds ``_MODEL_LOAD_LOCK`` for its whole body, so the
    second thread waits for the first and the pair takes two delays. Post-fix
    both calls hold the shared reader side and take one delay.
    """

    _require_dynamo()
    compiled = torch_compile(_overlap_body, backend="eager")
    value = torch.ones(2)
    assert compiled(value).tolist() == [2.0, 2.0], "the frame was not compiled"
    assert compiled(value).tolist() == [2.0, 2.0]
    frames_before = dynamo_traced_frames()

    per_call = {}
    errors = []

    def worker(tag):
        try:
            started = time.perf_counter()
            compiled(value)
            per_call[tag] = time.perf_counter() - started
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(tag,), daemon=True)
        for tag in ("t1", "t2")
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.perf_counter() - started

    assert not any(thread.is_alive() for thread in threads), (
        "a compiled forward did not finish"
    )
    assert not errors, f"compiled forwards failed: {errors!r}"
    assert dynamo_traced_frames() == frames_before, (
        "the overlap run re-traced; it did not exercise the warm compiled path"
    )
    assert per_call["t1"] < _OVERLAP_DELAY * 1.8 and per_call["t2"] < _OVERLAP_DELAY * 1.8, (
        "the two compiled forwards serialized: "
        f"per-call { {k: round(v, 3) for k, v in per_call.items()} }"
    )
    assert elapsed < _OVERLAP_DELAY * 1.8, (
        f"two compiled forwards took {elapsed:.3f}s, i.e. two {_OVERLAP_DELAY}s delays"
    )


def test_compiled_forward_does_not_wait_for_the_load_mutex():
    """A load's weight I/O must not stall a compiled forward.

    ``_MODEL_LOAD_LOCK`` is held for the whole load so two loads cannot
    interleave. Compiled calls must not take it; only the global-state writer
    around the dtype window may exclude them.
    """

    _require_dynamo()
    compiled = torch_compile(lambda value: value * 2, backend="eager")
    value = torch.ones(2)
    assert compiled(value).tolist() == [2.0, 2.0], "the frame was not compiled"

    holding = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []

    def holder():
        with loader_module.model_load_lock():
            holding.set()
            release.wait(timeout=60)

    def worker():
        try:
            compiled(value)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            finished.set()

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    assert holding.wait(timeout=30), "the load-mutex holder never started"

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    try:
        assert finished.wait(timeout=2), (
            "a compiled forward waited for the whole load lock; only the "
            "default-dtype construction window may block it"
        )
    finally:
        release.set()
        holder_thread.join(timeout=60)
        worker_thread.join(timeout=60)

    assert not errors, f"the compiled forward failed: {errors!r}"


def test_global_state_lock_shares_readers_and_excludes_them_from_a_writer():
    """The lock itself: readers overlap, and a writer excludes both sides."""

    from gptqmodel.models.loader import _ModelGlobalStateLock

    lock = _ModelGlobalStateLock()
    both_readers = threading.Barrier(2)
    release = threading.Event()
    errors = []

    def reader():
        try:
            with lock.reader():
                both_readers.wait(timeout=10)
                release.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(2)]
    for thread in readers:
        thread.start()
    # Both readers passed the barrier together; let them leave the lock.
    time.sleep(0.2)
    release.set()
    for thread in readers:
        thread.join(timeout=10)
    assert not errors, f"readers did not overlap: {errors!r}"

    entered = threading.Event()
    release_writer = threading.Event()
    left = threading.Event()
    writer_errors = []

    def writer():
        try:
            with lock:
                entered.set()
                release_writer.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 - reported below
            writer_errors.append(exc)
        finally:
            left.set()

    # A writer must wait for an in-flight reader ...
    with lock.reader():
        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()
        time.sleep(0.1)
        assert not entered.is_set(), "the writer overlapped an in-flight reader"
    assert entered.wait(timeout=10), "the writer never acquired the lock"

    # ... and a reader that starts while the writer is active must wait for it.
    reader_done = threading.Event()

    def late_reader():
        with lock.reader():
            reader_done.set()

    late = threading.Thread(target=late_reader, daemon=True)
    late.start()
    try:
        assert not reader_done.wait(timeout=0.2), (
            "a reader entered while the writer held the lock"
        )
    finally:
        release_writer.set()
        late.join(timeout=10)
        writer_thread.join(timeout=10)

    assert reader_done.is_set(), "the reader never resumed after the writer left"
    assert left.is_set()
    assert not writer_errors, f"writer failed: {writer_errors!r}"


def test_global_state_lock_is_reentrant_for_load_threads():
    """A load thread may run compiled code (reader) while it owns the writer."""

    from gptqmodel.models.loader import _ModelGlobalStateLock

    lock = _ModelGlobalStateLock()
    with lock:
        with lock:
            with lock.reader():
                with lock.reader():
                    pass
    assert lock._writer is None and lock._readers == 0
