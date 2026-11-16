# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shutdown is one contract: drain accepted work, then reject new work.

A worker exits on its sentinel, and ``stop()`` queues that sentinel after every
task the pool already accepted. Breaking out of the run loop as soon as the stop
flag was set instead abandoned the queued tasks: their futures stayed pending
forever and their in-flight counts never dropped, so ``future.result()`` and
``pool.wait()`` could hang. Submission had the mirror problem -- nothing stopped
a caller from enqueueing behind a sentinel, or from lazily starting a fresh
worker after the pool had shut down.
"""

import threading
import time

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool, _SyncWorker


def _pool(workers: int = 1) -> DeviceThreadPool:
    return DeviceThreadPool(
        devices=[torch.device("cpu")],
        workers={"cpu": workers},
        empty_cache_every_n=0,
    )


def test_shutdown_drains_a_queued_task():
    """A task accepted before shutdown still runs and resolves its future."""
    pool = _pool()
    entered, release = threading.Event(), threading.Event()

    def first():
        entered.set()
        assert release.wait(10), "the test never released the blocking task"
        return "first"

    first_future = pool.submit("cpu", first)
    assert entered.wait(10), "the first task never started"

    # Accepted while the single worker is busy, so it sits in the queue.
    second_future = pool.submit("cpu", lambda: "second")

    shutdown_thread = threading.Thread(target=lambda: pool.shutdown(wait=True), daemon=True)
    shutdown_thread.start()

    # Wait for the worker to receive its stop request before releasing the
    # blocked task; otherwise the queue order is not yet established.
    worker = pool._worker_groups["cpu"][0]
    assert worker._stop.wait(10), "the worker never received the stop request"
    release.set()

    shutdown_thread.join(30)
    assert not shutdown_thread.is_alive(), "shutdown did not return"

    assert first_future.done() and first_future.result(timeout=30) == "first"
    assert second_future.done(), "the queued task was abandoned"
    assert second_future.result(timeout=30) == "second"
    assert pool._inflight["cpu"] == 0, f"in-flight leaked: {pool._inflight}"

    # wait() must return instead of blocking on a count that never drains.
    waiter = threading.Thread(target=lambda: pool.wait("cpu"), daemon=True)
    waiter.start()
    waiter.join(30)
    assert not waiter.is_alive(), "wait() blocked on a count that never drained"


def test_shutdown_drains_every_queued_task():
    """Draining is not limited to a single queued item."""
    pool = _pool()
    entered, release = threading.Event(), threading.Event()

    def first():
        entered.set()
        assert release.wait(10)
        return 0

    pool.submit("cpu", first)
    assert entered.wait(10)
    queued = [pool.submit("cpu", lambda index=index: index) for index in range(5)]

    shutdown_thread = threading.Thread(target=lambda: pool.shutdown(wait=True), daemon=True)
    shutdown_thread.start()
    assert pool._worker_groups["cpu"][0]._stop.wait(10)
    release.set()
    shutdown_thread.join(30)

    assert not shutdown_thread.is_alive()
    assert [future.result(timeout=30) for future in queued] == [0, 1, 2, 3, 4]
    assert pool._inflight["cpu"] == 0


def test_submit_after_shutdown_is_rejected():
    pool = _pool()
    pool.shutdown(wait=True)

    with pytest.raises(RuntimeError, match="shutting down"):
        pool.submit("cpu", lambda: None)


def test_submit_serial_after_shutdown_is_rejected():
    pool = _pool()
    pool.shutdown(wait=True)

    with pytest.raises(RuntimeError, match="shutting down"):
        pool.submit_serial("cpu", lambda: None)


def test_lazy_worker_selection_does_not_spawn_after_shutdown():
    """An empty group must not be repopulated by a late submission."""
    pool = _pool()
    pool.shutdown(wait=True)

    # Every worker removed itself on exit, so the lazy-spawn fallback in
    # worker selection is reachable here.
    assert not pool._worker_groups.get("cpu")

    with pytest.raises(RuntimeError, match="shutting down"):
        pool.submit("cpu", lambda: None)

    assert not pool._worker_groups.get("cpu"), "a worker was spawned after shutdown"


def test_a_shutdown_pool_rejects_work_promptly():
    """Rejection must not depend on a worker being available."""
    pool = _pool()
    pool.shutdown(wait=True)

    started = time.monotonic()
    for _ in range(50):
        with pytest.raises(RuntimeError):
            pool.submit("cpu", lambda: None)
    assert time.monotonic() - started < 5.0


def test_sync_worker_stop_does_not_self_deadlock():
    """The dispatch lock must be reentrant for a worker whose stop re-enters it.

    ``_SyncWorker.stop()`` calls ``_on_worker_exit`` synchronously, and that
    takes the dispatch lock. ``shutdown()`` issues stops while holding the same
    lock, so a plain ``Lock`` there would deadlock the shutting-down thread.
    """
    pool = _pool()
    worker = _SyncWorker(
        key="cpu",
        device=torch.device("cpu"),
        rwlock=pool._locks["cpu"],
        on_task_finished=pool._on_task_finished,
        on_worker_exit=pool._on_worker_exit,
    )
    pool._worker_groups["cpu"].append(worker)

    shutdown_thread = threading.Thread(target=lambda: pool.shutdown(wait=True), daemon=True)
    shutdown_thread.start()
    shutdown_thread.join(30)

    assert not shutdown_thread.is_alive(), "shutdown deadlocked on the dispatch lock"
