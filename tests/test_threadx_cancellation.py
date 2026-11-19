# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Cancellation is a future-state contract, not a hint to the worker.

``DeviceThreadPool.submit`` returns a ``concurrent.futures.Future``, so callers
expect ``future.cancel()`` to mean something. Before this change it did not: the
worker never moved the future out of PENDING, so

* ``cancel()`` returned True for a task that was already running, and the worker
  then silently dropped that task's result or exception, and
* a task cancelled while still queued ran anyway, spending a device slot on
  work the caller had abandoned.

The contract pinned here is the one ``concurrent.futures`` documents: the
PENDING -> RUNNING transition happens on the worker thread immediately before
the task runs. A future cancelled while queued is skipped in place, so the tasks
submitted before and after it keep their order, and it releases its in-flight
slot without being counted as completed work. A future whose task has started
refuses ``cancel()`` and always resolves with the task's result or exception.
"""

import threading
from concurrent.futures import CancelledError

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool


def _pool() -> DeviceThreadPool:
    return DeviceThreadPool(
        devices=[torch.device("cpu")],
        workers={"cpu": 1},
        empty_cache_every_n=0,
    )


def _drain(pool: DeviceThreadPool, key: str = "cpu", timeout: float = 30.0) -> None:
    """Run ``pool.wait(key)`` under a bound; a leaked slot would hang it."""

    waiter = threading.Thread(target=lambda: pool.wait(key), daemon=True)
    waiter.start()
    waiter.join(timeout)
    assert not waiter.is_alive(), "pool.wait() blocked on a count that never drained"


def _occupy(pool: DeviceThreadPool):
    """Start a task that holds the only worker until the returned release is set."""

    entered = threading.Event()
    release = threading.Event()

    def blocker():
        entered.set()
        assert release.wait(30), "the test never released the blocker"
        return "blocker"

    future = pool.submit("cpu", blocker)
    assert entered.wait(30), "the blocker never started"
    return future, release


def test_cancelled_queued_task_never_runs():
    """A task cancelled while queued must not execute or leave side effects."""

    pool = _pool()
    try:
        blocker, release = _occupy(pool)

        ran = threading.Event()

        def abandoned():
            ran.set()
            return "abandoned"

        queued = pool.submit("cpu", abandoned)
        assert queued.cancel() is True, "a queued future must accept cancellation"

        release.set()
        assert blocker.result(timeout=30) == "blocker"

        # The drain proves the worker consumed the cancelled queue item, so a
        # set ``ran`` event below can only come from executing it.
        _drain(pool)

        assert queued.cancelled()
        with pytest.raises(CancelledError):
            queued.result(timeout=1)
        assert not ran.is_set(), "a cancelled queued task still ran"
        assert pool._inflight["cpu"] == 0
    finally:
        pool.shutdown(wait=True)


def test_cancelled_task_keeps_its_neighbours_in_order():
    """Skipping a cancelled item must not reorder or drop the tasks around it."""

    pool = _pool()
    try:
        blocker, release = _occupy(pool)

        order = []

        def record(name):
            order.append(name)
            return name

        abandoned = pool.submit("cpu", record, "abandoned")
        assert abandoned.cancel() is True
        following = pool.submit("cpu", record, "following")

        release.set()
        assert blocker.result(timeout=30) == "blocker"
        assert following.result(timeout=30) == "following"
        _drain(pool)

        assert order == ["following"], f"unexpected execution order: {order}"
        assert abandoned.cancelled()
    finally:
        pool.shutdown(wait=True)


def test_running_task_refuses_cancel_and_still_resolves():
    """``cancel()`` must fail once the task runs, and its result must arrive."""

    pool = _pool()
    try:
        started = threading.Event()
        release = threading.Event()

        def running():
            started.set()
            assert release.wait(30)
            return "result"

        future = pool.submit("cpu", running)
        assert started.wait(30), "the task never started"

        assert future.cancel() is False, "a running task must not report cancellation"
        release.set()
        assert future.result(timeout=30) == "result"
        assert not future.cancelled()
    finally:
        pool.shutdown(wait=True)


def test_running_task_exception_survives_a_cancel_attempt():
    """A late ``cancel()`` must not swallow the exception the caller waits for."""

    class Boom(RuntimeError):
        pass

    pool = _pool()
    try:
        started = threading.Event()
        release = threading.Event()

        def failing():
            started.set()
            assert release.wait(30)
            raise Boom("task failed")

        future = pool.submit("cpu", failing)
        assert started.wait(30), "the task never started"

        assert future.cancel() is False
        release.set()
        with pytest.raises(Boom, match="task failed"):
            future.result(timeout=30)
    finally:
        pool.shutdown(wait=True)


def test_cancelled_task_releases_inflight_without_counting_completion():
    """The slot must drain; the janitor's completion counters must not move."""

    pool = _pool()
    try:
        blocker, release = _occupy(pool)

        completed_before = pool.stats()["total"]
        queued = pool.submit("cpu", lambda: "abandoned")
        assert queued.cancel() is True
        release.set()
        assert blocker.result(timeout=30) == "blocker"
        _drain(pool)

        # Only the blocker ran; the cancelled task never advanced the counters
        # that drive GC thresholds and ``stats()``.
        assert pool.stats()["total"] == completed_before + 1
        assert pool._inflight["cpu"] == 0
    finally:
        pool.shutdown(wait=True)


def test_cancel_before_a_worker_picks_the_task_up():
    """Cancellation must win over the worker's queue scan, never lose to it."""

    pool = _pool()
    try:
        blocker, release = _occupy(pool)

        ran = threading.Event()
        futures = []
        for _ in range(5):
            future = pool.submit("cpu", lambda: ran.set() or "ran")
            futures.append(future)
        for future in futures:
            assert future.cancel() is True, "a queued future must accept cancellation"

        release.set()
        assert blocker.result(timeout=30) == "blocker"
        _drain(pool)

        assert not ran.is_set(), "a cancelled queued task still ran"
        assert all(future.cancelled() for future in futures)
        assert pool._inflight["cpu"] == 0
    finally:
        pool.shutdown(wait=True)


def test_cancelled_serial_task_never_runs():
    """``submit_serial`` shares the worker path, so it must share the contract."""

    pool = _pool()
    try:
        entered = threading.Event()
        release = threading.Event()

        def blocker():
            entered.set()
            assert release.wait(30)
            return "blocker"

        first = pool.submit_serial("cpu", blocker)
        assert entered.wait(30), "the blocker never started"

        ran = threading.Event()

        def abandoned():
            ran.set()
            return "abandoned"

        queued = pool.submit_serial("cpu", abandoned)
        assert queued.cancel() is True

        release.set()
        assert first.result(timeout=30) == "blocker"
        _drain(pool)

        assert not ran.is_set(), "a cancelled serial task still ran"
        assert queued.cancelled()
        assert pool._inflight["cpu"] == 0
    finally:
        pool.shutdown(wait=True)
