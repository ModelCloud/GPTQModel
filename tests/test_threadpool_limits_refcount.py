# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression tests for the reference-counted threadpool limits guard (#4206).

Native thread-pool limits are process-wide. The plain ``threadpoolctl`` scope
saves the limits on every entry and restores them on every exit, so two
overlapping scopes leave the process pinned at the reduced thread count when
the scope that saved the reduced value exits last.
"""

import threading
import time

from threadpoolctl import threadpool_info

from gptqmodel.utils.safe import THREADPOOLCTL, threadpool_limits


def _limits():
    return {entry["internal_api"]: entry["num_threads"] for entry in threadpool_info()}


def _openmp_threads():
    return [
        entry["num_threads"]
        for entry in threadpool_info()
        if entry["user_api"] == "openmp"
    ]


def _assert_limited():
    assert 1 in _limits().values(), "the scope did not apply limits=1"


def test_single_use_restores_original_limits():
    original = _limits()
    assert original

    with threadpool_limits(limits=1):
        _assert_limited()

    assert _limits() == original


def test_nested_scope_restores_after_outer_exit():
    original = _limits()
    assert original

    with threadpool_limits(limits=1):
        with threadpool_limits(limits=1):
            _assert_limited()
        # The outer scope still owns the override.
        _assert_limited()

    assert _limits() == original


def test_overlapping_threads_restore_original_limits():
    """The first thread to exit must not restore while the other is inside.

    A enters first and exits first; B enters second (saving the already-limited
    value) and exits last. With per-entry save/restore the final limits are the
    reduced ones; the refcounted guard restores the original limits.
    """

    original = _limits()
    assert original

    a_entered = threading.Event()
    b_entered = threading.Event()
    failures = []

    def thread_a():
        try:
            with threadpool_limits(limits=1):
                a_entered.set()
                assert b_entered.wait(timeout=10), "thread B never entered"
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    def thread_b():
        try:
            assert a_entered.wait(timeout=10), "thread A never entered"
            with threadpool_limits(limits=1):
                b_entered.set()
                # Keep B inside while A exits; this is the racy ordering.
                time.sleep(0.3)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    threads = [
        threading.Thread(target=thread_a, name="limits-a"),
        threading.Thread(target=thread_b, name="limits-b"),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive(), f"{thread.name} did not finish"

    assert not failures, failures
    assert _limits() == original


def test_per_thread_openmp_limit_stays_scoped_to_its_thread():
    """OpenMP limits are thread-local, so each thread keeps its own scope."""

    original = _limits()
    a_entered = threading.Event()
    b_entered = threading.Event()
    observed = []
    failures = []

    def worker(entered, other):
        try:
            with threadpool_limits(limits=1):
                observed.append(_openmp_threads())
                entered.set()
                assert other.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(a_entered, b_entered)),
        threading.Thread(target=worker, args=(b_entered, a_entered)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()

    assert not failures, failures
    # When OpenMP controllers are present, both overlapping workers see 1.
    for counts in observed:
        assert not counts or all(count == 1 for count in counts), observed
    assert _limits() == original


def test_exception_in_body_still_restores_original_limits():
    original = _limits()
    assert original

    try:
        with threadpool_limits(limits=1):
            _assert_limited()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert _limits() == original


def test_threadpoolctl_proxy_uses_refcounted_guard():
    # The public THREADPOOLCTL API keeps working and routes through the guard.
    assert THREADPOOLCTL.threadpool_limits is threadpool_limits

    original = _limits()
    with THREADPOOLCTL.threadpool_limits(limits=1):
        _assert_limited()
    assert _limits() == original
