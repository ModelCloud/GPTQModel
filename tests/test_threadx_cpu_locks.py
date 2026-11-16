# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU pool lock-scope identity handling (#3878).

The default pool config contains ``cpu`` and ``model_loader:cpu``. Those keys
share one ``_RWLock`` (a virtual pool reuses its parent's lock), while the
family map lists both keys. A family or ``all`` read scope therefore used to
acquire the same lock twice, which deadlocks: ``_RWLock.acquire_read`` blocks
new readers while a writer waits, so the second acquisition waits for a writer
that is itself waiting for the reader's first acquisition to be released.

These tests are CPU-only and deliberately avoid the module-level CUDA marker in
``tests/test_threadx.py``. Every wait is time-bounded so a regression fails
instead of hanging the suite.
"""

import threading

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool, _ReadLockGroup, _RWLock


LOCK_WAIT = 5.0
JOIN_WAIT = 5.0


@pytest.fixture
def cpu_alias_pool():
    pool = DeviceThreadPool(
        devices=[torch.device("cpu")],
        workers={"cpu": 1, "model_loader:cpu": 1},
        empty_cache_every_n=0,
    )
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def _lock_identities(group):
    return [id(lock) for _, lock in group._pairs]


def test_virtual_alias_shares_its_parent_lock(cpu_alias_pool):
    pool = cpu_alias_pool

    assert pool._locks["cpu"] is pool._locks["model_loader:cpu"]
    # Both keys are still visible in the family scope; deduplication happens at
    # acquisition time, not by hiding a pool key.
    assert "cpu" in pool._family_keys["cpu"]
    assert "model_loader:cpu" in pool._family_keys["cpu"]


def test_family_read_lock_acquires_the_shared_lock_once(cpu_alias_pool):
    pool = cpu_alias_pool

    with pool.read_lock("cpu") as group:
        identities = _lock_identities(group)

    assert len(identities) == len(set(identities)) == 1


def test_all_read_lock_acquires_each_lock_once(cpu_alias_pool):
    pool = cpu_alias_pool

    with pool.read_lock("all") as group:
        identities = _lock_identities(group)

    assert len(identities) == len(set(identities))
    assert len(identities) == len({id(lock) for lock in pool._locks.values()})


def test_wait_lock_true_acquires_each_lock_once(cpu_alias_pool):
    pool = cpu_alias_pool

    ctx = pool.wait(lock=True)
    with ctx:
        identities = _lock_identities(ctx._group)

    assert len(identities) == len(set(identities)) == 1


def test_family_read_lock_does_not_deadlock_against_a_waiting_writer(cpu_alias_pool):
    """Deterministic alias/read/write interleaving from the issue.

    The reader is paused after its first ``acquire_read`` so a writer can queue
    in ``_writers_waiting``. On the buggy code the reader's second acquisition
    of the same lock then waits behind that writer while the writer waits for
    the reader: neither thread can ever finish.
    """

    pool = cpu_alias_pool
    lock = pool._locks["cpu"]
    first_acquired = threading.Event()
    resume_reader = threading.Event()
    original_acquire = lock.acquire_read
    acquisitions = 0

    def controlled_acquire():
        nonlocal acquisitions
        original_acquire()
        acquisitions += 1
        if acquisitions == 1:
            first_acquired.set()
            # Bounded: a regression fails the assertions below instead of
            # blocking the reader forever.
            resume_reader.wait(LOCK_WAIT)

    def reader():
        with pool.read_lock("cpu"):
            pass

    def writer():
        with pool.device_lock("cpu"):
            pass

    lock.acquire_read = controlled_acquire
    reader_thread = threading.Thread(target=reader, name="alias-reader", daemon=True)
    writer_thread = threading.Thread(target=writer, name="alias-writer", daemon=True)
    try:
        reader_thread.start()
        assert first_acquired.wait(LOCK_WAIT), "reader never acquired the family read lock"

        writer_thread.start()
        deadline = LOCK_WAIT
        while lock._writers_waiting != 1 and deadline > 0:
            resume_reader.wait(0.005)
            deadline -= 0.005
        assert lock._writers_waiting == 1, "writer never queued behind the reader"

        resume_reader.set()

        reader_thread.join(JOIN_WAIT)
        writer_thread.join(JOIN_WAIT)

        assert not reader_thread.is_alive(), "family read lock deadlocked against the waiting writer"
        assert not writer_thread.is_alive(), "queued writer never acquired the lock"
    finally:
        resume_reader.set()
        lock.acquire_read = original_acquire


def test_lock_group_still_acquires_every_distinct_lock():
    """A genuine multi-lock scope keeps taking all locks, in order."""

    order = []

    class _RecordingLock(_RWLock):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def acquire_read(self):
            order.append(("acquire", self.name))
            super().acquire_read()

        def release_read(self):
            order.append(("release", self.name))
            super().release_read()

    first = _RecordingLock("first")
    second = _RecordingLock("second")

    with _ReadLockGroup([("first", first), ("second", second)]):
        assert first._readers == 1
        assert second._readers == 1

    assert order == [
        ("acquire", "first"),
        ("acquire", "second"),
        ("release", "second"),
        ("release", "first"),
    ]
    assert first._readers == 0
    assert second._readers == 0
