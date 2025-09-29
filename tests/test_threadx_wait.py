# SPDX-License-Identifier: Apache-2.0
# Author: ModelCloud.ai / qubitium
import threading
import time

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def require_n_gpus(n: int):
    if torch.cuda.device_count() < n:
        pytest.skip(f"requires >= {n} CUDA devices")


# ---------- Helpers ----------

def _host_long(ms=150):
    # Host-side delay (keeps worker thread busy, independent of GPU concurrency)
    time.sleep(ms / 1000.0)
    return ms


def _start_then_sleep(start_evt: threading.Event, ms=200):
    start_evt.set()
    time.sleep(ms / 1000.0)
    return ms


# ---------- Fixtures ----------

@pytest.fixture()
def pool_default_two_cuda():
    """Default pool with 2 CUDA devices, 1 worker per device."""
    require_n_gpus(2)
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    p = DeviceThreadPool(devices=devices, inference_mode=True, empty_cache_every_n=0)
    try:
        yield p
    finally:
        p.shutdown(wait=True)


@pytest.fixture()
def pool_workers_override():
    """
    Pool validating worker-count overrides:
      - cuda:0 -> 3 workers (override)
      - cuda:1 -> 1 worker (via 'cuda:per')
      - cpu    -> 4 workers
    """
    require_n_gpus(2)
    devices = [torch.device("cuda", 0), torch.device("cuda", 1), torch.device("cpu")]
    p = DeviceThreadPool(
        devices=devices,
        inference_mode=True,
        empty_cache_every_n=0,
        workers={"cuda:per": 1, "cuda:0": 3, "cpu": 4},
    )
    try:
        yield p
    finally:
        p.shutdown(wait=True)


# ---------- wait() API tests ----------

def test_wait_cuda_without_lock(pool_default_two_cuda):
    d0 = torch.device("cuda", 0)

    # Submit several long host tasks to d0
    futs = [pool_default_two_cuda.submit(d0, _host_long, 150) for _ in range(3)]

    # Wait for CUDA scope to drain
    pool_default_two_cuda.wait("cuda")

    # All should be done now
    for f in futs:
        assert f.done()
        assert f.result() == 150


def test_wait_cuda_with_lock_blocks_new_tasks(pool_default_two_cuda):
    d0 = torch.device("cuda", 0)

    started_after_lock = threading.Event()

    # Submit one long task so there's something to drain
    f0 = pool_default_two_cuda.submit(d0, _host_long, 120)

    # Acquire wait+lock over all CUDA devices
    with pool_default_two_cuda.wait("cuda", lock=True):
        # At this point, the initial task finished and we hold the writer lock.
        # Submit a new task: it should enqueue but **not** start until we exit.
        fut = pool_default_two_cuda.submit(d0, _start_then_sleep, started_after_lock, 200)

        # Give the worker a moment; it must NOT start while lock is held.
        time.sleep(0.05)
        assert not started_after_lock.is_set()
        assert not fut.done()

    # After releasing the lock, the task can start and complete
    assert fut.result(timeout=2) == 200
    assert started_after_lock.is_set()
    assert f0.result(timeout=0.5) == 120


def test_wait_specific_device_vs_family(pool_default_two_cuda):
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)

    started_d0 = threading.Event()
    started_d1 = threading.Event()

    fut0 = pool_default_two_cuda.submit(d0, _start_then_sleep, started_d0, 200)
    fut1 = pool_default_two_cuda.submit(d1, _start_then_sleep, started_d1, 200)

    # Wait only for cuda:0; cuda:1 may still be running.
    pool_default_two_cuda.wait("cuda:0")

    # d0 must be done; d1 may or may not be done depending on timing, but no deadlocks.
    assert fut0.done()
    assert started_d0.is_set()
    assert fut0.result(timeout=0.5) == 200

    # To be deterministic, drain all CUDA before leaving the test
    pool_default_two_cuda.wait("cuda")
    assert fut1.result(timeout=0.5) == 200
    assert started_d1.is_set()


def test_wait_returns_context_manager_with_lock(pool_default_two_cuda):
    d0 = torch.device("cuda", 0)

    with pool_default_two_cuda.wait("cuda", lock=True):
        # While holding the exclusive lock, submitting a task will park at the lock
        started = threading.Event()
        fut = pool_default_two_cuda.submit(d0, _start_then_sleep, started, 120)
        time.sleep(0.05)
        assert not started.is_set()
        assert not fut.done()

    # After lock release, it proceeds
    assert fut.result(timeout=2) == 120
    assert started.is_set()


# ---------- Worker-count policy tests ----------

def test_worker_count_override_cuda0_vs_cuda1(pool_workers_override):
    """
    With workers={"cuda:per":1, "cuda:0":3}:
      - cuda:0 has 3 workers -> up to 3 tasks can start almost immediately.
      - cuda:1 has 1 worker  -> tasks start one-by-one.
    We measure host-level concurrency via start events.
    """
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)

    # cuda:0 — expect ~3 tasks to start quickly
    start_events_0 = [threading.Event() for _ in range(6)]
    futs0 = [pool_workers_override.submit(d0, _start_then_sleep, ev, 200) for ev in start_events_0]
    time.sleep(0.10)  # give workers time to start tasks
    started0 = sum(ev.is_set() for ev in start_events_0)
    assert started0 >= 3  # at least the configured worker count
    # Drain
    for f in futs0:
        assert f.result(timeout=3) == 200

    # cuda:1 — only 1 worker; after a short delay, only ~1 task should have started
    start_events_1 = [threading.Event() for _ in range(4)]
    futs1 = [pool_workers_override.submit(d1, _start_then_sleep, ev, 150) for ev in start_events_1]
    time.sleep(0.08)
    started1 = sum(ev.is_set() for ev in start_events_1)
    assert started1 <= 2  # typically 1; allow 2 to avoid flakiness
    # Drain
    for f in futs1:
        assert f.result(timeout=3) == 150


def test_worker_count_override_cpu(pool_workers_override):
    """
    With workers={"cpu":4}, we expect ~4 CPU tasks to start quickly in parallel.
    """
    d_cpu = torch.device("cpu")

    starts = [threading.Event() for _ in range(8)]
    futs = [pool_workers_override.submit(d_cpu, _start_then_sleep, ev, 200) for ev in starts]

    time.sleep(0.10)
    started = sum(ev.is_set() for ev in starts)
    assert started >= 4  # at least configured worker count should be active

    for f in futs:
        assert f.result(timeout=3) == 200


def test_wait_all_scope_with_mixed_devices(pool_workers_override):
    """
    Ensure wait(None)/wait('all') drains all devices (CPU + both CUDA).
    """
    d_cpu = torch.device("cpu")
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)

    futs = []
    for _ in range(3):
        futs.append(pool_workers_override.submit(d_cpu, _host_long, 120))
        futs.append(pool_workers_override.submit(d0, _host_long, 120))
        futs.append(pool_workers_override.submit(d1, _host_long, 120))

    # Wait for everything
    pool_workers_override.wait()           # same as wait('all')
    pool_workers_override.wait('all')      # idempotent

    for f in futs:
        assert f.done()
        assert f.result(timeout=0.5) == 120


def test_wait_cuda_lock_allows_other_families(pool_workers_override):
    """
    Holding wait('cuda', lock=True) should not block CPU tasks.
    """
    d_cpu = torch.device("cpu")
    d0 = torch.device("cuda", 0)

    # Fill CUDA with a couple tasks
    fut0 = pool_workers_override.submit(d0, _host_long, 120)
    fut1 = pool_workers_override.submit(d0, _host_long, 120)

    with pool_workers_override.wait("cuda", lock=True):
        # CUDA is drained & locked exclusively here
        # CPU task should still start+finish
        cpu_done = threading.Event()

        def cpu_task():
            cpu_done.set()
            return _host_long(50)

        f_cpu = pool_workers_override.submit(d_cpu, cpu_task)
        # Give it a moment; must run even while CUDA is locked
        time.sleep(0.02)
        assert cpu_done.is_set()
        assert f_cpu.result(timeout=2) == 50

        # Submitting a CUDA task now will queue but not start
        started = threading.Event()
        f_blocked = pool_workers_override.submit(d0, _start_then_sleep, started, 80)
        time.sleep(0.03)
        assert not started.is_set()
        assert not f_blocked.done()

    # After release, the blocked CUDA task proceeds
    assert fut0.result(timeout=2) == 120
    assert fut1.result(timeout=2) == 120
    assert f_blocked.result(timeout=2) == 80
