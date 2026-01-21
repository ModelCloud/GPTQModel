# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn as nn

from models.model_test import ModelTest

from gptqmodel.utils import threadx as threadx_mod
from gptqmodel.utils.threadx import DeviceThreadPool


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def require_n_gpus(n):
    if torch.cuda.device_count() < n:
        pytest.skip(f"requires >= {n} CUDA devices")


# ----------------------------- Fixtures -----------------------------

@pytest.fixture(scope="module")
def devices_two():
    require_n_gpus(2)
    return [torch.device("cuda", 0), torch.device("cuda", 1)]


@pytest.fixture()
def pool(devices_two):
    # small threshold for fast tests
    p = DeviceThreadPool(devices=devices_two, inference_mode=True, empty_cache_every_n=3)
    yield p
    p.shutdown(wait=True)


# ----------------------------- Helpers ------------------------------

def _sleep_kernel_ms(ms: int):
    """Lightweight delay on device via cuda._sleep; fall back to host sleep."""
    cycles = int(ms * 1_000_000)
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(cycles)
    else:
        time.sleep(ms / 1000.0)


# ----------------------------- Core Tests ---------------------------

def test_basic_submit_and_do(pool, devices_two):
    d0 = devices_two[0]

    def add(a, b):
        return a + b

    a = torch.randn(512, 512, device=d0)
    b = torch.randn(512, 512, device=d0)

    out = pool.do(d0, add, a, b)
    assert out.device == d0
    torch.testing.assert_close(out, a + b)

    fut = pool.submit(d0, add, a, b)
    out2 = fut.result(timeout=5)
    torch.testing.assert_close(out2, a + b)


def test_linear_forward(pool, devices_two):
    d0 = devices_two[0]
    m = nn.Linear(128, 64).to(d0)
    x = torch.randn(32, 128, device=d0)

    def forward(module, inp):
        return module(inp)

    y = pool.do(d0, forward, m, x)
    assert y.shape == (32, 64) and y.device == d0


def test_tensor_manipulation_and_minmax(pool, devices_two):
    d0 = devices_two[0]
    t = torch.randn(2048, device=d0)

    def stats(u):
        return u.min().item(), u.max().item(), u.abs().mean().item()

    mn, mx, am = pool.do(d0, stats, t)
    assert isinstance(mn, float) and isinstance(mx, float) and isinstance(am, float)
    assert mn <= mx


def test_d2h_and_h2d_with_pinned_memory(pool, devices_two):
    d0 = devices_two[0]
    n = 1 << 19
    src = torch.randn(n, device=d0, dtype=torch.float32)
    host = torch.empty(n, dtype=torch.float32, pin_memory=True)

    def d2h(a, h):
        h.copy_(a, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return float(h[:10].sum().item())

    s = pool.do(d0, d2h, src, host)
    assert isinstance(s, float)

    def h2d(h):
        b = torch.empty_like(src, device=d0)
        b.copy_(h, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return b

    dst = pool.do(d0, h2d, host)
    torch.testing.assert_close(dst, host.to(device=d0))


def test_p2p_copy_between_devices(pool, devices_two):
    d0, d1 = devices_two
    n = 1 << 18
    a = torch.randn(n, device=d0, dtype=torch.float16)

    def p2p(x, target_dev):
        return x.to(target_dev, non_blocking=True)

    b = pool.do(d0, p2p, a, d1)
    assert b.device == d1
    torch.testing.assert_close(b.to(d0), a, atol=1e-3, rtol=1e-3)


def test_cuda_event_wait_is_honored(pool, devices_two):
    d0 = devices_two[0]
    flag = torch.zeros(1, device=d0, dtype=torch.int32)
    stream = torch.cuda.Stream(device=d0)
    event = torch.cuda.Event(enable_timing=False, blocking=False)

    with torch.cuda.stream(stream):
        _sleep_kernel_ms(10)
        flag.fill_(1)
        event.record()

    def read_flag(tensor):
        return int(tensor.item())

    ok = pool.do(d0, read_flag, flag, cuda_event=event)
    assert ok == 1


def test_parallel_submissions_from_many_threads(pool, devices_two):
    d0, d1 = devices_two

    def work_scale_add(scale, x, y):
        return (x * scale) + y

    xs = [torch.randn(1024, device=d0) for _ in range(8)]
    ys = [torch.randn(1024, device=d0) for _ in range(8)]
    zs = [torch.randn(1024, device=d1) for _ in range(8)]

    def submit_task(i):
        if i % 2 == 0:
            return pool.do(d0, work_scale_add, 1.5, xs[i // 2], ys[i // 2])
        else:
            return pool.do(d1, work_scale_add, 0.5, zs[i // 2], zs[i // 2])

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(submit_task, i) for i in range(8)]
        outs = [f.result(timeout=10) for f in futs]

    cnt0 = sum(1 for i in range(8) if i % 2 == 0)
    cnt1 = 8 - cnt0
    assert len(outs) == 8
    assert sum(o.device == d0 for o in outs) == cnt0
    assert sum(o.device == d1 for o in outs) == cnt1


def test_submit_serial_recovers_after_worker_exit(pool, devices_two):
    d0 = devices_two[0]
    key = pool._key(d0)

    def echo(v):
        return v

    # Ensure the serial worker exists and is healthy.
    assert pool.submit_serial(d0, echo, 1).result(timeout=2) == 1
    pool.wait(d0)

    with pool._dispatch_lock:
        worker = pool._serial_workers[key]

    # Simulate an unexpected worker shutdown.
    worker.stop()
    worker.join()

    # Submitting serial work should spawn a fresh worker and run to completion.
    futs = [pool.submit_serial(d0, echo, i) for i in range(3)]
    outs = [f.result(timeout=5) for f in futs]
    assert outs == [0, 1, 2]

    with pool._dispatch_lock:
        new_worker = pool._serial_workers.get(key)

    assert new_worker is not None
    assert new_worker is not worker


def test_device_lock_blocks_only_that_device(pool, devices_two):
    d0, d1 = devices_two

    started0 = threading.Event()
    finished0 = threading.Event()
    started1 = threading.Event()
    finished1 = threading.Event()

    def long_op(mark_start: threading.Event, mark_done: threading.Event, ms=150):
        mark_start.set()
        _sleep_kernel_ms(ms)
        mark_done.set()
        return ms

    with pool.device_lock(d0):
        fut0 = pool.submit(d0, long_op, started0, finished0, 100)
        fut1 = pool.submit(d1, long_op, started1, finished1, 50)

        started1.wait(timeout=2)
        assert started1.is_set()
        finished1.wait(timeout=2)
        assert finished1.is_set()
        assert not started0.is_set()

    assert fut0.result(timeout=2) == 100
    assert started0.is_set() and finished0.is_set()
    assert fut1.result(timeout=0.5) == 50


def test_global_lock_blocks_all_devices(pool, devices_two):
    d0, d1 = devices_two

    started = [threading.Event(), threading.Event()]

    def long_op(i, ms=100):
        started[i].set()
        _sleep_kernel_ms(ms)
        return ms

    with pool.lock():
        f0 = pool.submit(d0, long_op, 0, 50)
        f1 = pool.submit(d1, long_op, 1, 50)
        time.sleep(0.05)
        assert not started[0].is_set()
        assert not started[1].is_set()

    assert f0.result(timeout=2) == 50
    assert f1.result(timeout=2) == 50
    assert started[0].is_set() and started[1].is_set()


# ---------------------- Counters + Janitor Tests ---------------------

def test_counters_increment_and_global_sum(pool, devices_two):
    d0, d1 = devices_two

    def noop():
        return 1

    # run several tasks across both devices
    for _ in range(4):
        pool.do(d0, noop)
    for _ in range(5):
        pool.do(d1, noop)

    stats = pool.stats()
    per = stats["per_device"]
    total = stats["total"]
    assert per[f"cuda:{d0.index}"] == 4
    assert per[f"cuda:{d1.index}"] == 5
    assert total == 9


def test_janitor_triggers_empty_cache_every_n(pool, devices_two, monkeypatch):
    """
    Set threshold small (3). After each device completes 3 tasks, we expect the janitor
    to acquire the global lock and call empty_cache once per device.
    We monkeypatch torch.cuda.empty_cache to record invocations and delay slightly so
    we can observe the janitor pass.
    """
    d0, d1 = devices_two

    calls = []
    in_gc = threading.Event()

    orig_empty = torch.cuda.empty_cache

    def spy_empty_cache():
        in_gc.set()
        # record current device id
        cur = torch.cuda.current_device()
        calls.append(cur)
        # small delay so we can see blocking effects if any
        time.sleep(0.03)
        return

    monkeypatch.setattr(torch.cuda, "empty_cache", spy_empty_cache)

    def noop():
        return 1

    # Threshold is 3 => after 3 completions on d0, GC pass runs (both devices visited)
    for _ in range(3):
        pool.do(d0, noop)

    # Wait for janitor to start
    in_gc.wait(timeout=2)
    assert in_gc.is_set()

    # Give janitor time to run both device empties
    t0 = time.time()
    while len(calls) < 2 and time.time() - t0 < 2.0:
        time.sleep(0.01)

    # We should see two calls, one per device (order not guaranteed)
    assert len(calls) >= 2
    assert sorted(set(calls)) == sorted({d0.index, d1.index})

    # Restore
    monkeypatch.setattr(torch.cuda, "empty_cache", orig_empty)


def test_janitor_resets_device_watermark(pool, devices_two, monkeypatch):
    """
    Ensure devices that only partially progressed before a GC pass still trigger
    the next pass after completing their own threshold of work.
    """
    d0, d1 = devices_two

    calls: list[int] = []
    lock = threading.Lock()
    first_pass = threading.Event()
    second_pass = threading.Event()

    orig_empty = torch.cuda.empty_cache

    def spy_empty_cache():
        cur = torch.cuda.current_device()
        with lock:
            calls.append(cur)
            if len(calls) >= 2:
                first_pass.set()
            if len(calls) >= 4:
                second_pass.set()
        time.sleep(0.02)

    monkeypatch.setattr(torch.cuda, "empty_cache", spy_empty_cache)

    def noop():
        return 1

    # Device 1 records a single completion before the first GC pass.
    pool.do(d1, noop)

    # Device 0 hits the threshold and triggers the first GC.
    for _ in range(3):
        pool.do(d0, noop)

    assert first_pass.wait(timeout=2.0)

    # Device 1 finishes two more tasks (total=3) and should trigger another GC.
    for _ in range(2):
        pool.do(d1, noop)

    assert second_pass.wait(timeout=2.0)

    assert len(calls) >= 4

    pool.wait('all')

    # Restore immediately so later tests see the original implementation.
    monkeypatch.setattr(torch.cuda, "empty_cache", orig_empty)

######## test_threadx_janitor.py ########
class _DummyLock:
    @contextlib.contextmanager
    def writer(self):
        yield

class TestThreadxJanitor():

    DeviceThreadPool = threadx_mod.DeviceThreadPool

    def _make_pool(self):
        pool = DeviceThreadPool.__new__(DeviceThreadPool)
        pool._gc_event = threading.Event()
        pool._stop_event = threading.Event()
        pool._auto_gc_disable_cv = threading.Condition()
        pool._auto_gc_disable_count = 0
        pool._gc_debounce_s = 0.0
        pool._gc_min_interval_s = 0.0
        pool._stats_lock = threading.Lock()
        pool._per_device_done = {}
        pool._total_done = 0
        pool._empty_cache_every_n = 3
        pool._devices_by_key = {}
        pool._locks = {}
        pool._ordered_keys = []
        pool._worker_groups = {}
        pool._inflight = {}
        pool._inflight_cv = {}
        pool._last_gc_done_per_device = {}
        pool._gc_passes = 0
        pool._last_gc_ts = None
        pool._gc_generation = 0
        pool._last_consumed_gc_generation = 0
        pool._synchronize_all = lambda: None
        pool._virtual_to_parent = {}
        pool._family_keys = {}
        pool._dispatch_lock = threading.Lock()
        pool._warmup_lock = threading.Lock()
        pool._warmup_ran_keys = set()
        pool._worker_warmups = {}
        pool._serial_workers = {}
        pool._ordered_keys = []
        # Bind instance methods that rely on self
        pool._collect_state_snapshot = DeviceThreadPool._collect_state_snapshot.__get__(pool, DeviceThreadPool)
        pool._should_run_gc_from_snapshot = DeviceThreadPool._should_run_gc_from_snapshot.__get__(pool, DeviceThreadPool)
        pool._update_gc_watermarks = DeviceThreadPool._update_gc_watermarks.__get__(pool, DeviceThreadPool)
        pool._mark_finished = DeviceThreadPool._mark_finished.__get__(pool, DeviceThreadPool)
        pool._on_task_finished = DeviceThreadPool._on_task_finished.__get__(pool, DeviceThreadPool)
        return pool


    @pytest.mark.parametrize("threshold_triggers", [3])
    def test_janitor_coalesces_pending_triggers(self, monkeypatch, threshold_triggers):
        pool = self._make_pool()
        pool._empty_cache_every_n = threshold_triggers

        key = "cuda:0"
        dev = torch.device("cuda", 0)
        pool._devices_by_key[key] = dev
        pool._locks[key] = _DummyLock()
        pool._ordered_keys = [key]
        pool._worker_groups[key] = []
        pool._inflight[key] = 0
        pool._inflight_cv[key] = threading.Condition()
        pool._last_gc_done_per_device[key] = 0
        pool._per_device_done[key] = 0

        calls = {"count": 0}

        def fake_empty_cache():
            calls["count"] += 1

        monkeypatch.setattr(threadx_mod.torch.cuda, "empty_cache", fake_empty_cache, raising=False)
        monkeypatch.setattr(threadx_mod, "TORCH_CUDA_EMPTY_CACHE", fake_empty_cache, raising=False)

        @contextlib.contextmanager
        def fake_cuda_device(index):
            yield

        monkeypatch.setattr(threadx_mod.torch.cuda, "device", fake_cuda_device, raising=False)

        # Simulate multiple threshold triggers before janitor runs.
        for _ in range(threshold_triggers * 3):
            pool._inflight[key] = pool._inflight.get(key, 0) + 1
            pool._on_task_finished(key)

        assert pool._gc_generation == 3
        assert pool._gc_event.is_set()

        janitor = threading.Thread(target=pool._janitor_loop, daemon=True)
        janitor.start()

        start = time.time()
        while calls["count"] < 1 and time.time() - start < 1.0:
            time.sleep(0.01)

        # Allow janitor time to spin in case extra passes would occur.
        time.sleep(0.1)

        pool._stop_event.set()
        pool._gc_event.set()
        janitor.join(timeout=1.0)

        assert calls["count"] == 1
        assert pool._gc_passes == 1
        assert pool._last_consumed_gc_generation == pool._gc_generation

######## test_threadx_mps.py #########

mps_available = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

@pytest.mark.mps
@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_mps_worker_basic():
    d_mps = torch.device("mps")
    p = DeviceThreadPool(devices=[d_mps], inference_mode=True, empty_cache_every_n=3)
    try:
        def add(a, b): return a + b
        a = torch.randn(256, 256, device=d_mps)
        b = torch.randn(256, 256, device=d_mps)
        out = p.do(d_mps, add, a, b)
        assert out.device.type == "mps"
        torch.testing.assert_close(out, a + b)
    finally:
        p.shutdown()

@pytest.mark.mps
@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_mps_linear_forward_and_counters(monkeypatch):
    d_mps = torch.device("mps")
    calls = []

    # Spy on torch.mps.empty_cache to confirm janitor invocation
    if hasattr(torch, "mps"):
        orig = torch.mps.empty_cache
        def spy():
            calls.append("ec")
            # tiny delay to ensure pass runs
            time.sleep(0.01)
        monkeypatch.setattr(torch.mps, "empty_cache", spy)

    p = DeviceThreadPool(devices=[d_mps], inference_mode=True, empty_cache_every_n=2)
    try:
        m = nn.Linear(64, 32).to(d_mps)
        x = torch.randn(16, 64, device=d_mps)

        def fwd(mod, inp): return mod(inp)

        # two tasks -> threshold 2 -> janitor should run once
        p.do(d_mps, fwd, m, x)
        y = p.do(d_mps, fwd, m, x)
        assert y.shape == (16, 32)

        # allow janitor to run
        time.sleep(0.1)

        st = p.stats()
        assert st["per_device"]["mps"] >= 2
        assert st["total"] >= 2
        if hasattr(torch, "mps"):
            assert len(calls) >= 1
    finally:
        p.shutdown()
        if hasattr(torch, "mps"):
            monkeypatch.setattr(torch.mps, "empty_cache", orig)

######### test_threadx_virtual.py ###########

@pytest.fixture()
def virtual_pool():
    pool = DeviceThreadPool(
        devices=[torch.device("cpu")],
        inference_mode=False,
        workers={
            "cpu": 4,
            "turtle:cpu": 2,
        },
        empty_cache_every_n=0,
    )
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def test_virtual_pool_respects_concurrency_limit(virtual_pool):
    running = 0
    max_seen = 0
    lock = threading.Lock()

    def busy(delay: float):
        nonlocal running, max_seen
        with lock:
            running += 1
            if running > max_seen:
                max_seen = running
        time.sleep(delay)
        with lock:
            running -= 1
        return delay

    delays = [0.05] * 6
    futs = [virtual_pool.submit("turtle:cpu", busy, d) for d in delays]
    results = [f.result(timeout=2) for f in futs]

    assert results == delays
    assert 1 <= max_seen <= 2

    stats = virtual_pool.stats()["per_device"]
    assert stats.get("turtle:cpu") == len(delays)
    assert stats.get("cpu", 0) == 0


def test_virtual_pool_blocks_under_parent_lock(virtual_pool):
    started = threading.Event()

    def marker(evt: threading.Event):
        evt.set()
        return True

    with virtual_pool.device_lock("cpu"):
        fut = virtual_pool.submit("turtle:cpu", marker, started)
        time.sleep(0.05)
        assert not started.is_set()

    assert fut.result(timeout=1) is True
    assert started.wait(timeout=1)


def test_virtual_pool_wait_and_stats_isolated(virtual_pool):
    done_flags = [threading.Event() for _ in range(3)]

    def do_work(flag: threading.Event):
        flag.set()
        time.sleep(0.02)
        return 1

    futs = [virtual_pool.submit("turtle:cpu", do_work, flag) for flag in done_flags]
    virtual_pool.wait("turtle:cpu")
    assert all(flag.is_set() for flag in done_flags)
    assert [f.result(timeout=1) for f in futs] == [1, 1, 1]

    stats = virtual_pool.stats()["per_device"]
    assert stats.get("turtle:cpu") >= 3
    assert stats.get("cpu", 0) == 0


def test_virtual_pool_raises_when_exceeding_parent_capacity():
    with pytest.raises(ValueError):
        DeviceThreadPool(
            devices=[torch.device("cpu")],
            workers={
                "cpu": 2,
                "hare:cpu": 3,
            },
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_virtual_pool_cuda_parent_requires_index():
    with pytest.raises(ValueError):
        DeviceThreadPool(
            devices=[torch.device("cuda")],
            workers={
                "cuda": 2,
                "gryphon:cuda": 1,
            },
            empty_cache_every_n=0,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_virtual_pool_cuda_alias_behaviour():
    dev = torch.device("cuda", 0)
    pool = DeviceThreadPool(
        devices=[dev],
        inference_mode=True,
        workers={
            "cuda:0": 2,
            "gryphon:cuda:0": 1,
        },
        empty_cache_every_n=0,
    )

    running = 0
    max_seen = 0
    lock = threading.Lock()

    def busy(delay: int):
        nonlocal running, max_seen
        with lock:
            running += 1
            max_seen = max(max_seen, running)
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(int(delay) * 1_000_000)
        else:
            time.sleep(delay / 1000.0)
        with lock:
            running -= 1
        return delay

    try:
        delays = [50, 60, 70]
        futs = [pool.submit("gryphon:cuda:0", busy, d) for d in delays]
        results = [f.result(timeout=5) for f in futs]
        assert results == delays
        assert max_seen == 1

        start_evt = threading.Event()

        def marker(evt: threading.Event):
            evt.set()
            return True

        with pool.device_lock("cuda:0"):
            fut = pool.submit("gryphon:cuda:0", marker, start_evt)
            time.sleep(0.05)
            assert not start_evt.is_set()

        assert fut.result(timeout=2) is True
        assert start_evt.wait(timeout=2)

        stats = pool.stats()["per_device"]
        assert stats.get("gryphon:cuda:0") == len(delays) + 1
        assert stats.get("cuda:0", 0) == 0
    finally:
        pool.shutdown(wait=True)

######## test_threadx_wait.py #########

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

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

