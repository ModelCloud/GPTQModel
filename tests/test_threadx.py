# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn as nn

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
