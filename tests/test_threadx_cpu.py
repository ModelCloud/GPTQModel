# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
import time

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

# ----------------------------- Existing GPU fixtures/tests omitted for brevity -----------------------------
# (Keep your previously provided tests as-is.)

# ----------------------------- New CPU-inclusive fixtures -----------------------------

@pytest.fixture(scope="module")
def devices_two_plus_cpu():
    require_n_gpus(2)
    return [torch.device("cuda", 0), torch.device("cuda", 1), torch.device("cpu")]

@pytest.fixture()
def pool_cpu(devices_two_plus_cpu):
    # small threshold so we can easily trip GC on GPUs; CPU won't trigger GC itself
    p = DeviceThreadPool(devices=devices_two_plus_cpu, inference_mode=True, empty_cache_every_n=3)
    yield p
    p.shutdown(wait=True)

# ----------------------------- CPU tests -----------------------------

def test_cpu_worker_basic(pool_cpu):
    d_cpu = torch.device("cpu")

    def add(a, b):
        return a + b

    a = torch.randn(256, 256, device=d_cpu)
    b = torch.randn(256, 256, device=d_cpu)
    out = pool_cpu.do(d_cpu, add, a, b)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, a + b)

def test_cpu_linear_forward(pool_cpu):
    d_cpu = torch.device("cpu")
    m = nn.Linear(128, 64)
    x = torch.randn(32, 128, device=d_cpu)

    def forward(module, inp):
        return module(inp)

    y = pool_cpu.do(d_cpu, forward, m, x)
    assert y.shape == (32, 64) and y.device.type == "cpu"

def test_cpu_device_lock_blocks_only_cpu(pool_cpu):
    d_cpu = torch.device("cpu")
    d0 = torch.device("cuda", 0)

    started_cpu = threading.Event()
    finished_cpu = threading.Event()
    started_gpu = threading.Event()
    finished_gpu = threading.Event()

    def long_cpu(mark_start, mark_done, ms=150):
        mark_start.set()
        time.sleep(ms / 1000.0)
        mark_done.set()
        return ms

    def long_gpu(mark_start, mark_done, ms=100):
        mark_start.set()
        # use a tiny CUDA sleep to do “real” device work
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(ms * 1000_000)
        else:
            time.sleep(ms / 1000.0)
        mark_done.set()
        return ms

    with pool_cpu.device_lock(d_cpu):
        f_cpu = pool_cpu.submit(d_cpu, long_cpu, started_cpu, finished_cpu, 100)
        f_gpu = pool_cpu.submit(d0, long_gpu, started_gpu, finished_gpu, 50)

        # GPU should run while CPU is blocked
        started_gpu.wait(timeout=2)
        assert started_gpu.is_set()
        finished_gpu.wait(timeout=2)
        assert finished_gpu.is_set()

        # CPU must not have started
        assert not started_cpu.is_set()

    # After release, CPU proceeds
    assert f_cpu.result(timeout=2) == 100
    assert started_cpu.is_set() and finished_cpu.is_set()
    assert f_gpu.result(timeout=0.5) == 50

def test_global_lock_includes_cpu(pool_cpu):
    d_cpu = torch.device("cpu")
    d0 = torch.device("cuda", 0)

    started = [threading.Event(), threading.Event()]  # [cpu, gpu]

    def long_cpu():
        started[0].set()
        time.sleep(0.05)
        return 1

    def long_gpu():
        started[1].set()
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(50 * 1000_000)
        else:
            time.sleep(0.05)
        return 1

    with pool_cpu.lock():
        f_cpu = pool_cpu.submit(d_cpu, long_cpu)
        f_gpu = pool_cpu.submit(d0, long_gpu)
        time.sleep(0.02)
        # Neither should start under global lock
        assert not started[0].is_set()
        assert not started[1].is_set()

    assert f_cpu.result(timeout=2) == 1
    assert f_gpu.result(timeout=2) == 1
    assert started[0].is_set() and started[1].is_set()

def test_counters_include_cpu(pool_cpu):
    d_cpu = torch.device("cpu")
    d0 = torch.device("cuda", 0)

    def noop():
        return 1

    # CPU tasks
    for _ in range(4):
        pool_cpu.do(d_cpu, noop)

    # GPU tasks
    for _ in range(2):
        pool_cpu.do(d0, noop)

    stats = pool_cpu.stats()
    per = stats["per_device"]
    assert per["cpu"] == 4
    assert per[f"cuda:{d0.index}"] >= 2  # at least the two we just ran
    assert stats["total"] >= 6
