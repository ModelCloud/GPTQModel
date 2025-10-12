# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
import time

import pytest
import torch

from gptqmodel.utils.threadx import DeviceThreadPool


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
