# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Dict, List

import torch

from gptqmodel.utils.safe import THREADPOOLCTL
from gptqmodel.utils.threadx import DeviceThreadPool


def _run_thread_limit(pool: DeviceThreadPool, limit: int) -> Dict[str, float]:
    d_cpu = torch.device("cpu")
    futures = []

    def worker():
        with THREADPOOLCTL.threadpool_limits(limits=limit):
            start = time.perf_counter()
            info = THREADPOOLCTL.threadpool_info()
            counts = [entry.get("num_threads", 0) for entry in info if entry.get("num_threads", 0) > 0]
            # Exercise BLAS path
            a = torch.randn(512, 256, device=d_cpu)
            b = torch.randn(256, 512, device=d_cpu)
            _ = a @ b
            elapsed = time.perf_counter() - start
            max_threads = max(counts) if counts else 0
            return elapsed, max_threads

    for _ in range(8):
        futures.append(pool.submit(d_cpu, worker))

    pool.wait(d_cpu)

    timings = []
    thread_counts = []
    for fut in futures:
        elapsed, max_threads = fut.result(timeout=5)
        timings.append(elapsed)
        thread_counts.append(max_threads)

    mean_time = sum(timings) / len(timings)
    return {
        "mean_time": mean_time,
        "thread_counts": thread_counts,
    }


def test_threadpool_limits_inside_device_threadpool():
    d_cpu = torch.device("cpu")
    pool = DeviceThreadPool(
        devices=[d_cpu],
        include_cuda=False,
        include_xpu=False,
        include_mps=False,
        include_cpu=True,
        workers={"cpu": 8},
        inference_mode=True,
    )

    try:
        limits = [1, 2, 4, 8, 16, 32]
        results: List[Dict[str, float]] = []

        for limit in limits:
            result = _run_thread_limit(pool, limit)
            results.append(result)
            for count in result["thread_counts"]:
                if count:
                    assert count <= limit
        for limit, result in zip(limits, results):
            print(
                f"[thread limit={limit}] mean worker time: {result['mean_time'] * 1e3:.3f} ms "
                f"| thread counts: {result['thread_counts']}"
            )
    finally:
        pool.shutdown(wait=True)

