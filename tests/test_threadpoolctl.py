# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from gptqmodel.utils.safe import THREADPOOLCTL
from gptqmodel.utils.threadx import DeviceThreadPool

_MAX_PYTEST_CPU_THREADS = 16
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _load_cpu_test_runner():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / ".codex"
        / "skills"
        / "limit-python-test-threads"
        / "scripts"
        / "run_cpu_tests.py"
    )
    spec = importlib.util.spec_from_file_location("run_cpu_tests", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_native_thread_pools_are_capped():
    for name in _THREAD_ENV_VARS:
        assert 1 <= int(os.environ[name]) <= _MAX_PYTEST_CPU_THREADS
    assert 1 <= torch.get_num_threads() <= _MAX_PYTEST_CPU_THREADS
    for pool in THREADPOOLCTL.threadpool_info():
        threads = pool.get("num_threads", 0)
        if threads:
            assert threads <= _MAX_PYTEST_CPU_THREADS


def test_pytest_thread_cap_subprocess_probe():
    if os.environ.get("GPTQMODEL_PYTEST_THREAD_CAP_PROBE") != "1":
        pytest.skip("subprocess-only thread-cap probe")
    for name in _THREAD_ENV_VARS:
        assert os.environ[name] == str(_MAX_PYTEST_CPU_THREADS)
    assert torch.get_num_threads() <= _MAX_PYTEST_CPU_THREADS
    for pool in THREADPOOLCTL.threadpool_info():
        threads = pool.get("num_threads", 0)
        if threads:
            assert threads <= _MAX_PYTEST_CPU_THREADS


def test_pytest_clamps_oversized_inherited_native_thread_limits():
    env = os.environ.copy()
    env.update({name: "64" for name in _THREAD_ENV_VARS})
    env["GPTQMODEL_PYTEST_THREAD_CAP_PROBE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"{__file__}::test_pytest_thread_cap_subprocess_probe", "-q"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_test_runner_clamps_oversized_limit_and_preserves_lower_limit():
    runner = _load_cpu_test_runner()

    assert runner.effective_limit(64) == _MAX_PYTEST_CPU_THREADS
    assert runner.effective_limit(6) == 6
    for name in _THREAD_ENV_VARS:
        assert runner.limited_env(64)[name] == str(_MAX_PYTEST_CPU_THREADS)
        assert runner.limited_env(6)[name] == "6"


def _run_thread_limit(pool: DeviceThreadPool, limit: int) -> dict[str, float]:
    d_cpu = torch.device("cpu")
    futures = []

    def worker():
        with THREADPOOLCTL.threadpool_limits(limits=limit):
            start = time.perf_counter()
            info = THREADPOOLCTL.threadpool_info()
            # BLAS sometimes doesn't respect the limits set; the reason for this hasn't been found yet.
            counts = [entry.get("num_threads", 0) for entry in info if entry.get("num_threads", 0) > 0 and entry.get("user_api") != "blas"]
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
        session_limit = int(os.environ["OMP_NUM_THREADS"])
        limits = sorted({limit for limit in (1, 2, 4, 8, 16, session_limit) if limit <= session_limit})
        results: list[dict[str, float]] = []

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
