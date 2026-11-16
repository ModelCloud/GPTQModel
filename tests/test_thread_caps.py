# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The pytest harness caps Torch's intra-op pool regardless of host CPU count.

The host can advertise hundreds of CPUs. Without the cap in ``conftest.py``,
Torch sizes its pool from that advertised count and a single test process
spins up a thread per CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _expected_cap() -> int:
    default = min(8, os.cpu_count() or 1)
    try:
        return int(os.environ.get("OMP_NUM_THREADS", default))
    except ValueError:
        return default


def _run_python(
    code: str, env_overrides: dict[str, str], args: tuple[str, ...] = ()
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_REPO_ROOT), env.get("PYTHONPATH")) if part
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=True,
    )


def test_pytest_process_uses_capped_intraop_pool():
    cap = _expected_cap()

    assert cap <= 8
    assert torch.get_num_threads() == cap
    # conftest.py applied the same value to the environment before importing
    # torch, so the explicit environment path and the torch path agree.
    assert int(os.environ["OMP_NUM_THREADS"]) == cap


def test_process_thread_count_does_not_scale_with_cpu_count():
    """A fresh pytest process must not spawn a thread per advertised CPU.

    Measured in a subprocess on purpose. Once any test has run ``quantize()``
    the parent process owns ``DeviceThreadPool`` workers (each with its own
    OpenMP team) and the tokenizer's rayon pool, so a process-wide count here
    depends on test order: 399 threads against the 80 limit after
    ``test_paro_branch_replay_masks``, while this file passes 5/5 alone. The
    contract under test is that a process which loads ``conftest.py`` and enters
    a parallel region stays bounded by the cap instead of by ``os.cpu_count()``.
    """
    code = (
        "import os, runpy, sys\n"
        "runpy.run_path(sys.argv[1])\n"
        "import torch\n"
        "matrix = torch.randn(256, 256)\n"
        "matrix @ matrix\n"
        "print(len(os.listdir('/proc/self/task')))\n"
        "print(os.cpu_count())\n"
        "print(torch.get_num_threads())\n"
    )
    result = _run_python(code, {}, (str(_REPO_ROOT / "tests" / "conftest.py"),))

    task_count, cpu_count, pool = (
        int(value) for value in result.stdout.strip().splitlines()[-3:]
    )
    assert pool == _expected_cap()
    assert task_count <= max(32, cpu_count // 4), (task_count, cpu_count)


def test_explicit_omp_num_threads_is_honored_by_conftest():
    # Load conftest.py the way pytest does (module execution) and inspect the
    # resulting torch pool. The value 3 is deliberately unrelated to the
    # default so this test cannot pass by coincidence.
    code = (
        "import runpy, sys\n"
        "runpy.run_path(sys.argv[1])\n"
        "import torch\n"
        "print(torch.get_num_threads())\n"
    )
    result = _run_python(
        code,
        {"OMP_NUM_THREADS": "3"},
        (str(_REPO_ROOT / "tests" / "conftest.py"),),
    )
    assert result.stdout.strip().splitlines()[-1] == "3"


def test_helper_honors_explicit_omp_num_threads():
    code = (
        "import os\n"
        "import torch\n"
        "from gptqmodel.utils.threads import cap_torch_threads\n"
        "print(cap_torch_threads())\n"
        "print(torch.get_num_threads())\n"
        "print(os.environ['OMP_NUM_THREADS'])\n"
    )
    result = _run_python(code, {"OMP_NUM_THREADS": "5"})

    assert result.stdout.strip().splitlines()[-3:] == ["5", "5", "5"]


def test_helper_explicit_argument_wins_without_overwriting_environment():
    code = (
        "import os\n"
        "import torch\n"
        "from gptqmodel.utils.threads import cap_torch_threads\n"
        "print(cap_torch_threads(2))\n"
        "print(torch.get_num_threads())\n"
        "print(os.environ['OMP_NUM_THREADS'])\n"
    )
    result = _run_python(code, {"OMP_NUM_THREADS": "5"})

    assert result.stdout.strip().splitlines()[-3:] == ["2", "2", "5"]


# The device pool multiplies Torch's intra-op pool by its worker count unless
# every worker is told its own budget: 12 CPU workers on a 320-CPU host held
# 2538 OS threads (#4356). These tests pin the pool contract, not the host's
# CPU count.


def test_pool_workers_get_split_intraop_budget():
    from gptqmodel.utils.threadx import split_intraop_threads

    assert split_intraop_threads(1, 8) == 8
    assert split_intraop_threads(8, 8) == 1
    # More workers than the budget still yields a usable team per worker.
    assert split_intraop_threads(64, 8) == 1
    assert split_intraop_threads(3, 8) == 2


def test_device_thread_pool_workers_cap_their_own_intraop_team():
    code = (
        "import torch\n"
        "from gptqmodel.utils.threadx import DeviceThreadPool\n"
        "torch.set_num_threads(8)\n"
        "pool = DeviceThreadPool(\n"
        "    devices=[torch.device('cpu')], include_cuda=False, include_xpu=False,\n"
        "    include_npu=False, include_mps=False, workers={'cpu': 4},\n"
        "    empty_cache_every_n=0,\n"
        ")\n"
        "seen = []\n"
        "futures = [\n"
        "    pool.submit(torch.device('cpu'), lambda: seen.append(torch.get_num_threads()))\n"
        "    for _ in range(8)\n"
        "]\n"
        "[future.result(timeout=60) for future in futures]\n"
        "pool.shutdown(wait=True)\n"
        "print(f'RESULT workers {min(seen)} {max(seen)}')\n"
        "print(f'RESULT caller {torch.get_num_threads()}')\n"
    )
    result = _run_python(code, {})
    values = {
        line.split()[1]: tuple(int(part) for part in line.split()[2:])
        for line in result.stdout.splitlines()
        if line.startswith("RESULT ")
    }

    # 8 threads split over 4 workers.
    assert values["workers"] == (2, 2)
    # The worker's thread-local change must not touch the caller's pool.
    assert values["caller"] == (8,)


def test_device_thread_pool_explicit_worker_intraop_override():
    code = (
        "import torch\n"
        "from gptqmodel.utils.threadx import DeviceThreadPool\n"
        "torch.set_num_threads(8)\n"
        "pool = DeviceThreadPool(\n"
        "    devices=[torch.device('cpu')], include_cuda=False, include_xpu=False,\n"
        "    include_npu=False, include_mps=False, workers={'cpu': 2},\n"
        "    intraop_threads_per_worker=1, empty_cache_every_n=0,\n"
        ")\n"
        "seen = []\n"
        "future = pool.submit(torch.device('cpu'), lambda: seen.append(torch.get_num_threads()))\n"
        "future.result(timeout=60)\n"
        "pool.shutdown(wait=True)\n"
        "print(f'RESULT {seen[0]}')\n"
    )
    result = _run_python(code, {})

    assert "RESULT 1" in result.stdout.splitlines()


def test_kernel_config_pool_intraop_threads_drives_the_default_pool():
    from gptqmodel import _build_device_thread_pool
    from gptqmodel.utils.kernel import KernelConfig, configure_kernel, reset_kernel_config

    try:
        configure_kernel(KernelConfig(pool_intraop_threads=3))
        pool = _build_device_thread_pool(warmup_devices=[])
        try:
            assert pool._intraop_threads_per_worker == 3
        finally:
            pool.shutdown(wait=True)

        configure_kernel(KernelConfig(pool_intraop_threads=0))
        pool = _build_device_thread_pool(warmup_devices=[])
        try:
            assert pool._intraop_threads_per_worker is None
        finally:
            pool.shutdown(wait=True)
    finally:
        reset_kernel_config()


def test_device_thread_pool_can_keep_the_caller_intraop_pool():
    code = (
        "import torch\n"
        "from gptqmodel.utils.threadx import DeviceThreadPool\n"
        "torch.set_num_threads(3)\n"
        "pool = DeviceThreadPool(\n"
        "    devices=[torch.device('cpu')], include_cuda=False, include_xpu=False,\n"
        "    include_npu=False, include_mps=False, workers={'cpu': 2},\n"
        "    intraop_threads_per_worker=0, empty_cache_every_n=0,\n"
        ")\n"
        "seen = []\n"
        "future = pool.submit(torch.device('cpu'), lambda: seen.append(torch.get_num_threads()))\n"
        "future.result(timeout=60)\n"
        "pool.shutdown(wait=True)\n"
        "print(f'RESULT {seen[0]}')\n"
    )
    result = _run_python(code, {})

    assert "RESULT 3" in result.stdout.splitlines()
