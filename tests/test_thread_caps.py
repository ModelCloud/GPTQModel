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
    # Force the OpenMP region to materialize before counting.
    matrix = torch.randn(256, 256)
    matrix @ matrix

    cpu_count = os.cpu_count() or 1
    task_count = len(os.listdir("/proc/self/task"))

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
