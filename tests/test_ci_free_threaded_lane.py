# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The 3.14t unit-test workflow must prove it is really running free-threaded.

The CUDA jobs force the GIL off and assert the state after importing the model
stack. The dedicated lane runs bounded concurrency tests on CPU and on two GPUs.
"""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/unit_tests.yml"
REUSABLE = REPO_ROOT / ".github/workflows/unit_tests_reusable.yml"
GATE = REPO_ROOT / ".github/scripts/ci_assert_free_threaded.py"

# The lane must keep covering the bounded concurrency paths, not just any test.
REQUIRED_CPU_TESTS = {
    "test_threadx_cpu_locks",
    "test_threadx_shutdown",
    "test_looper_concurrency",
    "test_checkpoint_store",
}
REQUIRED_GPU_TESTS = {"test_stream", "test_thread_pool_exit"}
REQUIRED_MULTI_GPU_TESTS = {"test_p2p", "test_p2p_copy_utils"}


def _load(path):
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _job_steps(workflow):
    return workflow["jobs"]["test"]["steps"]


def _step_index(steps, name):
    for index, step in enumerate(steps):
        if step.get("name") == name:
            return index
    raise AssertionError(f"missing workflow step: {name}")


def test_free_threaded_lane_exists_and_forces_the_interpreter():
    workflow = _load(WORKFLOW)
    jobs = workflow["jobs"]
    assert "free-threaded" in jobs, "unit_tests.yml has no dedicated free-threaded lane"

    lane = jobs["free-threaded"]
    assert lane["uses"] == "./.github/workflows/unit_tests_reusable.yml"
    assert lane["with"]["matrix_json"]

    assert workflow["env"]["PYTHON_GIL"] == "0"


def test_reusable_workflow_forces_gil0_and_checks_the_installed_stack():
    reusable = _load(REUSABLE)
    assert reusable["env"]["PYTHON_GIL"] == "0"
    steps = _job_steps(reusable)
    setup_index = _step_index(steps, "Set up CI runtime")
    install_index = _step_index(steps, "Install package from source")
    assert_index = _step_index(steps, "Assert the GIL stays disabled after importing the model stack")
    run_index = _step_index(steps, "Run tests")
    assert setup_index < install_index < assert_index < run_index
    assert steps[setup_index]["with"]["python-version"] == "3.14.10t"
    assert "if" not in steps[assert_index]
    assert "-X gil=0 .github/scripts/ci_assert_free_threaded.py --installed" in steps[assert_index]["run"]


def test_lane_runs_bounded_cpu_and_multi_gpu_concurrency_tests():
    workflow = _load(WORKFLOW)
    matrix = json.loads(workflow["jobs"]["free-threaded"]["with"]["matrix_json"])
    by_script = {entry["test_script"]: entry for entry in matrix}

    for script in REQUIRED_CPU_TESTS | REQUIRED_GPU_TESTS | REQUIRED_MULTI_GPU_TESTS:
        assert script in by_script, f"free-threaded lane does not run {script}"
        assert (REPO_ROOT / "tests" / f"{script}.py").is_file()
        assert set(by_script[script]) == {
            "test_script",
            "test_group",
            "alloc_gpu_count",
            "require_single_gpu",
            "include_model_test_mode",
        }

    for script in REQUIRED_CPU_TESTS:
        assert by_script[script]["alloc_gpu_count"] == "0"

    for script in REQUIRED_MULTI_GPU_TESTS:
        assert by_script[script]["alloc_gpu_count"] == "2"
        assert by_script[script]["require_single_gpu"] == "false"


def _free_threaded_build() -> bool:
    """True only for a free-threaded (3.13t/3.14t) interpreter.

    ``hasattr(sys, "_is_gil_enabled")`` is true on every CPython >= 3.13,
    including GIL builds, so it cannot gate a ``PYTHON_GIL=0`` subprocess: a
    regular build aborts with "Disabling the GIL is not supported by this
    build" instead of running the gate (#4768).
    """

    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def _run_gate(gil, *args):
    env = dict(os.environ)
    env["PYTHON_GIL"] = gil
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_gate_fails_when_the_gil_is_enabled():
    result = _run_gate("1")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    if _free_threaded_build():
        assert "the GIL is enabled before the model stack is imported" in output
    else:
        assert "not a free-threaded build" in output


@pytest.mark.skipif(
    not (_free_threaded_build() and sys.version_info[:3] == (3, 14, 10)),
    reason="requires the pinned free-threaded CI interpreter",
)
def test_gate_verifies_gil0_without_importing_the_model_stack():
    result = _run_gate("0", "--python-only")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "starts with the GIL disabled" in output


@pytest.mark.skipif(
    not (_free_threaded_build() and sys.version_info[:3] == (3, 14, 10)),
    reason="requires the pinned free-threaded CI interpreter",
)
def test_gate_passes_with_gil0():
    result = _run_gate("0")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "GIL stays disabled after importing the model stack" in output
