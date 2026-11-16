# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The 3.14t unit-test workflow must prove it is really running free-threaded.

`unit_tests.yml` selects Python `3.14t` but used to leave `PYTHON_GIL: 0`
commented out, and `unit_tests_reusable.yml` neither set nor asserted it. A
3.14t job can therefore re-enable the GIL while importing
`triton._C.libtriton` and run fully serialized while every GIL=0-only test
skips. These tests pin the dedicated lane: it forces the interpreter, asserts
the GIL is still disabled after the model stack is imported, and runs bounded
concurrency tests on CPU and on two GPUs.
"""

import json
import os
import subprocess
import sys
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
    assert lane["with"]["free_threaded"] is True
    assert lane["with"]["matrix_json"]

    # The default matrix must not pretend to be free-threaded; only the lane
    # forces it, and it does so through the reusable workflow.
    assert "PYTHON_GIL" not in (workflow.get("env") or {})


def test_reusable_workflow_exports_gil0_before_asserting_and_testing():
    reusable = _load(REUSABLE)
    inputs = reusable[True]["workflow_call"]["inputs"]
    assert inputs["free_threaded"]["type"] == "boolean"
    assert inputs["free_threaded"]["default"] is False

    steps = _job_steps(reusable)
    export_index = _step_index(steps, "Force the free-threaded interpreter")
    assert_index = _step_index(steps, "Assert the GIL stays disabled after importing the model stack")
    run_index = _step_index(steps, "Run tests")

    assert export_index < assert_index < run_index
    export_step = steps[export_index]
    assert export_step["if"] == "${{ inputs.free_threaded }}"
    assert "PYTHON_GIL=0" in export_step["run"]
    assert "$GITHUB_ENV" in export_step["run"]

    assert_step = steps[assert_index]
    assert assert_step["if"] == "${{ inputs.free_threaded }}"
    assert "ci_assert_free_threaded.py" in assert_step["run"]


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


def _run_gate(gil):
    env = dict(os.environ)
    env["PYTHON_GIL"] = gil
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, str(GATE)],
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
    if hasattr(sys, "_is_gil_enabled"):
        assert "the GIL is enabled before the model stack is imported" in output
    else:
        assert "not a free-threaded build" in output


@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled"),
    reason="requires a free-threaded interpreter to prove the gate can pass",
)
def test_gate_passes_with_gil0():
    result = _run_gate("0")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "GIL stays disabled after importing the model stack" in output
