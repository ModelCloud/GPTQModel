# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Contract checks for the CI runtime and release publication boundary."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github"
WORKFLOWS = CI / "workflows"
CUDA_IMAGE = "nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf"
CI_WORKFLOW = CI / "scripts/ci_workflow.py"
GITHUB_MATRIX_ENTRY_LIMIT = 256
GITHUB_WORKFLOW_INPUT_LIMIT = 65_535
LINUX_MAX_ARG_STRLEN = 131_072


def _yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _release_flow(job):
    return next(step for step in job["steps"] if step.get("uses") == "./.github/actions/release-source-common")


def test_release_installs_the_sdist_on_both_runners_before_publication():
    workflow = _yaml(WORKFLOWS / "release.yml")
    assert workflow["env"]["PYTHON_GIL"] == "0"
    jobs = workflow["jobs"]
    for job_name in ("release-source", "release-source-github"):
        job = jobs[job_name]
        assert _release_flow(job)["with"].get("test-install") == "true"
        assert job.get("container", {}).get("image") == CUDA_IMAGE
        assert "GPTQ_BYPASS_SAFETY_CHECKS" not in job.get("env", {})
        setup = next(step for step in job["steps"] if step.get("uses") == "./.github/actions/setup-ci-runtime")
        assert setup["with"] == {"python-version": "3.14.10t", "source-build": "true"}
        assert _release_flow(job)["with"]["publish"] == "true"


def test_publication_requires_success_and_an_explicit_opt_in():
    action = _yaml(CI / "actions/release-source-common/action.yml")
    steps = {step["name"]: step for step in action["runs"]["steps"]}
    ready = "success() && steps.install.outcome == 'success' && inputs.publish == 'true'"
    release_requested = "(github.event_name == 'release' || github.event.inputs.upload_release == 'true')"
    pypi_requested = "(github.event_name == 'release' || github.event.inputs.upload_pypi == 'true')"
    confirmed = "(inputs.require-pypi-confirmation != 'true' || env.PYPI_RELEASE_CONFIRMATION == '1')"
    for name, condition in (
        ("Upload package to release", f"{ready} && {release_requested}"),
        ("Waiting for confirmation", f"{ready} && {pypi_requested}"),
        ("Upload sdist to pypi", f"{ready} && {pypi_requested} && {confirmed}"),
    ):
        assert steps[name]["if"] == "${{ " + condition + " }}"
    assert action["inputs"]["publish"]["default"] == "false"
    assert not steps["Test installation"].get("continue-on-error")


def test_unit_jobs_use_the_same_toolkit_and_keep_pinned_free_threaded_python():
    workflow = _yaml(WORKFLOWS / "unit_tests.yml")
    reusable = _yaml(WORKFLOWS / "unit_tests_reusable.yml")
    assert reusable["jobs"]["test"]["container"]["image"] == CUDA_IMAGE
    assert workflow["jobs"]["check-torch"]["container"]["image"] == CUDA_IMAGE
    assert _yaml(CI / "scripts/test.yaml")["common"]["py"] == "3.14.10t"
    assert "GPTQ_BYPASS_SAFETY_CHECKS" not in workflow["env"]
    assert "GPTQ_BYPASS_SAFETY_CHECKS" not in reusable["env"]


def test_unit_planning_and_setuptools_jobs_use_the_pinned_free_threaded_interpreter():
    unit = _yaml(WORKFLOWS / "unit_tests.yml")
    setuptools = _yaml(WORKFLOWS / "setuptools_compatibility_reusable.yml")
    for workflow, job_name, consumer in (
        (unit, "list-test-files", "List files"),
        (setuptools, "prepare-setuptools", "Generate version matrix"),
        (setuptools, "check-setuptools", "Install package with selected setuptools"),
    ):
        assert workflow["env"]["PYTHON_GIL"] == "0"
        steps = workflow["jobs"][job_name]["steps"]
        setup = next(i for i, step in enumerate(steps) if step.get("uses") == "./.github/actions/setup-ci-runtime")
        use = next(i for i, step in enumerate(steps) if step.get("name") == consumer)
        assert setup < use
        assert steps[setup]["with"]["python-version"] == "3.14.10t"


def test_validation_is_a_read_only_workflow_without_publish_credentials():
    validation = WORKFLOWS / "ci_runtime.yml"
    assert validation.is_file(), "the CUDA sdist smoke needs a standalone non-publishing workflow"
    workflow = _yaml(validation)
    assert set(workflow[True]) == {"pull_request", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["PYTHON_GIL"] == "0"
    job = workflow["jobs"]["validate-sdist"]
    assert job["container"]["image"] == CUDA_IMAGE
    flow = _release_flow(job)
    assert flow["with"]["test-install"] == "true"
    assert flow["with"].get("publish", "false") == "false"
    assert "pypi-token" not in flow["with"]
    assert "secrets." not in validation.read_text(encoding="utf-8")


@pytest.mark.parametrize("install_status", [0, 37])
def test_install_smoke_uses_the_pinned_backend_and_propagates_resolver_failures(tmp_path, install_status):
    venv = tmp_path / "isolated"
    log = tmp_path / "commands"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env bash
set -eo pipefail
printf 'uv %s\n' "$*" >> "$SMOKE_LOG"
case "$1 $2" in
  'pip install') exit "$INSTALL_STATUS" ;;
esac
if [[ "$1" == venv ]]; then
  mkdir -p "${@: -1}/bin"
  cat > "${@: -1}/bin/python" <<'SH'
#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$SMOKE_LOG"
SH
  chmod +x "${@: -1}/bin/python"
fi
""",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SMOKE_LOG": str(log),
        "INSTALL_STATUS": str(install_status),
    }

    result = subprocess.run(
        ["bash", str(CI / "scripts/ci_release_test_install.sh"), "gptqmodel-fixture.tar.gz", str(venv)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == install_status, result.stdout + result.stderr
    commands = log.read_text(encoding="utf-8").splitlines()
    constraint = str(CI / "ci/torch-cu132.txt")
    assert f"uv pip install --python {venv}/bin/python --torch-backend cu132 --constraint {constraint} dist/gptqmodel-fixture.tar.gz" in commands
    if install_status == 0:
        assert commands[2:] == [
            f"uv pip check --python {venv}/bin/python",
            "python -I -X gil=0 .github/scripts/ci_assert_free_threaded.py --installed",
        ]
    else:
        assert len(commands) == 2


def _load_ci_workflow():
    spec = importlib.util.spec_from_file_location("ci_workflow_under_test", CI_WORKFLOW)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_unit_test_plan():
    module = _load_ci_workflow()
    workflow = _yaml(WORKFLOWS / "unit_tests.yml")
    plan = module.build_test_plan(
        ignored_test_files=workflow["env"]["IGNORED_TEST_FILES"],
        test_names="",
        test_regex="",
        tests_root=ROOT / "tests",
    )
    return plan, module


def test_default_unit_test_plan_is_sharded_within_the_github_matrix_limit():
    plan, module = _default_unit_test_plan()
    limit = getattr(module, "DEFAULT_MAX_MATRIX_ENTRIES", GITHUB_MATRIX_ENTRY_LIMIT)
    assert 0 < limit <= GITHUB_MATRIX_ENTRY_LIMIT

    # The default unfiltered Torch selection is larger than one matrix, so the
    # planner must emit more than one shard for it.
    assert len(plan["torch_matrix"]) > GITHUB_MATRIX_ENTRY_LIMIT
    for group in ("cpu", "torch", "model"):
        matrix = plan[f"{group}_matrix"]
        shards = plan.get(f"{group}_matrix_shards") or [matrix]
        assert [entry for shard in shards for entry in shard] == matrix
        assert all(len(shard) <= limit for shard in shards), (group, [len(shard) for shard in shards])
        # Each shard is passed to the reusable workflow as one input payload.
        assert all(len(json.dumps(shard)) <= GITHUB_WORKFLOW_INPUT_LIMIT for shard in shards)
    assert len(plan["torch_matrix_shards"]) > 1


def test_unit_test_workflow_hands_the_plan_through_a_file():
    workflow = _yaml(WORKFLOWS / "unit_tests.yml")
    job = workflow["jobs"]["list-test-files"]
    step = next(step for step in job["steps"] if step.get("name") == "List files")
    run = step["run"]

    assert 'plan_file="$RUNNER_TEMP/unit-test-plan.json"' in run
    assert '> "$plan_file"' in run
    assert 'python - "$plan_file"' in run
    assert "TEST_PLAN=" not in run
    assert "test_plan=$(python" not in run
    for group in ("cpu", "torch", "model"):
        assert f"{group}-shards" in job["outputs"]
        assert f"{group}-shard-indexes" in job["outputs"]


def test_unit_test_jobs_expand_one_shard_per_matrix():
    workflow = _yaml(WORKFLOWS / "unit_tests.yml")
    for job_name, group in (("cpu", "cpu"), ("torch", "torch"), ("models", "model")):
        job = workflow["jobs"][job_name]
        assert job["strategy"]["max-parallel"] == 1
        shard = job["strategy"]["matrix"]["shard"]
        assert f"needs.list-test-files.outputs.{group}-shard-indexes" in shard
        matrix_json = job["with"]["matrix_json"]
        assert f"needs.list-test-files.outputs.{group}-shards" in matrix_json
        assert "[matrix.shard]" in matrix_json


def test_matrix_shard_helper_preserves_order_and_rejects_loss():
    module = _load_ci_workflow()
    entries = [{"test_script": f"test_{index}"} for index in range(600)]

    shards = module.split_matrix_entries(entries, GITHUB_MATRIX_ENTRY_LIMIT)
    assert [len(shard) for shard in shards] == [256, 256, 88]
    assert [entry for shard in shards for entry in shard] == entries
    module.validate_matrix_shards("torch", entries, shards, GITHUB_MATRIX_ENTRY_LIMIT)

    with pytest.raises(ValueError):
        module.split_matrix_entries(entries, 0)
    with pytest.raises(ValueError):
        module.validate_matrix_shards("torch", entries, [entries[:100]], GITHUB_MATRIX_ENTRY_LIMIT)


def test_default_unit_test_plan_file_exceeds_one_environment_value(tmp_path):
    workflow = _yaml(WORKFLOWS / "unit_tests.yml")
    plan_path = tmp_path / "unit-test-plan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CI_WORKFLOW),
            "list-tests",
            "--ignored-test-files",
            workflow["env"]["IGNORED_TEST_FILES"],
            "--test-regex",
            "",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    plan_path.write_text(result.stdout, encoding="utf-8")

    # The Linux per-argument limit is why the workflow cannot keep using an
    # environment variable for the plan.
    assert plan_path.stat().st_size > LINUX_MAX_ARG_STRLEN
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for group in ("cpu", "torch", "model"):
        matrix = plan[f"{group}_matrix"]
        shards = plan.get(f"{group}_matrix_shards") or [matrix]
        assert [entry for shard in shards for entry in shard] == matrix
        assert all(len(shard) <= GITHUB_MATRIX_ENTRY_LIMIT for shard in shards)
