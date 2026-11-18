# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Contract checks for the CI runtime and release publication boundary."""

import os
import subprocess
from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github"
WORKFLOWS = CI / "workflows"
CUDA_IMAGE = "nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf"


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
