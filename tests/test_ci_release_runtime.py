# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Contract checks for the CI runtime and release publication boundary."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github"
WORKFLOWS = CI / "workflows"
CUDA_IMAGE = "nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf"
CI_WORKFLOW = CI / "scripts/ci_workflow.py"
CI_TESTS = CI / "scripts/ci_tests.py"
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


def _start_allocator_stub(gpu_ids):
    """Local allocator that reports ``gpu_ids`` for every keepalive."""
    keepalives = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self):
            payload = json.dumps({"gpuIds": gpu_ids}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(length)
            if self.path.endswith("/keepalive"):
                keepalives.append(self.path)
            self._reply()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, keepalives


def _run_ci_tests_loopback(tmp_path, *, extra_args, cuda_visible_devices, gpu_ids):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    # The contracts job installs only pytest/pyyaml, so isolate the keepalive
    # payload's optional host-inventory dependency at its import boundary.
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "device_smi.py").write_text(
        "class Device:\n"
        "    def __init__(self, kind):\n"
        "        self.model = kind\n"
        "        self.arch = 'x86_64'\n"
        "        self.name = 'Linux'\n",
        encoding="utf-8",
    )
    seen_env = tmp_path / "seen-cuda-visible-devices.txt"
    (tests_dir / "test_ci_lease_probe.py").write_text(
        "import os\n"
        "import pathlib\n"
        "import time\n"
        "\n"
        "\n"
        "def test_lease_probe():\n"
        "    pathlib.Path(os.environ['PROBE_OUTPUT']).write_text(\n"
        "        os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), encoding='utf-8'\n"
        "    )\n"
        "    time.sleep(2.5)\n",
        encoding="utf-8",
    )
    server, keepalives = _start_allocator_stub(gpu_ids)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(CI_TESTS),
                "run",
                "--base-url",
                f"http://127.0.0.1:{server.server_address[1]}",
                "--run-id",
                "5370",
                "--test-script",
                "test_ci_lease_probe",
                "--runner",
                "loopback",
                "--gpu-id",
                cuda_visible_devices,
                "--monitor-interval-sec",
                "1",
                "--log-root",
                str(tmp_path / "logs"),
                *extra_args,
            ],
            cwd=tmp_path,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "PROBE_OUTPUT": str(seen_env),
                "PYTHONPATH": os.pathsep.join(
                    value
                    for value in (str(stub_dir), os.environ.get("PYTHONPATH", ""))
                    if value
                ),
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
    return result, keepalives, seen_env


def test_cpu_runtime_without_a_lease_clears_cuda_and_skips_keepalives(tmp_path):
    result, keepalives, seen_env = _run_ci_tests_loopback(
        tmp_path,
        extra_args=[],
        cuda_visible_devices="0",
        gpu_ids="-1",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert keepalives == []
    assert "start to keep alive" not in result.stdout
    assert seen_env.read_text(encoding="utf-8") == ""


def test_leased_gpu_runtime_keeps_cuda_and_starts_keepalives(tmp_path):
    result, keepalives, seen_env = _run_ci_tests_loopback(
        tmp_path,
        extra_args=["--gpu-lease"],
        cuda_visible_devices="0",
        gpu_ids="0",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert keepalives
    assert "start to keep alive" in result.stdout
    assert seen_env.read_text(encoding="utf-8") == "0"


def test_leased_runtime_without_cuda_visibility_fails_closed(tmp_path):
    result, keepalives, seen_env = _run_ci_tests_loopback(
        tmp_path,
        extra_args=["--gpu-lease"],
        cuda_visible_devices="",
        gpu_ids="0",
    )

    assert result.returncode == 22
    assert keepalives == []
    assert "GPU lease declared" in result.stdout
    assert not seen_env.exists()


def test_unit_workflow_leases_only_after_gpu_allocation():
    reusable = _yaml(WORKFLOWS / "unit_tests_reusable.yml")
    run_step = next(step for step in reusable["jobs"]["test"]["steps"] if step.get("name") == "Run tests")
    script = run_step["run"]
    block = script.split("extra_args=()", 1)[1].split('if [[ "${XPU_MODE}"', 1)[0]
    skip_branch, lease_branch = block.split("else", 1)

    assert 'if [[ "${SKIP_GPU_ALLOCATION}" == "true" ]]; then' in skip_branch
    assert "--clear-cuda" in skip_branch
    assert "--gpu-lease" not in skip_branch
    assert "--gpu-lease" in lease_branch
    assert "extra_args+=(--xpu-mode)" in script


def test_unit_workflow_passes_the_matrix_group_to_runtime_resolution():
    reusable = _yaml(WORKFLOWS / "unit_tests_reusable.yml")
    activate = next(step for step in reusable["jobs"]["test"]["steps"] if step.get("name") == "Activate test environment")

    assert '"${{ matrix.test_group }}"' in activate["run"]
    assert '--test-group "$test_group"' in (CI / "scripts/ci_unit_activate_uv_env.sh").read_text(encoding="utf-8")


def test_cpu_marked_test_skips_allocation_without_xpu_mode():
    module = _load_ci_workflow()
    runtime = module.resolve_test_runtime("test_threadx_shutdown", tests_root=ROOT / "tests")

    assert runtime.skip_gpu_allocation is True
    assert runtime.xpu_mode is False


def test_cpu_matrix_entry_without_a_marker_still_skips_allocation():
    module = _load_ci_workflow()

    assert module.resolve_test_runtime("test_threadx_cpu_locks", tests_root=ROOT / "tests").skip_gpu_allocation is False
    runtime = module.resolve_test_runtime(
        "test_threadx_cpu_locks",
        tests_root=ROOT / "tests",
        test_group="cpu",
    )
    assert runtime.skip_gpu_allocation is True
    assert runtime.xpu_mode is False


def test_cpu_runtime_resolution_clears_cuda_for_the_whole_job(tmp_path):
    github_env = tmp_path / "github-env"
    result = subprocess.run(
        [
            sys.executable,
            str(CI_WORKFLOW),
            "resolve-test-runtime",
            "--test-name",
            "test_threadx_cpu_locks",
            "--test-group",
            "cpu",
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_ENV": str(github_env)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "skip_gpu_allocation=true" in result.stdout
    written = github_env.read_text(encoding="utf-8").splitlines()
    assert "SKIP_GPU_ALLOCATION=true" in written
    assert "CUDA_VISIBLE_DEVICES=" in written


def test_gpu_runtime_resolution_keeps_cuda_for_the_job(tmp_path):
    github_env = tmp_path / "github-env"
    subprocess.run(
        [
            sys.executable,
            str(CI_WORKFLOW),
            "resolve-test-runtime",
            "--test-name",
            "test_threadx_cpu_locks",
            "--test-group",
            "torch",
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_ENV": str(github_env)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "CUDA_VISIBLE_DEVICES=" not in github_env.read_text(encoding="utf-8").splitlines()
