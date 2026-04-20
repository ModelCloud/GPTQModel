# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".github" / "scripts"


def _load_script_module(module_name: str):
    script_path = SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"tests_{module_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_test_runtime_detects_gpu_skip_marker():
    ci_workflow = _load_script_module("ci_workflow")

    runtime = ci_workflow.resolve_test_runtime("test_require_pkgs")

    assert runtime.test_name == "test_require_pkgs"
    assert runtime.test_path == "tests/test_require_pkgs.py"
    assert runtime.skip_gpu_allocation is True
    assert runtime.xpu_mode is False


def test_resolve_test_runtime_marks_xpu_tests():
    ci_workflow = _load_script_module("ci_workflow")

    runtime = ci_workflow.resolve_test_runtime("test_torch_xpu")

    assert runtime.test_path == "tests/test_torch_xpu.py"
    assert runtime.skip_gpu_allocation is True
    assert runtime.xpu_mode is True


def test_build_test_matrix_marks_model_entries():
    ci_workflow = _load_script_module("ci_workflow")

    matrix = ci_workflow.build_test_matrix(
        torch_tests=["test_require_pkgs"],
        model_tests=["models/test_opt"],
    )

    assert matrix == [
        {
            "test_script": "test_require_pkgs",
            "test_group": "torch",
            "alloc_gpu_count": "resolved",
            "require_single_gpu": "false",
            "include_model_test_mode": "false",
        },
        {
            "test_script": "models/test_opt",
            "test_group": "model",
            "alloc_gpu_count": "1",
            "require_single_gpu": "true",
            "include_model_test_mode": "true",
        },
    ]


@pytest.mark.parametrize(
    "script_name",
    [
        "ci_prepare_checkout.sh",
        "ci_release_build_sdist.sh",
        "ci_release_check_dist.sh",
        "ci_release_test_install.sh",
        "ci_release_upload_local.sh",
        "ci_release_wait_for_confirmation.sh",
        "ci_restore_uv_cache.sh",
        "ci_install_modelcloud_git_deps.sh",
        "ci_write_runner_outputs.sh",
        "ci_unit_activate_uv_env.sh",
        "ci_unit_setup_uv_env.sh",
    ],
)
def test_ci_shell_scripts_have_valid_bash_syntax(script_name: str):
    subprocess.run(["bash", "-n", str(SCRIPT_DIR / script_name)], check=True)


def test_ci_write_runner_outputs_script_sets_outputs(tmp_path: Path):
    github_output = tmp_path / "github_output"
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(github_output)

    subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "ci_write_runner_outputs.sh"),
            "10.0.0.1",
            "12345",
            "99999",
            "7",
        ],
        check=True,
        env=env,
    )

    output = github_output.read_text(encoding="utf-8")
    assert "ip=10.0.0.1" in output
    assert "run_id=99999" in output
    assert 'max-parallel={"size": 7}' in output


def test_release_source_common_action_yaml_loads():
    action_path = SCRIPT_DIR.parent / "actions" / "release-source-common" / "action.yml"

    with action_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data["runs"]["using"] == "composite"


def test_ci_restore_uv_cache_script_skips_unchanged_archive(tmp_path: Path):
    archive_root = tmp_path / "archive-src"
    uv_dir = archive_root / "uv"
    uv_dir.mkdir(parents=True)
    (uv_dir / "hello.txt").write_text("from-archive", encoding="utf-8")

    tar_file = tmp_path / "uv.tar.xz"
    with tarfile.open(tar_file, "w:xz") as tar:
        tar.add(uv_dir, arcname="uv")

    cache_dir = tmp_path / "cache"
    subprocess.run(
        ["bash", str(SCRIPT_DIR / "ci_restore_uv_cache.sh"), str(tar_file), str(cache_dir)],
        check=True,
    )
    restored_file = cache_dir / "uv" / "hello.txt"
    assert restored_file.read_text(encoding="utf-8") == "from-archive"

    restored_file.write_text("mutated-cache", encoding="utf-8")
    subprocess.run(
        ["bash", str(SCRIPT_DIR / "ci_restore_uv_cache.sh"), str(tar_file), str(cache_dir)],
        check=True,
    )
    assert restored_file.read_text(encoding="utf-8") == "mutated-cache"


def test_ci_release_check_dist_script_sets_latest_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    older = dist_dir / "pkg-old.tar.gz"
    newer = dist_dir / "pkg-new.tar.gz"
    older.write_text("old", encoding="utf-8")
    time.sleep(0.01)
    newer.write_text("new", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_twine = fake_bin / "twine"
    fake_twine.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo \"$@\" > \"$TWINE_LOG\"\n",
        encoding="utf-8",
    )
    fake_twine.chmod(0o755)

    github_env = tmp_path / "github_env"
    twine_log = tmp_path / "twine.log"
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    monkeypatch.setenv("TWINE_LOG", str(twine_log))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    subprocess.run(
        ["bash", str(SCRIPT_DIR / "ci_release_check_dist.sh"), str(dist_dir)],
        check=True,
    )

    assert "PKG_NAME=pkg-new.tar.gz" in github_env.read_text(encoding="utf-8")
    assert twine_log.read_text(encoding="utf-8").strip() == f"check {newer}"
