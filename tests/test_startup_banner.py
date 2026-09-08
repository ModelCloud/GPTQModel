# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "gptqmodel" / "_banner.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "gptqmodel_banner_test_module", MODULE_PATH
)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None

banner_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(banner_module)


def _init_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            path.name,
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


@pytest.mark.parametrize("installation", ["wheel", "checkout", "worktree"])
def test_git_hash_uses_package_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installation: str
) -> None:
    foreign = tmp_path / "foreign"
    foreign_commit = _init_repository(foreign)
    source = tmp_path / "source"
    source_commit = _init_repository(source)
    assert source_commit != foreign_commit

    package_root = source
    if installation == "wheel":
        package_root = tmp_path / "site-packages"
    elif installation == "worktree":
        package_root = tmp_path / "worktree"
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "worktree",
                "add",
                "--detach",
                str(package_root),
                "HEAD",
            ],
            check=True,
            capture_output=True,
        )
        assert (package_root / ".git").is_file()

    monkeypatch.setattr(
        banner_module, "__file__", str(package_root / "gptqmodel" / "_banner.py")
    )
    monkeypatch.chdir(foreign)
    expected = "" if installation == "wheel" else f"+{source_commit}"
    assert banner_module._get_git_commit() == expected


def test_build_startup_banner_aligns_versions():
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
        triton_version="3.6.0",
    )

    lines = banner.splitlines()
    assert lines[0] == "LOGO"
    assert lines[1].strip().endswith("5.8.0")
    assert lines[2].strip().endswith("5.3.0")
    assert lines[3].strip().endswith("2.10.0+cu130")
    assert lines[4].strip().endswith("3.6.0")
    assert lines[1].startswith("GPT-QModel")
    assert lines[2].startswith("Transformers")
    assert lines[3].startswith("Torch")
    assert lines[4].startswith("Triton")
    assert {line.index(":") for line in lines[1:]} == {13}


def test_build_startup_banner_skips_missing_optional_versions():
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
    )

    assert "Triton version" not in banner
    assert "Triton" not in banner


def test_get_startup_banner_resolves_optional_versions(monkeypatch):
    def fake_resolve(package_names):
        if tuple(package_names) == banner_module.TRITON_PACKAGE_CANDIDATES:
            return "3.6.0"
        raise AssertionError(f"Unexpected package candidates: {package_names}")

    monkeypatch.setattr(
        banner_module,
        "resolve_installed_package_version",
        fake_resolve,
    )

    banner = banner_module.get_startup_banner(
        "LOGO\n",
        gptqmodel_version="5.8.0",
        transformers_version="5.3.0",
        torch_version="2.10.0+cu130",
    )

    assert any(
        line.startswith("Triton") and line.endswith("3.6.0")
        for line in banner.splitlines()
    )
