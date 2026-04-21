# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".github" / "scripts"


def _load_script_module(module_name: str):
    script_path = SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"tests_{module_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_pkgs_dedupes_common_requirements():
    ci_deps = _load_script_module("ci_deps")
    deps = {
        "common": ["transformers", "peft"],
        "tests": {
            "test_demo.py": ["transformers==4.44.2", "scipy"],
        },
    }

    specific, common = ci_deps.collect_pkgs(Path("tests/test_demo.py"), deps, dedupe_common=True)

    assert specific == ["scipy", "transformers==4.44.2"]
    assert common == ["peft"]


def test_run_uv_pip_batches_install(monkeypatch):
    ci_deps = _load_script_module("ci_deps")
    calls: list[tuple[list[str], bool]] = []

    def fake_check_call(cmd, shell):
        calls.append((cmd, shell))

    monkeypatch.setattr(ci_deps.subprocess, "check_call", fake_check_call)

    ci_deps.run_uv_pip("install", ["peft", "peft", "scipy"], extra_args=["--no-cache"])

    assert calls == [
        (["uv", "pip", "install", "--no-cache", "peft", "scipy"], False),
    ]


def test_run_uv_pip_uninstall_ignores_errors(monkeypatch):
    ci_deps = _load_script_module("ci_deps")
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_run(cmd, shell, check):
        calls.append((cmd, shell, check))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(ci_deps.subprocess, "run", fake_run)

    ci_deps.run_uv_pip("uninstall", ["flash_attn"], ignore_errors=True)

    assert calls == [
        (["uv", "pip", "uninstall", "-y", "flash_attn"], False, False),
    ]
