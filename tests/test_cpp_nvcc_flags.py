# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import gptqmodel.utils.cpp as cpp_utils


def _reset_local_nvcc_caches() -> None:
    cpp_utils._LOCAL_NVCC_VERSION_CACHE = None
    cpp_utils._LOCAL_NVCC_VERSION_INITIALISED = False


def test_local_nvcc_supports_static_global_template_stub_uses_version_boundary(monkeypatch):
    _reset_local_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None)

    calls: list[tuple[str, ...]] = []

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="Cuda compilation tools, release 12.8, V12.8.89", stderr="")

    monkeypatch.setattr(cpp_utils.subprocess, "run", fake_run)

    assert cpp_utils.local_nvcc_supports_static_global_template_stub() is True
    assert cpp_utils.local_nvcc_supports_static_global_template_stub() is True
    assert calls == [("/usr/local/cuda/bin/nvcc", "--version")]


def test_local_nvcc_supports_static_global_template_stub_rejects_older_nvcc(monkeypatch):
    _reset_local_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None)

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="Cuda compilation tools, release 12.7, V12.7.61", stderr="")

    monkeypatch.setattr(cpp_utils.subprocess, "run", fake_run)

    assert cpp_utils.local_nvcc_supports_static_global_template_stub() is False


def test_local_nvcc_supports_static_global_template_stub_rejects_missing_nvcc(monkeypatch):
    _reset_local_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda _cmd: None)

    assert cpp_utils.local_nvcc_supports_static_global_template_stub() is False
