# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
from __future__ import annotations

import subprocess
import threading
import time

import gptqmodel.utils.cpp as cpp_utils


def _reset_nvcc_caches() -> None:
    cpp_utils._NVCC_VERSION_CACHE = None


def test_is_nvcc_compatible_uses_version_boundary(monkeypatch):
    _reset_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None)

    calls: list[tuple[str, ...]] = []

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="Cuda compilation tools, release 12.8, V12.8.89", stderr="")

    monkeypatch.setattr(cpp_utils.subprocess, "run", fake_run)

    assert cpp_utils.is_nvcc_compatible() is True
    assert cpp_utils.is_nvcc_compatible() is True
    assert calls == [("/usr/local/cuda/bin/nvcc", "--version")]


def test_is_nvcc_compatible_rejects_older_nvcc(monkeypatch):
    _reset_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None)

    def fake_run(args, capture_output, text, check):
        del capture_output, text, check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="Cuda compilation tools, release 12.7, V12.7.61", stderr="")

    monkeypatch.setattr(cpp_utils.subprocess, "run", fake_run)

    assert cpp_utils.is_nvcc_compatible() is False


def test_is_nvcc_compatible_rejects_missing_nvcc(monkeypatch):
    _reset_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda _cmd: None)

    assert cpp_utils.is_nvcc_compatible() is False
    assert cpp_utils._NVCC_VERSION_CACHE == (0, 0)


def test_is_nvcc_compatible_probes_nvcc_once_with_concurrent_callers(monkeypatch):
    _reset_nvcc_caches()
    monkeypatch.setattr(cpp_utils.shutil, "which", lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None)

    call_count = 0
    call_count_lock = threading.Lock()
    start_barrier = threading.Barrier(8)

    def fake_run(args, capture_output, text, check):
        nonlocal call_count
        del capture_output, text, check
        time.sleep(0.05)
        with call_count_lock:
            call_count += 1
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="Cuda compilation tools, release 12.8, V12.8.89", stderr="")

    monkeypatch.setattr(cpp_utils.subprocess, "run", fake_run)

    results = [None] * 8

    def worker(index: int):
        start_barrier.wait()
        results[index] = cpp_utils.is_nvcc_compatible()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True] * len(results)
    assert call_count == 1
