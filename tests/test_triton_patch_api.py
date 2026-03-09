# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import ast
import threading
import time
from pathlib import Path

import gptqmodel
from gptqmodel.utils import nogil_patcher


def test_triton_patch_apply_runs_once(monkeypatch):
    call_count = 0

    def fake_patch():
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(nogil_patcher, "patch_triton_autotuner", fake_patch)
    monkeypatch.setattr(nogil_patcher.TritonPatch, "_applied", False)

    nogil_patcher.TritonPatch.apply()
    nogil_patcher.TritonPatch.apply()
    nogil_patcher.TritonPatch.apply()

    assert call_count == 1


def test_triton_patch_apply_runs_once_across_threads(monkeypatch):
    call_count = 0
    call_count_lock = threading.Lock()

    def fake_patch():
        nonlocal call_count
        time.sleep(0.01)
        with call_count_lock:
            call_count += 1

    monkeypatch.setattr(nogil_patcher, "patch_triton_autotuner", fake_patch)
    monkeypatch.setattr(nogil_patcher.TritonPatch, "_applied", False)

    threads = [threading.Thread(target=nogil_patcher.TritonPatch.apply) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert call_count == 1


def test_triton_patch_public_api_export():
    assert gptqmodel.TritonPatch is nogil_patcher.TritonPatch


def test_package_init_uses_triton_patch_api():
    init_py = Path(__file__).resolve().parents[1] / "gptqmodel" / "__init__.py"
    tree = ast.parse(init_py.read_text(encoding="utf-8"))

    call_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            call_names.append((node.func.value.id, node.func.attr))
        elif isinstance(node.func, ast.Name):
            call_names.append((node.func.id, ""))

    assert ("TritonPatch", "apply") in call_names
    assert ("patch_triton_autotuner", "") not in call_names
