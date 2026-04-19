# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
from __future__ import annotations

import gptqmodel.utils.machete as machete_module
from gptqmodel.utils import cpp as cpp_module


def test_machete_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    cutlass_root = tmp_path / "cutlass"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    (project_root / "gptqmodel_ext" / "cutlass_extensions").mkdir(parents=True)
    (cutlass_root / "include").mkdir(parents=True)
    (cutlass_root / "tools" / "library" / "include").mkdir(parents=True)
    (cutlass_root / "tools" / "util" / "include").mkdir(parents=True)
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(machete_module, "_machete_project_root", lambda: project_root)
    monkeypatch.setattr(machete_module, "_ensure_cutlass_source", lambda: cutlass_root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = machete_module._machete_include_paths()

    assert include_paths[:4] == [
        str(project_root / "gptqmodel_ext"),
        str(project_root / "gptqmodel_ext" / "cutlass_extensions"),
        str(cutlass_root / "include"),
        str(cutlass_root / "tools" / "library" / "include"),
    ]
    assert str(wheel_cuda_include) in include_paths


def test_machete_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    cutlass_root = tmp_path / "cutlass"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    (project_root / "gptqmodel_ext" / "cutlass_extensions").mkdir(parents=True)
    (cutlass_root / "include").mkdir(parents=True)
    (cutlass_root / "tools" / "library" / "include").mkdir(parents=True)
    (cutlass_root / "tools" / "util" / "include").mkdir(parents=True)
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    for header_name in machete_module._MACHETE_REQUIRED_CUDA_HEADERS:
        (local_cuda_include / header_name).write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(machete_module, "_machete_project_root", lambda: project_root)
    monkeypatch.setattr(machete_module, "_ensure_cutlass_source", lambda: cutlass_root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = machete_module._machete_include_paths()

    assert include_paths == [
        str(project_root / "gptqmodel_ext"),
        str(project_root / "gptqmodel_ext" / "cutlass_extensions"),
        str(cutlass_root / "include"),
        str(cutlass_root / "tools" / "library" / "include"),
        str(cutlass_root / "tools" / "util" / "include"),
    ]
