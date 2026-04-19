# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
from __future__ import annotations

import gptqmodel.utils.qqq as qqq_module
from gptqmodel.utils import cpp as cpp_module


def test_qqq_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    root = tmp_path / "qqq"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(qqq_module, "_qqq_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = qqq_module._qqq_include_paths()

    assert include_paths[0] == str(root)
    assert str(wheel_cuda_include) in include_paths


def test_qqq_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    root = tmp_path / "qqq"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (local_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(qqq_module, "_qqq_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = qqq_module._qqq_include_paths()

    assert include_paths == [str(root)]
