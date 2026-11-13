# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The legacy GPTQModel NF12 shim delegates to the vendored runtime."""

import importlib.util
from pathlib import Path

import pytest
import torch


def test_shared_library_names_match_build_and_loader():
    from gptqmodel_nf12 import _jit, _library

    assert _jit._LIBRARIES == ("libnf12.so", "libnf12_torch.so")
    assert _library.library_path.__defaults__ == ("libnf12.so",)
    source = Path(__file__).resolve().parents[1] / "gptqmodel_ext/nf12/CMakeLists.txt"
    cmake = source.read_text(encoding="utf-8")
    assert "OUTPUT_NAME nf12)" in cmake
    assert "OUTPUT_NAME nf12_torch\n" in cmake


@pytest.fixture
def nf12():
    # Exercise the shim without importing the full quantization framework;
    # its optional transformer dependencies are unrelated to this dispatch.
    path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/nf12.py"
    spec = importlib.util.spec_from_file_location("_test_gptqmodel_nf12", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_runtime_loads_vendored_package(monkeypatch, nf12):

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nf12.gptqmodel_nf12, "load", lambda: calls.append("load"))
    assert nf12.load()
    assert calls == ["load"]
    assert nf12.last_error() is None


def test_legacy_runtime_reports_vendored_load_error(monkeypatch, nf12):

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail():
        raise RuntimeError("vendored build failed")

    monkeypatch.setattr(nf12.gptqmodel_nf12, "load", fail)
    assert not nf12.load()
    assert "vendored build failed" in nf12.last_error()
    with pytest.raises(RuntimeError, match="vendored build failed"):
        nf12.encode(torch.empty((16, 64), dtype=torch.bfloat16))
