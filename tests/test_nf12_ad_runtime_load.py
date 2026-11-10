# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The legacy GPTQModel NF12 shim delegates to the vendored runtime."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def nf12_ad():
    # Exercise the shim without importing the full quantization framework;
    # its optional transformer dependencies are unrelated to this dispatch.
    path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/nf12_ad.py"
    spec = importlib.util.spec_from_file_location("_test_gptqmodel_nf12_ad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_runtime_loads_vendored_package(monkeypatch, nf12_ad):

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nf12_ad.gptqmodel_nf12, "load", lambda: calls.append("load"))
    assert nf12_ad.load()
    assert calls == ["load"]
    assert nf12_ad.last_error() is None


def test_legacy_runtime_reports_vendored_load_error(monkeypatch, nf12_ad):

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail():
        raise RuntimeError("vendored build failed")

    monkeypatch.setattr(nf12_ad.gptqmodel_nf12, "load", fail)
    assert not nf12_ad.load()
    assert "vendored build failed" in nf12_ad.last_error()
    with pytest.raises(RuntimeError, match="vendored build failed"):
        nf12_ad.encode(torch.empty((16, 64), dtype=torch.bfloat16))
