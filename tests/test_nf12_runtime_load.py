# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The legacy GPTQModel NF12 shim delegates to the vendored runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("static_weights", [False, True])
def test_linear_weight_readiness_is_explicit(monkeypatch, static_weights):
    from gptqmodel_nf12 import api

    calls = []
    result = object()

    def linear(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(api, "load", lambda: None)
    monkeypatch.setattr(api, "_check_input", lambda *_: None)
    monkeypatch.setattr(torch.ops, "nf12", SimpleNamespace(linear=linear))
    weight = SimpleNamespace(shape=(16, 64), packed=object(), escapes=object(), exp_base=60)
    x, bias = object(), object()
    assert api.linear(x, weight, bias, static_weights=static_weights) is result
    assert calls == [((x, weight.packed, weight.escapes, 60, bias),
                      {"static_weights": True} if static_weights else {})]


@pytest.mark.parametrize("filename", ["gemv.cu", "small_m_direct.cu", "small_m_async.cu"])
def test_jit_fingerprint_covers_new_cuda_families(tmp_path, filename):
    from gptqmodel_nf12 import _jit

    source = Path(__file__).resolve().parents[1] / "gptqmodel_ext/nf12"
    kernels = tmp_path / "csrc/linear"
    kernels.mkdir(parents=True)
    (tmp_path / "include").mkdir()
    (tmp_path / "CMakeLists.txt").write_bytes((source / "CMakeLists.txt").read_bytes())
    kernel = kernels / filename
    kernel.write_bytes((source / "csrc/linear" / filename).read_bytes())
    before = _jit._cache_key(tmp_path, [103])
    kernel.write_text(kernel.read_text() + "\n// changed kernel\n")
    assert _jit._cache_key(tmp_path, [103]) != before


@pytest.mark.parametrize("static_weights", [False, True])
def test_linear_fake_accepts_weight_readiness(monkeypatch, static_weights):
    from gptqmodel_nf12 import _library

    fakes = {}

    def register_fake(name):
        def register(fn):
            fakes[name] = fn
            return fn
        return register

    monkeypatch.setattr(torch.library, "register_fake", register_fake)
    _library._register_fakes()
    x = torch.empty((2, 3, 64), device="meta", dtype=torch.bfloat16)
    packed = torch.empty((16, 96), device="meta", dtype=torch.uint8)
    escapes = torch.empty((0, 8), device="meta", dtype=torch.int16)
    actual = fakes["nf12::linear"](x, packed, escapes, 60, static_weights=static_weights)
    assert actual.shape == (2, 3, 16)
    assert actual.dtype == x.dtype
    assert actual.device == x.device


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
