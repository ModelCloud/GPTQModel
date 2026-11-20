# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``optimize()`` must install the Paro CPU layout before compiling (#5581).

The AVX512/ATen CPU branches only read the runtime layout while dynamo traces,
so a model compiled before the first eager forward traced the dense dequantize
fallback and stayed there for the life of the process.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import paro as paro_module
from gptqmodel.nn_modules.qlinear.paro import ParoLinear


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32


def _paro(dtype=torch.bfloat16):
    module = ParoLinear(
        bits=4,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        krot=4,
        register_buffers=True,
    )
    module.scales = module.scales.to(dtype)
    if module.bias is not None:
        module.bias = module.bias.to(dtype)
    module.eval()
    return module


def _optimize(module, backend="eager"):
    holder = nn.Module()
    holder.add_module("layer", module)
    holder.add_module("norm", nn.LayerNorm(IN_FEATURES, dtype=torch.bfloat16))
    stub = SimpleNamespace(quantized=True, compiled=False, model=holder)
    BaseQModel.optimize(stub, backend=backend)
    return stub


def _activation(rows=2):
    return torch.randn(rows, IN_FEATURES, dtype=torch.bfloat16)


def _layout(module):
    return module._live_cpu_layout()


def test_model_optimize_installs_the_cpu_layout():
    module = _paro()
    assert _layout(module) is None
    _optimize(module)
    assert _layout(module) in ("aten", "avx512")


def test_a_cold_compiled_call_installs_the_cpu_layout():
    module = _paro()
    compiled = torch.compile(module, backend="eager")
    with torch.no_grad():
        compiled(_activation())
    assert _layout(module) in ("aten", "avx512")


def test_the_warm_layout_matches_the_eager_result():
    module = _paro()
    eager = _paro()
    activation = _activation()
    with torch.no_grad():
        expected = eager(activation).clone()
    _optimize(module)
    assert _layout(module) in ("aten", "avx512")
    compiled = torch.compile(module, backend="eager")
    with torch.no_grad():
        actual = compiled(activation)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_bf16_prefers_the_avx512_layout(monkeypatch):
    module = _paro()
    calls = []
    monkeypatch.setattr(paro_module, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(
        module,
        "_ensure_cpu_layout",
        lambda layout, dtype=None: calls.append((layout, dtype)) or True,
    )
    assert module._prepare_cpu_inference_layout(torch.bfloat16) is True
    assert calls == [("avx512", None)]


def test_avx512_failure_falls_back_to_aten(monkeypatch):
    module = _paro()
    calls = []
    monkeypatch.setattr(paro_module, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(paro_module, "_has_local_int4pack_cpu_ops", lambda: True)
    monkeypatch.setattr(
        paro_module.TorchAtenAwqLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )

    def _ensure(layout, dtype=None):
        calls.append((layout, dtype))
        return layout == "aten"

    monkeypatch.setattr(module, "_ensure_cpu_layout", _ensure)
    assert module._prepare_cpu_inference_layout(torch.bfloat16) is True
    assert calls == [("avx512", None), ("aten", torch.bfloat16)]


def test_fp16_uses_the_aten_layout(monkeypatch):
    module = _paro()
    calls = []
    monkeypatch.setattr(paro_module, "_has_local_int4pack_cpu_ops", lambda: True)
    monkeypatch.setattr(
        paro_module.TorchAtenAwqLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    monkeypatch.setattr(
        module,
        "_ensure_cpu_layout",
        lambda layout, dtype=None: calls.append((layout, dtype)) or True,
    )
    assert module._prepare_cpu_inference_layout(torch.float16) is True
    assert calls == [("aten", torch.float16)]


def test_an_unsupported_shape_stays_dense(monkeypatch):
    module = _paro()
    monkeypatch.setattr(paro_module, "has_avx512_bf16", lambda: False)
    module.group_size = 16
    assert module._prepare_cpu_inference_layout(torch.bfloat16) is False


def test_training_modules_are_left_alone():
    module = _paro()
    module.train()
    assert module._prepare_cpu_inference_layout(torch.bfloat16) is False


def test_a_compiled_grad_ca_after_optimize():
    module = _paro()
    _optimize(module)
    compiled = torch.compile(module, backend="eager")
    x = _activation().requires_grad_(True)
    with torch.enable_grad():
        out = compiled(x)
    assert out.shape[-1] == OUT_FEATURES
