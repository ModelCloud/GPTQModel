# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemv_fast_awq as gemv_fast_awq


def _build_module() -> gemv_fast_awq.AwqGEMVFastQuantLinear:
    return gemv_fast_awq.AwqGEMVFastQuantLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=8,
        bias=False,
        register_buffers=True,
    )


def test_awq_gemv_fast_decode_uses_jit_decode_kernel(monkeypatch):
    module = _build_module()
    calls = {}

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)

    def fake_decode(inputs, qweight, scales, zeros, m, n, k, group_size):
        calls["decode"] = {
            "shape": tuple(inputs.shape),
            "m": m,
            "n": n,
            "k": k,
            "group_size": group_size,
        }
        return torch.ones((inputs.shape[0], inputs.shape[1], module.out_features), dtype=torch.float16)

    def fail_prefill(*_args, **_kwargs):
        raise AssertionError("prefill kernel should not be used for decode-shaped inputs")

    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", fake_decode)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", fail_prefill)

    x = torch.randn((4, 1, module.in_features), dtype=torch.float32)
    out = module(x)

    assert calls["decode"] == {
        "shape": (4, 1, module.in_features),
        "m": 4,
        "n": module.out_features,
        "k": module.in_features,
        "group_size": module.group_size,
    }
    assert out.shape == (4, 1, module.out_features)
    assert out.dtype == torch.float32


def test_awq_gemv_fast_prefill_uses_jit_prefill_kernel(monkeypatch):
    module = _build_module()
    calls = {"prefill": 0}

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("decode kernel should not be used for prefill-shaped inputs")

    def fake_prefill(inputs, qweight, scales, zeros):
        calls["prefill"] += 1
        calls["shape"] = tuple(inputs.shape)
        return torch.ones((inputs.shape[0], inputs.shape[1], module.out_features), dtype=torch.float16)

    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", fail_decode)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", fake_prefill)

    x = torch.randn((2, 4, module.in_features), dtype=torch.float16)
    out = module(x)

    assert calls["prefill"] == 1
    assert calls["shape"] == (2, 4, module.in_features)
    assert out.shape == (2, 4, module.out_features)
    assert out.dtype == torch.float16


def test_awq_gemv_fast_raises_runtime_error_when_jit_ops_missing(monkeypatch):
    module = _build_module()

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: False)
    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_error", lambda: "missing awq jit ops")

    with pytest.raises(ModuleNotFoundError, match="missing awq jit ops"):
        module(torch.randn((1, 1, module.in_features), dtype=torch.float16))
