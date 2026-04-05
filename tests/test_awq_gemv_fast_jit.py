# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemv_fast_awq as gemv_fast_awq


def _build_module() -> gemv_fast_awq.AwqGEMVFastLinear:
    return gemv_fast_awq.AwqGEMVFastLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=8,
        bias=False,
        register_buffers=True,
    )


def _build_llm_awq_module() -> gemv_fast_awq.LLMAwqLinear:
    return gemv_fast_awq.LLMAwqLinear(
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


def test_awq_gemv_fast_decode_normalizes_noncontiguous_inputs_and_buffers(monkeypatch):
    module = _build_module()
    module.qweight = module.qweight.t().contiguous().t()
    module.scales = module.scales.t().contiguous().t()
    module.qzeros = module.qzeros.t().contiguous().t()

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)

    def fake_decode(inputs, qweight, scales, zeros, m, n, k, group_size):
        assert inputs.is_contiguous()
        assert qweight.is_contiguous()
        assert scales.is_contiguous()
        assert zeros.is_contiguous()
        assert inputs.dtype == torch.float16
        assert scales.dtype == torch.float16
        assert zeros.dtype == torch.float16
        return torch.ones((inputs.shape[0], inputs.shape[1], module.out_features), dtype=torch.float16)

    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", fake_decode)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", lambda *_args, **_kwargs: None)

    x = torch.randn((module.in_features, 1, 4), dtype=torch.float32).permute(2, 1, 0)
    assert not x.is_contiguous()
    module(x)


def test_awq_gemv_fast_prefill_normalizes_noncontiguous_inputs(monkeypatch):
    module = _build_module()

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", lambda *_args, **_kwargs: None)

    def fake_prefill(inputs, qweight, scales, zeros):
        assert inputs.is_contiguous()
        assert qweight.is_contiguous()
        assert scales.is_contiguous()
        assert zeros.is_contiguous()
        assert inputs.dtype == torch.float16
        return torch.ones((inputs.shape[0], inputs.shape[1], module.out_features), dtype=torch.float16)

    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", fake_prefill)

    x = torch.randn((2, module.in_features, 4), dtype=torch.float16).transpose(1, 2)
    assert x.shape == (2, 4, module.in_features)
    assert not x.is_contiguous()
    module(x)


def test_llm_awq_decode_normalizes_scaled_zeros_without_dynamic_attr_access(monkeypatch):
    module = _build_llm_awq_module()
    module.scaled_zeros = module.scaled_zeros.t().contiguous().t()

    monkeypatch.setattr(gemv_fast_awq, "awq_runtime_available", lambda: True)

    def fake_decode(inputs, qweight, scales, zeros, m, n, k, group_size):
        assert zeros.is_contiguous()
        assert zeros.dtype == torch.float16
        assert zeros.data_ptr() == module.scaled_zeros.data_ptr()
        return torch.ones((inputs.shape[0], inputs.shape[1], module.out_features), dtype=torch.float16)

    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemv_forward_decode", fake_decode)
    monkeypatch.setattr(gemv_fast_awq, "awq_fast_gemm_forward_prefill", lambda *_args, **_kwargs: None)

    x = torch.randn((4, 1, module.in_features), dtype=torch.float16)
    module(x)
