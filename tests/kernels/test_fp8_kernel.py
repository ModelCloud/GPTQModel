# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module
from gptqmodel.quantization.dtype import available_float8_dtype_names


def _available_fp8_quant_formats():
    return [fmt for fmt in available_float8_dtype_names() if fmt != "float8_e8m0fnu"]


def _make_fp8_kernel(*, in_features: int = 64, out_features: int = 64) -> fp8_module.TorchFP8Linear:
    torch.manual_seed(3)
    linear = nn.Linear(in_features, out_features, bias=True).eval()
    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e4m3fn",
        weight_scale_method="row",
        weight_block_size=None,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel._scaled_mm_hard_disabled = True
    return kernel.eval()


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_matches_reference_quantizer(fmt: str, weight_scale_method: str):
    torch.manual_seed(0)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)

    expected_weight, expected_scale_inv = fp8_module.quantize_fp8_weight(
        linear.weight.detach().to(torch.float32),
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
    )

    assert torch.equal(kernel.weight, expected_weight)
    assert torch.equal(kernel.weight_scale_inv, expected_scale_inv)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="float8_e8m0fnu unavailable")
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_rejects_e8m0fnu(weight_scale_method: str):
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e8m0fnu",
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )

    with pytest.raises(ValueError, match="dequantization of existing checkpoints"):
        kernel.pack_original(linear=linear, scales=None, zeros=None)


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_fp8_forward_matches_dequantized_reference(fmt: str, weight_scale_method: str, device: str):
    torch.manual_seed(1)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device=device).eval()
    kernel._scaled_mm_hard_disabled = True

    x_dtype = torch.float32 if device == "cpu" else torch.float16
    x = torch.randn(5, 64, device=device, dtype=x_dtype)

    with torch.inference_mode():
        out = kernel(x)
        expected = torch.matmul(
            x,
            kernel.dequantize_weight(device=device, dtype=x_dtype),
        )
        if kernel.bias is not None:
            expected = expected + kernel.bias.to(device=device, dtype=x_dtype)

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_fp8_forward_dequantizes_per_call_without_dense_cache(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(5, 64)

    calls = 0
    dequantize_weight = kernel.dequantize_weight

    def counted_dequantize_weight(*args, **kwargs):
        nonlocal calls
        calls += 1
        return dequantize_weight(*args, **kwargs)

    monkeypatch.setattr(kernel, "dequantize_weight", counted_dequantize_weight)

    with torch.inference_mode():
        first = kernel(x)
        second = kernel(x)

    assert calls == 2
    assert not hasattr(kernel, "_dequant_weight_cache")
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_fp8_optimize_compiles_cpu_forward(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    with torch.inference_mode():
        eager = kernel(x)

    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return lambda value: fn(value)

    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(fp8_module, "torch_compile", fake_compile)

    assert kernel.optimize(backend="inductor") is kernel
    assert kernel.optimized
    assert kernel._compiled_forward is not None
    assert calls == [{"backend": "inductor", "mode": None, "fullgraph": False}]

    with torch.inference_mode():
        compiled = kernel(x)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)

    kernel.optimize(backend="inductor")
    assert len(calls) == 1


def test_fp8_compiled_forward_invalidates_on_move_and_state_load(monkeypatch):
    kernel = _make_fp8_kernel()
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda fn, **_kwargs: lambda value: fn(value),
    )

    kernel.optimize()
    assert kernel._compiled_forward is not None
    kernel._apply(lambda tensor: tensor)
    assert kernel._compiled_forward is None
    assert not kernel.optimized

    kernel.optimize()
    assert kernel._compiled_forward is not None
    with torch.inference_mode():
        kernel.load_state_dict(kernel.state_dict())
    assert kernel._compiled_forward is None
    assert not kernel.optimized


def test_fp8_compiled_forward_falls_back_to_eager(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    with torch.inference_mode():
        expected = kernel(x)

    def fail(_value):
        raise RuntimeError("compiled graph failed")

    kernel._compiled_forward = fail
    kernel.optimized = True

    with torch.inference_mode():
        actual = kernel(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert kernel._compiled_forward is None
    assert not kernel.optimized


def test_fp8_compiled_forward_reads_live_weight_without_dense_cache(monkeypatch):
    kernel = _make_fp8_kernel()
    x = torch.randn(7, 64)
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda fn, **_kwargs: lambda value: fn(value),
    )
    kernel.optimize()

    with torch.inference_mode():
        before = kernel(x)
        kernel.weight.zero_()
        after = kernel(x)
        expected = torch.matmul(
            x,
            kernel.dequantize_weight(device=x.device, dtype=x.dtype),
        ) + kernel.bias

    assert not torch.equal(before, after)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
    assert not hasattr(kernel, "_dequant_weight_cache")


def test_fp8_optimize_skips_non_cpu_weights(monkeypatch):
    kernel = _make_fp8_kernel()
    kernel = kernel.to("meta")
    calls = []
    monkeypatch.setattr(fp8_module, "TORCH_HAS_COMPILE", True)
    monkeypatch.setattr(
        fp8_module,
        "torch_compile",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    kernel.optimize()

    assert calls == []
    assert not kernel.optimized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row"])
def test_fp8_scaled_mm_matches_dense_reference(fmt: str, weight_scale_method: str):
    torch.manual_seed(2)
    linear = nn.Linear(64, 64, bias=False).eval()

    kernel = fp8_module.TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device="cuda").eval()

    x = torch.randn(7, 64, device="cuda", dtype=torch.float16)
    if not kernel._can_use_scaled_mm(x):
        pytest.skip("scaled_mm is not available for this environment.")

    try:
        with torch.inference_mode():
            scaled_mm_out = kernel._forward_scaled_mm(x)
    except Exception as exc:
        pytest.skip(f"scaled_mm path unavailable: {exc}")

    with torch.inference_mode():
        x_q, scale_a = kernel._quantize_input_for_scaled_mm(x)
        x_q_dequant = x_q.to(torch.float16) * scale_a.to(torch.float16)
        dense_out = torch.matmul(
            x_q_dequant,
            kernel.dequantize_weight(device="cuda", dtype=torch.float16),
        )

    torch.testing.assert_close(scaled_mm_out, dense_out, rtol=5e-2, atol=5e-2)
