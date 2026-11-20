# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import weakref
from contextlib import nullcontext

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from tests.fp8_test_utils import install_fp8_payload, pack_fp8_module


def test_fp8_packing_preserves_fp32_bias_range():
    linear = torch.nn.Linear(4, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(70000.0)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=4,
        out_features=1,
        bias=True,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    install_fp8_payload(
        module,
        linear,
        torch.zeros((1, 4), dtype=torch.float8_e4m3fn),
        torch.ones((1,), dtype=torch.float32),
    )

    output = module(torch.zeros((2, 4), dtype=torch.float32))

    assert module.bias.dtype == torch.float32
    assert torch.equal(output, torch.full((2, 1), 70000.0))


def test_fp8_bias_free_fallback_keeps_decoded_weight_range():
    """A finite cancellation must not NaN because the decoded weight narrowed."""
    linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(65536.0)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2,
        out_features=1,
        bias=False,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    pack_fp8_module(module, linear)

    x = torch.tensor([[1.0, -1.0]], dtype=torch.float16)
    decoded = module.weight.float() / module.weight_scale_inv.float().view(-1, 1)
    expected = (x.double() @ decoded.double().T).half()

    actual = module(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,input_value", [
    (torch.float16, 70.0),
    (torch.bfloat16, 68.5),
])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("cpu_autocast", [False, True])
def test_fp8_cpu_bias_cancels_before_output_rounding(
    monkeypatch, dtype, input_value, sign, compiled, cpu_autocast
):
    linear = torch.nn.Linear(1, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(-1000.0 * sign)
        linear.bias.fill_(70000.0 * sign)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=1,
        out_features=1,
        bias=True,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    pack_fp8_module(module, linear)
    x = torch.full((2, 1), input_value, dtype=dtype)

    # Decode the stored payload independently in FP32; include the bias before
    # rounding the visible result to the activation dtype.
    decoded = module.weight.float() / module.weight_scale_inv.float()[:, None]
    oracle = (x.float() @ decoded.T + module.bias.float()).to(dtype)
    packed_bytes = module.weight.view(torch.uint8).clone()
    dequant_refs = []
    dequantize = module.dequantize_weight

    def tracked_dequantize(*args, **kwargs):
        weight = dequantize(*args, **kwargs)
        dequant_refs.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize_weight", tracked_dequantize)
    if compiled:
        module._compiled_forward = lambda value, bias=None: module._forward_dequant_matmul(value, bias=bias)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if cpu_autocast else nullcontext()
    with torch.inference_mode(), context:
        first = module(x)
        second = module(x)

    assert first.dtype == dtype
    torch.testing.assert_close(first, oracle, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert module.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(module.weight.view(torch.uint8), packed_bytes)
    assert module.weight_scale_inv.dtype == torch.float32
    assert module.bias.dtype == torch.float32
    assert all(ref() is None for ref in dequant_refs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm runtime required")
@pytest.mark.parametrize("dtype,input_value", [
    (torch.float16, 70.0),
    (torch.bfloat16, 68.5),
])
def test_fp8_gpu_native_gemm_keeps_bias_wide(dtype, input_value):
    """#3737: a bias wider than the activation dtype must reach the epilogue.

    #4716 keeps this layer on the native GEMM: the decoded weight is narrowed
    to the activation dtype and the 70000 bias is added in the FP32 epilogue.
    The old assertion compared against an un-narrowed FP32 dequant, i.e. the
    fallback path that #4716 deliberately removed for this layer.
    """
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    linear = torch.nn.Linear(1, 1, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(-1000.0)
        linear.bias.fill_(70000.0)
    module = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=1, out_features=1, bias=True,
        register_buffers=False, format="float8_e4m3fn",
    )
    pack_fp8_module(module, linear)
    module = module.to("cuda")
    x = torch.full((2, 1), input_value, device="cuda", dtype=dtype)
    decoded = module.weight.float() / module.weight_scale_inv.float()[:, None]
    # The native path narrows the decoded weight to the activation dtype.
    oracle = (x.float() @ decoded.to(dtype).float().T + module.bias.float()).to(dtype)

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == dtype
    # Rounding the decoded weight to the activation dtype moves |x.w| by at
    # most one ulp (finfo.eps) of that dtype: 2^-10 for fp16, 2^-7 for bf16.
    # A bias narrowed to fp16 overflows to inf; a dropped bias leaves
    # ~|x.w| = 68500-70000, far outside this bound.
    bound = float(torch.finfo(dtype).eps) * abs(float(input_value) * -1000.0)
    assert module.bias.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle, rtol=0, atol=bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm runtime required")
@pytest.mark.parametrize("dtype,input_value", [
    (torch.float16, 35008.0),
    (torch.bfloat16, 35008.0),
])
def test_fp8_gpu_scaled_mm_adds_fp32_bias_before_rounding(dtype, input_value):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    linear = torch.nn.Linear(16, 16, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = -2.0
        linear.bias.zero_()
        linear.bias[0] = 70000.0
    module = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=16, out_features=16, bias=True,
        register_buffers=False, format="float8_e4m3fn",
        weight_scale_method="tensor",
        activation_quant_method="per_token",
    )
    pack_fp8_module(module, linear)
    module = module.to("cuda")
    x = torch.zeros((1, 16), device="cuda", dtype=dtype)
    x[0, 0] = input_value
    if not module._can_use_scaled_mm(x):
        pytest.skip("Native FP8 scaled matmul unavailable")
    try:
        module._forward_scaled_mm(x)
    except Exception as exc:
        pytest.skip(f"Native FP8 scaled matmul unavailable: {exc}")

    decoded = module.weight.float() / module.weight_scale_inv.float()
    oracle = (x.float() @ decoded.T + module.bias.float()).to(dtype)
    with torch.inference_mode():
        actual = module(x)

    assert not module._scaled_mm_hard_disabled
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


def _bias_free_fp8_module(weight: torch.Tensor) -> TorchFP8Linear:
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(weight)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    pack_fp8_module(module, linear)
    return module


def test_fp8_decoded_weight_bound_tracks_reload():
    """The range decision must be recomputed after the payload changes."""
    module = _bias_free_fp8_module(torch.ones(1, 2))
    assert module._weights_fit_activation_dtype(torch.float16)

    big = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        big.weight.fill_(65536.0)
    pack_fp8_module(module, big)
    assert not module._weights_fit_activation_dtype(torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm runtime required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_bias_free_cuda_in_range_weights_keep_native_path(monkeypatch, dtype):
    """In-range decoded weights must take the code-cast + FP32-epilogue GEMM.

    Before #5146 this asserted that ``dequantize_weight`` was called with the
    activation dtype. The leaner path does not materialise a decoded weight at
    all: it casts the exact FP8 codes and applies the row scale to the FP32
    product, so no FP32 dense weight or transpose copy is made.
    """
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 arithmetic unavailable on this device")
    generator = torch.Generator().manual_seed(0)
    module = _bias_free_fp8_module(torch.randn(32, 64, generator=generator) * 0.02)
    module = module.to("cuda")
    x = torch.randn(8, 64, device="cuda", dtype=dtype)

    dequantized = []
    dequantize = module.dequantize_weight

    def tracked(*args, **kwargs):
        dequantized.append(kwargs.get("dtype"))
        return dequantize(*args, **kwargs)

    lean_calls = []
    lean = module._forward_lean_scaled_matmul

    def counted_lean(value, bias=None):
        lean_calls.append(value.dtype)
        return lean(value, bias=bias)

    monkeypatch.setattr(module, "dequantize_weight", tracked)
    monkeypatch.setattr(module, "_forward_lean_scaled_matmul", counted_lean)
    with torch.inference_mode():
        output = module(x)

    assert lean_calls == [dtype]
    assert dequantized == []
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    decoded = module.weight.double() / module.weight_scale_inv.double().view(-1, 1)
    oracle = (x.double() @ decoded.T).to(dtype)
    torch.testing.assert_close(output, oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm runtime required")
def test_fp8_bias_free_cuda_out_of_range_weights_stay_wide(monkeypatch):
    """A decoded weight above the FP16 finite range must keep the wide path."""
    module = _bias_free_fp8_module(torch.full((1, 2), 65536.0)).to("cuda")
    x = torch.tensor([[1.0, -1.0]], device="cuda", dtype=torch.float16)

    seen = []
    dequantize = module.dequantize_weight

    def tracked(*args, **kwargs):
        seen.append(kwargs.get("dtype"))
        return dequantize(*args, **kwargs)

    monkeypatch.setattr(module, "dequantize_weight", tracked)
    with torch.inference_mode():
        output = module(x)

    assert seen == [torch.float32]
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
