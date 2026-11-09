# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


import weakref
from contextlib import nullcontext

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


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
    module.pack_calibrated(
        linear,
        torch.zeros((1, 4), dtype=torch.float8_e4m3fn),
        torch.ones((1,), dtype=torch.float32),
    )

    output = module(torch.zeros((2, 4), dtype=torch.float32))

    assert module.bias.dtype == torch.float32
    assert torch.equal(output, torch.full((2, 1), 70000.0))


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
    module.pack_original(linear, scales=None, zeros=None)
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
def test_fp8_gpu_dequant_fallback_keeps_bias_wide(dtype, input_value):
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
    module.pack_original(linear, scales=None, zeros=None)
    module = module.to("cuda")
    x = torch.full((2, 1), input_value, device="cuda", dtype=dtype)
    decoded = module.weight.float() / module.weight_scale_inv.float()[:, None]
    oracle = (x.float() @ decoded.T + module.bias.float()).to(dtype)

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


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
    )
    module.pack_original(linear, scales=None, zeros=None)
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
