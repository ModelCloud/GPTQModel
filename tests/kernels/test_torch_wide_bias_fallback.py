# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Stored FP32 bias must survive the portable Torch GPTQ epilogues."""

import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import (
    linear_needs_wide_bias,
    matmul_fp32_reduce,
)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from tests.test_torch_weight_storage import state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# The unit dot of 256 activations of 256 is 65536: legal in FP32 and BF16, but
# outside the FP16 finite range. A stored FP32 bias of -65535 must cancel it to
# 1 instead of being narrowed to FP16 ``-inf`` or a BF16 ``-65536`` rounding.
_K = 256
_N = 128
_VALUE = 256.0
_BIAS = -65535.0
_EXPECTED = 1.0


def _gptq_unit_module(dtype, *, streaming=False):
    """Quantized module whose dequantized weight is exactly one everywhere."""

    linear = torch.nn.Linear(_K, _N, bias=True)
    linear.weight.data.fill_(1.0)
    linear.bias.data.zero_()
    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=False,
        desc_act=False,
        in_features=_K,
        out_features=_N,
        bias=True,
        format=FORMAT.GPTQ_V2,
    )
    module.pack_block(
        linear,
        torch.ones((_N, 1)),
        torch.zeros((_N, 1)),
        torch.zeros(_K, dtype=torch.int32),
    )
    assert torch.equal(module.dequantize_weight(), torch.ones((_K, _N)))
    # Keep the fused GEMV/GEMM kernels off so the portable path runs.
    module._rocm_gemv_enabled = False
    module._rocm_gemm_enabled = False
    module = module.eval().to(device="cuda", dtype=dtype)
    module.register_buffer(
        "bias", torch.full((_N,), _BIAS, dtype=torch.float32, device="cuda")
    )
    if streaming:
        module.enable_streaming(True, tile_cols=64)
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 65])
def test_torch_fallback_keeps_stored_bias_wide(dtype, rows):
    """Tiny decode and the larger batches share one rounding point."""

    module = _gptq_unit_module(dtype)
    x = torch.full((rows, _K), _VALUE, device="cuda", dtype=dtype)
    x_flat = x.reshape(-1, _K)
    assert not module._can_use_rocm_gemv(x_flat)
    assert not module._can_use_rocm_gemm(x_flat)
    assert not module._should_use_streaming(x_flat)

    packed_before = {
        name: getattr(module, name).clone()
        for name in ("qweight", "qzeros", "scales", "g_idx", "bias")
    }
    references = []
    dequantize = module.dequantize_weight

    def track_dequantize(*args, **kwargs):
        weight = dequantize(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    module.dequantize_weight = track_dequantize
    owned_before = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(module)}
    oracle = (x.double() @ torch.ones((_K, _N), dtype=torch.float64, device="cuda"))
    oracle = (oracle + _BIAS).to(dtype)
    for _ in range(2):
        with torch.inference_mode():
            actual = module(x)
        assert actual.dtype is dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
        assert actual[0, 0].item() == _EXPECTED
    assert all(reference() is None for reference in references)
    for name, before in packed_before.items():
        assert torch.equal(getattr(module, name), before)

    # Inference owns no expanded weight after the call completes.
    assert all(
        tensor.untyped_storage().data_ptr() in owned_before
        for tensor in state_tensors(module)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 65])
def test_torch_streaming_keeps_stored_bias_wide(dtype, rows):
    """The streaming tiles use the same wide bias epilogue as eager."""

    module = _gptq_unit_module(dtype, streaming=True)
    x = torch.full((rows, _K), _VALUE, device="cuda", dtype=dtype)
    assert module._should_use_streaming(x.reshape(-1, _K))

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, torch.full_like(actual, _EXPECTED))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 65])
def test_matmul_fp32_reduce_keeps_stored_bias_wide(dtype, rows):
    """The shared helper must not narrow a stored wide bias on any device."""

    x = torch.full((rows, _K), _VALUE, device="cuda", dtype=dtype)
    weight = torch.ones((_K, _N), device="cuda", dtype=dtype)
    bias = torch.full((_N,), _BIAS, device="cuda", dtype=torch.float32)
    assert linear_needs_wide_bias(x, bias)
    oracle = ((x.double() @ weight.double()) + _BIAS).to(dtype)

    actual = matmul_fp32_reduce(x, weight, bias)

    assert actual.dtype is dtype
    assert actual.shape == (rows, _N)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
    assert actual[0, 0].item() == _EXPECTED


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 65])
def test_torch_fused_and_fallback_boundary(dtype, rows):
    """Every M on either side of the GEMV/GEMM switch rounds once."""

    module = _gptq_unit_module(dtype)
    module._rocm_gemv_enabled = True
    module._rocm_gemm_enabled = True
    x = torch.full((rows, _K), _VALUE, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual = module(x)

    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, torch.full_like(actual, _EXPECTED))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_fp32_reduce_representable_wide_bias_stays_accurate(dtype):
    """A representable FP32 bias is added wide and rounded once."""

    generator = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.rand((9, _K), generator=generator, device="cuda") - 0.5).to(dtype)
    weight = (torch.rand((_K, _N), generator=generator, device="cuda") - 0.5).to(dtype)
    bias = torch.rand((_N,), generator=generator, device="cuda") - 0.5
    bias = bias.to(dtype).to(torch.float32)
    assert linear_needs_wide_bias(x, bias)
    oracle = (x.double() @ weight.double() + bias.double()).to(dtype)

    actual = matmul_fp32_reduce(x, weight, bias)

    assert actual.dtype is dtype
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_fp32_reduce_keeps_wide_bias_under_autocast(dtype):
    """Autocast must not narrow the operands, or the bias cannot cancel them."""

    x = torch.full((3, _K), _VALUE, device="cuda", dtype=dtype)
    weight = torch.ones((_K, _N), device="cuda", dtype=dtype)
    bias = torch.full((_N,), _BIAS, device="cuda", dtype=torch.float32)
    oracle = ((x.double() @ weight.double()) + _BIAS).to(dtype)

    with torch.autocast("cuda", dtype=torch.float16):
        actual = matmul_fp32_reduce(x, weight, bias)

    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_fp32_reduce_wide_bias_gradients(dtype):
    """The wide epilogue still differentiates input, weight, and bias."""

    generator = torch.Generator(device="cuda").manual_seed(13)
    x_values = (torch.rand((4, 32), generator=generator, device="cuda") - 0.5).to(dtype)
    weight_values = (torch.rand((32, 48), generator=generator, device="cuda") - 0.5).to(dtype)
    bias_values = torch.full((48,), _BIAS, device="cuda", dtype=torch.float32)

    x = x_values.clone().requires_grad_()
    weight = weight_values.clone().requires_grad_()
    bias = bias_values.clone().requires_grad_()
    output = matmul_fp32_reduce(x, weight, bias)
    output.float().sum().backward()

    ref_x = x_values.clone().requires_grad_()
    ref_weight = weight_values.clone().requires_grad_()
    ref_bias = bias_values.clone().requires_grad_()
    expected = (ref_x.float() @ ref_weight.float() + ref_bias.float()).to(dtype)
    expected.float().sum().backward()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(bias.grad, ref_bias.grad, rtol=0, atol=0)
