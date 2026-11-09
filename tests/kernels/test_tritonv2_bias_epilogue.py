# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT
from tests.test_torch_weight_storage import build_case, state_tensors


def _unit_module(with_bias):
    module = TritonV2Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=with_bias,
        format=FORMAT.GPTQ_V2, register_buffers=True,
    ).cuda().eval()
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(1)
    if with_bias:
        module.bias.fill_(-1024)
    module.post_init()
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (2, 3, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_tritonv2_bias_rounds_once_and_backward_returns_all_gradients(dtype, shape, with_bias):
    pytest.importorskip("triton")
    module = _unit_module(with_bias)
    if with_bias:
        module.bias.requires_grad_(True)
    x = torch.full(shape, 516 if with_bias else 1, dtype=dtype, device="cuda", requires_grad=True)
    first = module(x).detach()
    actual = module(x)
    torch.testing.assert_close(first, actual, rtol=0, atol=0)
    weight = torch.ones((128, 128), device="cuda", dtype=torch.float64)
    expected = x.double().reshape(-1, 128) @ weight
    if with_bias:
        expected -= 1024
    expected = expected.to(dtype).reshape(actual.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()

    actual.float().sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 128), rtol=0, atol=0)
    if with_bias:
        torch.testing.assert_close(
            module.bias.grad, torch.full_like(module.bias, x.numel() // 128), rtol=0, atol=0,
        )
    assert not any(t.shape == (128, 128) for t in state_tensors(module))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tritonv2_random_packed_bias_and_gradients_match_torch_oracle(dtype):
    pytest.importorskip("triton")
    source, dense_weight = build_case(
        bits=4, dtype=dtype, group_size=128, in_features=256, out_features=128,
    )
    module = TritonV2Linear(
        bits=4, group_size=128, sym=False, desc_act=True,
        in_features=256, out_features=128, bias=True,
        format=FORMAT.GPTQ_V2, register_buffers=True,
    )
    module.load_state_dict(source.state_dict(), strict=False)
    module.post_init()
    module = module.cuda().eval()
    torch.manual_seed(182)
    module.bias.copy_(torch.randn_like(module.bias).mul_(0.1))
    module.bias.requires_grad_(True)
    x = torch.randn((2, 3, 256), device="cuda", dtype=dtype, requires_grad=True) / 8
    x.retain_grad()
    actual = module(x)

    # The reference uses the packer's independent dense source, cast to the
    # activation dtype just as the Triton dequantizer stores its temporary.
    weight = dense_weight.to(device="cuda", dtype=dtype).double()
    expected = (x.double().reshape(-1, 256) @ weight + module.bias.double())
    expected = expected.to(dtype).reshape(actual.shape)
    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        ulp = torch.finfo(dtype).eps * expected.abs().float()
        assert torch.all((actual.float() - expected.float()).abs() <= ulp + 1e-3)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected_x_grad = (grad.double().reshape(-1, 128) @ weight.t()).to(dtype).reshape_as(x)
    if dtype == torch.float16:
        torch.testing.assert_close(x.grad, expected_x_grad, rtol=2e-3, atol=2e-3)
    else:
        ulp = torch.finfo(dtype).eps * expected_x_grad.abs().float()
        assert torch.all((x.grad.float() - expected_x_grad.float()).abs() <= ulp + 1e-3)
    expected_bias_grad = grad.float().reshape(-1, 128).sum(0).to(module.bias.dtype)
    torch.testing.assert_close(module.bias.grad, expected_bias_grad, rtol=0, atol=0)
    assert not any(t.shape == (256, 128) for t in state_tensors(module))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tritonv2_empty_rows_return_zero_bias_gradient():
    pytest.importorskip("triton")
    module = _unit_module(True)
    module.bias.requires_grad_(True)
    x = torch.empty((2, 0, 128), device="cuda", dtype=torch.float16, requires_grad=True)
    actual = module(x)
    assert actual.shape == (2, 0, 128)
    actual.sum().backward()
    assert x.grad.shape == x.shape
    torch.testing.assert_close(module.bias.grad, torch.zeros_like(module.bias), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tritonv2_direct_function_keeps_eight_argument_backward():
    pytest.importorskip("triton")
    from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction

    module = _unit_module(False)
    x = torch.ones((1, 128), device="cuda", dtype=torch.float16, requires_grad=True)
    output = QuantLinearFunction.apply(
        x, module.qweight, module.scales, module.qzeros, module.g_idx,
        module.bits, module.pack_dtype_bits, module.maxq,
    )
    output.float().sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 128), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operand", ["g_idx", "qweight", "scales", "qzeros"])
def test_tritonv2_dequant_accepts_strided_packed_operands(dtype, operand):
    from gptqmodel.nn_modules.triton_utils.dequant import dequant

    in_features, out_features = 128, 64
    # Every nibble is one; the two group scales independently specify weights 1 and 2.
    qweight = torch.full((in_features // 8, out_features), 0x11111111, device="cuda", dtype=torch.int32)
    qzeros = torch.zeros((2, out_features // 8), device="cuda", dtype=torch.int32)
    scales = torch.stack((
        torch.ones((out_features,), device="cuda", dtype=dtype),
        torch.full((out_features,), 2, device="cuda", dtype=dtype),
    ))
    g_idx = torch.arange(in_features, device="cuda", dtype=torch.int32) // 64
    arguments = {"qweight": qweight, "scales": scales, "qzeros": qzeros, "g_idx": g_idx}
    source = arguments[operand]
    if operand == "g_idx":
        storage = torch.zeros((in_features * 2,), device="cuda", dtype=source.dtype)
        view = storage[::2]
    else:
        storage = torch.empty((source.shape[0], source.shape[1] + 8), device="cuda", dtype=source.dtype)
        storage.fill_(-1 if operand == "qzeros" else 0)
        view = storage[:, :source.shape[1]]
    view.copy_(source)
    assert not view.is_contiguous()
    arguments[operand] = view

    actual = dequant(dtype, arguments["qweight"], arguments["scales"], arguments["qzeros"],
                     arguments["g_idx"], 4, 32, 15)
    expected = torch.where(g_idx[:, None] == 0, 1.0, 2.0).expand(in_features, out_features).to(dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
