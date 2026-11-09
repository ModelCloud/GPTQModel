# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTritonLinear
from gptqmodel.quantization.awq.modules.triton.gemv import awq_gemv_triton
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq_columns


def _module(k=128, n=128, group_size=128, bias=True):
    return AwqGEMVTritonLinear(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=bias, register_buffers=True,
    ).cuda()


def _independent_dense_weight(qweight, qzeros, scales, group_size):
    # The GEMV format uses consecutive nibbles, unlike AWQ GEMM's interleave.
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int64)
    k = qweight.shape[1] * 8
    codes = ((qweight.long()[:, :, None] >> shifts) & 15).reshape(qweight.shape[0], k)
    zeros = ((qzeros.long()[:, :, None] >> shifts) & 15).reshape(qzeros.shape[0], -1)
    groups = torch.arange(k, device=qweight.device) // group_size
    return (codes.double() - zeros[:, groups].double()) * scales[:, groups].double()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 9])
@pytest.mark.parametrize("group_size", [8, 24, 96])
def test_awq_gemv_direct_non_power_of_two_groups(dtype, rows, group_size):
    torch.manual_seed(3901 + group_size + rows)
    k, n = 192, 11
    groups = k // group_size
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    zero_codes = torch.randint(0, 16, (n, ((groups + 7) // 8) * 8), device="cuda", dtype=torch.int32)
    qweight = pack_awq_columns(codes, 4)
    qzeros = pack_awq_columns(zero_codes, 4)
    scales = torch.rand((n, groups), device="cuda", dtype=torch.float16).mul_(0.02).add_(0.01)
    x = torch.randn((rows, k), device="cuda", dtype=dtype)

    wide = awq_gemv_triton(x, qweight, scales, qzeros, group_size, output_dtype=torch.float32)
    expected = x.double() @ _independent_dense_weight(qweight, qzeros, scales, group_size).T
    torch.testing.assert_close(wide, expected.float(), rtol=2e-3, atol=2e-3)

    visible = awq_gemv_triton(x, qweight, scales, qzeros, group_size)
    assert visible.dtype == dtype
    torch.testing.assert_close(visible, wide.to(dtype), rtol=0, atol=torch.finfo(dtype).eps * wide.abs().max().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", [0, 4, 12])
def test_awq_gemv_direct_rejects_unaligned_group(group_size):
    x = torch.ones((1, 192), device="cuda", dtype=torch.float16)
    qweight = torch.zeros((8, 24), device="cuda", dtype=torch.int32)
    qzeros = torch.zeros((8, 6), device="cuda", dtype=torch.int32)
    scales = torch.ones((8, 48), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="positive multiple of 8"):
        awq_gemv_triton(x, qweight, scales, qzeros, group_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 9])
def test_awq_gemv_triton_bias_cancels_large_finite_matmul(dtype, rows):
    pytest.importorskip("triton")
    module = _module().eval()
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(1)
    module.bias.fill_(-1024)
    module.post_init()
    x = torch.full((rows, 128), 516, device="cuda", dtype=dtype)

    for _ in range(2):
        actual = module(x)
        expected = (x.half().double() @ torch.ones((128, 128), device="cuda", dtype=torch.float64)
                    - 1024).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        del actual

    # The public module retains only packed weights, zeros, scales, and bias.
    persistent = list(module.buffers()) + list(module.parameters())
    persistent += [value for value in vars(module).values() if isinstance(value, torch.Tensor)]
    assert not any(t.shape == (128, 128) for t in persistent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (2, 3, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_awq_gemv_triton_training_input_and_bias_gradients(dtype, shape, with_bias):
    pytest.importorskip("triton")
    torch.manual_seed(3265)
    k, n, group_size = 128, 256, 64
    module = _module(k, n, group_size, bias=with_bias).train()
    codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    zeros = torch.zeros((n, 8), device="cuda", dtype=torch.int32)
    zeros[:, :2] = torch.randint(0, 16, (n, 2), device="cuda", dtype=torch.int32)
    module.qweight.copy_(pack_awq_columns(codes, 4))
    module.qzeros.copy_(pack_awq_columns(zeros, 4))
    module.scales.copy_(torch.rand_like(module.scales).mul_(0.15).add_(0.05))
    if with_bias:
        module.bias.copy_(torch.randn_like(module.bias).mul_(0.1))
    module.post_init()
    if with_bias:
        module.bias.requires_grad_(True)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)

    actual = module(x)
    weight = _independent_dense_weight(module.qweight, module.qzeros, module.scales, group_size)
    expected = x.half().double().reshape(-1, k) @ weight.T
    if with_bias:
        expected = expected + module.bias.double()
    expected = expected.to(dtype).reshape(actual.shape)
    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    else:
        ulp = torch.finfo(dtype).eps * expected.abs().float()
        assert torch.all((actual.float() - expected.float()).abs() <= ulp + 1e-3)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected_x_grad = (grad.double().reshape(-1, n) @ weight).to(dtype).reshape(shape)
    if dtype == torch.float16:
        torch.testing.assert_close(x.grad, expected_x_grad, rtol=2e-3, atol=2e-3)
    else:
        ulp = torch.finfo(dtype).eps * expected_x_grad.abs().float()
        assert torch.all((x.grad.float() - expected_x_grad.float()).abs() <= ulp + 1e-3)
    if with_bias:
        expected_bias_grad = grad.float().reshape(-1, n).sum(0).to(module.bias.dtype)
        torch.testing.assert_close(module.bias.grad, expected_bias_grad, rtol=0, atol=0)
    persistent = list(module.buffers()) + list(module.parameters())
    persistent += [value for value in vars(module).values() if isinstance(value, torch.Tensor)]
    assert not any(t.shape == (n, k) for t in persistent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_gemv_triton_empty_rows_keep_zero_bias_gradient():
    module = _module()
    module.bias.requires_grad_(True)
    x = torch.empty((2, 0, 128), device="cuda", dtype=torch.float16, requires_grad=True)
    out = module(x)
    assert out.shape == (2, 0, 128)
    out.sum().backward()
    assert x.grad.shape == x.shape
    torch.testing.assert_close(module.bias.grad, torch.zeros_like(module.bias), rtol=0, atol=0)
