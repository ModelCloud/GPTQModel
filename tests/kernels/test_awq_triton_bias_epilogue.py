# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGemmTritonFn, AwqGEMMTritonLinear


def _packed_unit_weight(device):
    # Every AWQ nibble is one; construct the oracle weight independently.
    k = n = 128
    qweight = torch.full((k, n // 8), 0x11111111, dtype=torch.int32, device=device)
    qzeros = torch.zeros((1, n // 8), dtype=torch.int32, device=device)
    scales = torch.ones((1, n), dtype=torch.float16, device=device)
    return qweight, qzeros, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp32_accum", [False, True])
@pytest.mark.parametrize("shape", [(1, 128), (3, 43, 128)])
def test_awq_triton_bias_is_added_before_fp16_rounding_and_has_gradient(shape, fp32_accum):
    pytest.importorskip("triton")
    qweight, qzeros, scales = _packed_unit_weight("cuda")
    x = torch.full(shape, 516, dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.full((128,), -1024, dtype=torch.float16, device="cuda", requires_grad=True)
    # Legacy selects fused Triton for M=1 and dense dequantization for M=129.
    actual = AwqGemmTritonFn.apply(
        x, qweight, qzeros, scales, 4, 128, bias, 128,
        {"schedule_mode": "legacy", "fp32_accum": fp32_accum},
    )

    reference_x = x.detach().float().requires_grad_()
    reference_bias = bias.detach().float().requires_grad_()
    weight = torch.ones((128, 128), device="cuda", dtype=torch.float32)
    reference = (
        torch.matmul(reference_x.reshape(-1, 128), weight).reshape(actual.shape)
        + reference_bias
    ).half()
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual[0].flatten()[0].item() == 65024

    grad = torch.ones_like(actual)
    actual.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(x.grad, reference_x.grad.half(), rtol=0, atol=0)
    torch.testing.assert_close(bias.grad, reference_bias.grad.half(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_triton_empty_rows_return_zero_bias_gradient():
    qweight, qzeros, scales = _packed_unit_weight("cuda")
    x = torch.empty((2, 0, 128), dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.ones((128,), dtype=torch.float16, device="cuda", requires_grad=True)
    output = AwqGemmTritonFn.apply(x, qweight, qzeros, scales, 4, 128, bias, 128)
    assert output.shape == (2, 0, 128)
    output.sum().backward()
    assert x.grad.shape == x.shape
    torch.testing.assert_close(bias.grad, torch.zeros_like(bias), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 129])
def test_awq_triton_module_repeated_biased_forward_keeps_packed_weights(rows):
    pytest.importorskip("triton")
    qweight, qzeros, scales = _packed_unit_weight("cuda")
    module = AwqGEMMTritonLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=True,
        fp32_accum=False, register_buffers=True,
    ).cuda().eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.fill_(-1024)
    module.post_init()
    x = torch.full((rows, 128), 516, device="cuda", dtype=torch.float16)

    for _ in range(2):
        output = module(x)
        torch.testing.assert_close(output, torch.full_like(output, 65024), rtol=0, atol=0)
        del output
        persistent = list(module.buffers()) + list(module.parameters())
        persistent += [value for value in vars(module).values() if isinstance(value, torch.Tensor)]
        assert not any(t.shape == (128, 128) for t in persistent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 129])
@pytest.mark.parametrize("grad_target", ["input", "bias"])
def test_awq_triton_eval_preserves_requested_gradients(dtype, rows, grad_target):
    qweight, qzeros, scales = _packed_unit_weight("cuda")
    module = AwqGEMMTritonLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=True,
        schedule_mode="legacy", register_buffers=True,
    ).cuda().eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.fill_(1)
    module.post_init()
    module.bias.requires_grad_(grad_target == "bias")
    x = torch.full((rows, 128), 0.25, device="cuda", dtype=dtype,
                   requires_grad=grad_target == "input")

    out = module(x)
    oracle = (x.detach().double() @ torch.ones((128, 128), device="cuda", dtype=torch.float64)
              + module.bias.detach().double()).to(dtype)
    assert out.requires_grad
    torch.testing.assert_close(out, oracle, rtol=2e-3, atol=2e-3)
    out.sum().backward()
    if grad_target == "input":
        torch.testing.assert_close(x.grad, torch.full_like(x, 128), rtol=0, atol=0)
    else:
        torch.testing.assert_close(module.bias.grad, torch.full_like(module.bias, rows), rtol=0, atol=0)

    # Ordinary evaluation with no requested gradient still uses inference mode.
    module.bias.requires_grad_(False)
    plain = module(x.detach())
    assert not plain.requires_grad and plain.dtype == dtype
    if dtype == torch.float16:
        assert plain.is_inference()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_triton_eval_adapter_can_update_output():
    qweight, qzeros, scales = _packed_unit_weight("cuda")
    module = AwqGEMMTritonLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=False,
        schedule_mode="legacy", register_buffers=True,
    ).cuda().eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.post_init()
    a = torch.full((128, 1), 0.01, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.full((1, 128), 0.5, device="cuda", dtype=torch.float16, requires_grad=True)
    module.adapter = Lora(rank=1, lora_A=a, lora_B=b)
    x = torch.full((1, 128), 0.25, device="cuda", dtype=torch.float16, requires_grad=True)

    out = module(x)
    expected = (x.double() @ torch.ones((128, 128), device="cuda", dtype=torch.float64)
                + (x.double() @ a.double()) @ b.double()).half()
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    out.sum().backward()
    assert a.grad is not None and b.grad is not None
    expected_x_grad = (torch.ones((1, 128), device="cuda", dtype=torch.float64)
                       @ (torch.ones((128, 128), device="cuda", dtype=torch.float64)
                          + a.double() @ b.double()).T).half()
    torch.testing.assert_close(x.grad, expected_x_grad, rtol=2e-3, atol=2e-3)
