# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Wrapper/autograd contracts with the optional native AWQ extension stubbed."""

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import gemm_awq


def _module(monkeypatch, *, bias=True):
    # The host may lack nvcc; only replace the extension boundary. The real
    # module and its custom autograd Function remain under test.
    monkeypatch.setattr(
        gemm_awq.AwqGEMMLinear, "cached_validate_once", classmethod(lambda cls: (True, None)),
    )
    weight = torch.ones((128, 128), device="cuda", dtype=torch.float16)
    monkeypatch.setattr(
        gemm_awq, "_awq_cuda_gemm_forward",
        lambda x, *_args, **_kwargs: x @ weight,
    )
    monkeypatch.setattr(
        gemm_awq, "awq_dequantize_weights",
        lambda *_args, **_kwargs: weight,
    )
    module = gemm_awq.AwqGEMMLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=bias, register_buffers=True,
    ).cuda().eval()
    module.qweight.fill_(0x11111111)
    module.qzeros.zero_()
    module.scales.fill_(1)
    if bias:
        module.bias.fill_(1)
    module.post_init()
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grad_target", ["input", "bias"])
def test_awq_native_eval_preserves_requested_gradients(monkeypatch, grad_target):
    module = _module(monkeypatch)
    module.bias.requires_grad_(grad_target == "bias")
    x = torch.full((2, 128), 0.25, device="cuda", dtype=torch.float16,
                   requires_grad=grad_target == "input")

    out = module(x)
    oracle = (x.detach().double() @ torch.ones((128, 128), device="cuda", dtype=torch.float64)
              + module.bias.detach().double()).half()
    assert out.requires_grad
    torch.testing.assert_close(out, oracle, rtol=0, atol=0)
    out.sum().backward()
    if grad_target == "input":
        torch.testing.assert_close(x.grad, torch.full_like(x, 128), rtol=0, atol=0)
    else:
        torch.testing.assert_close(module.bias.grad, torch.full_like(module.bias, 2), rtol=0, atol=0)

    module.bias.requires_grad_(False)
    assert module(x.detach()).is_inference()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_native_empty_rows_return_zero_bias_gradient(monkeypatch):
    module = _module(monkeypatch)
    module.bias.requires_grad_(True)
    x = torch.empty((2, 0, 128), device="cuda", dtype=torch.float16, requires_grad=True)
    out = module(x)
    out.sum().backward()
    assert x.grad.shape == x.shape
    torch.testing.assert_close(module.bias.grad, torch.zeros_like(module.bias), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_native_eval_adapter_can_update_output(monkeypatch):
    module = _module(monkeypatch, bias=False)
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
