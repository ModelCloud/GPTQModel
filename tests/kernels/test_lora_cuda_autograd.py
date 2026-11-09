# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""LoRA output composition must preserve CUDA custom-Function gradients."""

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_cuda_lora_preserves_base_and_adapter_gradients():
    linear = torch.nn.Linear(128, 128, bias=False, dtype=torch.float16)
    linear.weight.data.fill_(0.5)
    module = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=False,
        format="float8_e4m3fn", weight_scale_method="row",
    )
    module.pack(linear, None, None)
    module = module.cuda().eval()
    a = torch.full((128, 1), 0.01, device="cuda", dtype=torch.float16, requires_grad=True)
    b = torch.full((1, 128), 0.5, device="cuda", dtype=torch.float16, requires_grad=True)
    module.adapter = Lora(rank=1, lora_A=a, lora_B=b)
    x = torch.full((1, 128), 0.25, device="cuda", dtype=torch.float16, requires_grad=True)

    out = module(x)
    # Decode the stored FP8 payload independently of the module's forward.
    weight = (module.weight.double() / module.weight_scale_inv.double()[:, None]).half().double()
    oracle = (x.detach().double() @ weight.T + (x.detach().double() @ a.double()) @ b.double()).half()
    assert out.requires_grad and out.dtype == x.dtype
    torch.testing.assert_close(out, oracle, rtol=2e-3, atol=2e-3)

    out.sum().backward()
    expected_x_grad = (torch.ones((1, 128), device="cuda", dtype=torch.float64)
                       @ (weight + a.double() @ b.double()).T).half()
    torch.testing.assert_close(x.grad, expected_x_grad, rtol=2e-3, atol=2e-3)
    assert a.grad is not None and b.grad is not None
    with torch.inference_mode():
        repeated = module(x.detach())
    torch.testing.assert_close(repeated, out.detach(), rtol=0, atol=0)
    assert module.weight.dtype == torch.float8_e4m3fn
    persistent = list(module.buffers()) + list(module.parameters())
    persistent += [t for t in vars(module).values() if isinstance(t, torch.Tensor)]
    assert not any(t.shape == (128, 128) and t.dtype in (torch.float16, torch.float32)
                   for t in persistent)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_lora_extra_output_rank_is_out_of_place_and_differentiable(device):
    x = torch.tensor([[0.25, 0.5, -0.25], [0.5, 0.25, 0.75]], device=device,
                     dtype=torch.float32, requires_grad=True)
    base = torch.ones((2, 1, 4), device=device, dtype=torch.float32, requires_grad=True)
    a = torch.full((3, 2), 0.25, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.full((2, 4), 0.5, device=device, dtype=torch.float32, requires_grad=True)
    adapter = Lora(rank=2, lora_A=a, lora_B=b)

    result = adapter.apply(x=x, out=base)
    oracle = (base.detach().double().view(2, 4) + (x.detach().double() @ a.double()) @ b.double()).view(2, 1, 4)
    torch.testing.assert_close(result, oracle.float(), rtol=0, atol=0)
    assert result.data_ptr() != base.data_ptr()
    torch.testing.assert_close(base, torch.ones_like(base), rtol=0, atol=0)
    result.sum().backward()
    assert x.grad is not None and base.grad is not None
    assert a.grad is not None and b.grad is not None


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_lora_accepts_base_inference_tensor_with_adapter_gradients(device):
    x = torch.ones((1, 3), device=device, dtype=torch.float32, requires_grad=True)
    a = torch.ones((3, 1), device=device, dtype=torch.float32, requires_grad=True)
    b = torch.ones((1, 4), device=device, dtype=torch.float32, requires_grad=True)
    with torch.inference_mode():
        base = torch.full((1, 4), 2.0, device=device, dtype=torch.float32)
    result = Lora(rank=1, lora_A=a, lora_B=b).apply(x, base)
    torch.testing.assert_close(result, torch.full_like(result, 5.0), rtol=0, atol=0)
    torch.testing.assert_close(base, torch.full_like(base, 2.0), rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0), rtol=0, atol=0)
    assert a.grad is not None and b.grad is not None


def test_lora_rejects_update_that_expands_base_shape():
    x = torch.ones((2, 3))
    base = torch.ones((1, 4))
    adapter = Lora(rank=1, lora_A=torch.ones((3, 1)), lora_B=torch.ones((1, 4)))
    with pytest.raises(RuntimeError, match="expand the base output shape"):
        adapter.apply(x, base)
    torch.testing.assert_close(base, torch.ones_like(base), rtol=0, atol=0)
