# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import weakref
from contextlib import nullcontext

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


@pytest.mark.parametrize(
    "dtype,weight_value,bias_value,last_zero,wide_bias,expected_value",
    [
        (torch.float16, 128.0, -65504.0, False, False, 32.0),
        (torch.bfloat16, 1.0, -512.0, True, False, -1.0),
        (torch.float16, 128.0, -65536.0, False, True, 0.0),
    ],
)
@pytest.mark.parametrize("autocast", [False, True])
def test_packed_gptq_cpu_bias_cancels_before_output_rounding(
    monkeypatch, dtype, weight_value, bias_value, last_zero, wide_bias, expected_value,
    autocast,
):
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    source = torch.nn.Linear(512, 32, bias=True, dtype=dtype)
    with torch.no_grad():
        source.weight.fill_(weight_value)
        if last_zero:
            source.weight[:, -1] = 0
        source.bias.fill_(bias_value if not wide_bias else 0)
    module = TorchLinear(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=512,
        out_features=32,
        bias=True,
        format=FORMAT.GPTQ_V2,
    )
    scales = torch.full((32, 4), weight_value)
    zeros = torch.zeros((32, 4))
    module.pack_block(source, scales, zeros, torch.arange(512, dtype=torch.int32) // 128, workers=1)
    if wide_bias:
        module.bias = torch.full((32,), bias_value, dtype=torch.float32)
    module.eval()
    packed_before = {
        name: getattr(module, name).clone()
        for name in ("qweight", "qzeros", "scales", "g_idx", "bias")
    }
    x = torch.ones((65, 512), dtype=dtype)
    reference = (x.float() @ source.weight.float().T + bias_value).to(dtype)
    assert reference[0, 0].item() == expected_value

    temporary_refs = []
    dequantize = module.dequantize_weight

    def track_dequantize(*args, **kwargs):
        weight = dequantize(*args, **kwargs)
        temporary_refs.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(module, "dequantize_weight", track_dequantize)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
    with torch.inference_mode(), context:
        first = module(x)
        second = module(x)

    assert first.dtype == dtype
    torch.testing.assert_close(first, reference, rtol=0, atol=0)
    torch.testing.assert_close(second, reference, rtol=0, atol=0)
    assert all(ref() is None for ref in temporary_refs)
    for name, before in packed_before.items():
        torch.testing.assert_close(getattr(module, name), before, rtol=0, atol=0)
    assert set(module.state_dict()) == {"qweight", "qzeros", "scales", "g_idx", "bias"}


def test_packed_gptq_cpu_fused_bias_keeps_input_gradient(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    source = torch.nn.Linear(32, 32, bias=True, dtype=torch.float16)
    with torch.no_grad():
        source.weight.fill_(1)
        source.bias.fill_(-8)
    module = TorchLinear(
        bits=4, group_size=32, sym=False, desc_act=False,
        in_features=32, out_features=32, bias=True, format=FORMAT.GPTQ_V2,
    )
    module.pack_block(
        source,
        torch.ones((32, 1)),
        torch.zeros((32, 1)),
        torch.zeros(32, dtype=torch.int32),
        workers=1,
    )
    module.train()
    x = torch.ones((2, 32), dtype=torch.float16, requires_grad=True)

    module(x).float().sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.full_like(x, 32), rtol=0, atol=0)
