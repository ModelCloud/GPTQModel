# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MACaQ bf16 matmuls must use the repo's FP32-reduce helper (#4221).

A plain ``F.linear`` lets cuBLAS reduce a bf16 GEMM in bf16.  Both the Torch
fallback and the W4A16 Triton module's Torch tail must go through
``linear_fp32_reduce`` so the reduction stays wide on CUDA.
"""

import pytest
import torch

import gptqmodel.nn_modules.qlinear.macaq as macaq_module
import gptqmodel.utils.macaq_triton as macaq_triton
from gptqmodel.nn_modules.qlinear._numerics import linear_fp32_reduce
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization import macaq as macaq_core


def _make_layer(*, in_features=128, out_features=64):
    layer = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        activation_precision="w4a16",
        rotation="none",
        pre_mode="none",
        dtype=torch.bfloat16,
    )
    layer.W_packed.random_(0, 255)
    layer.W_scales.fill_(122)
    layer.post_init()
    layer.eval()
    return layer


def test_macaq_torch_forward_uses_fp32_reduce(monkeypatch):
    layer = _make_layer()
    seen = []

    def tracked(x, weight, bias=None):
        seen.append((x.dtype, None if bias is None else bias.dtype))
        return linear_fp32_reduce(x, weight, bias)

    monkeypatch.setattr(macaq_module, "linear_fp32_reduce", tracked)
    x = torch.randn(3, 128, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = layer(x)

    assert seen == [(torch.bfloat16, torch.bfloat16)]
    assert out.dtype == torch.bfloat16


def test_macaq_triton_w4a16_tail_uses_fp32_reduce(monkeypatch):
    in_features, out_features = 128, 64
    seen = []
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    def tracked(x, weight_arg, bias=None):
        seen.append(x.dtype)
        return linear_fp32_reduce(x, weight_arg, bias)

    # The Triton dequant is the only Triton launch in this path and is replaced
    # here, so the availability gate is lifted to observe the Torch tail on CPU.
    monkeypatch.setattr(macaq_triton, "TRITON_AVAILABLE", True)
    monkeypatch.setattr(macaq_triton, "dequant_w4_to_bf16", lambda packed, scales: weight)
    monkeypatch.setattr(macaq_triton, "linear_fp32_reduce", tracked)
    x = torch.randn(2, in_features, dtype=torch.bfloat16)

    out = macaq_triton.mxfp4_w4a16_linear(
        x,
        torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        torch.zeros((out_features, in_features // 32), dtype=torch.uint8),
        compute_dtype=torch.bfloat16,
    )

    assert seen == [torch.bfloat16]
    torch.testing.assert_close(out, linear_fp32_reduce(x, weight), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP32-reduce path")
def test_macaq_torch_forward_matches_fp32_reduce_on_cuda():
    layer = _make_layer(in_features=11008, out_features=64).to("cuda")
    x = torch.randn(8, 11008, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        out = layer(x)

    weight = macaq_core.dequantize_mxfp4(
        layer.W_packed, layer.W_scales, 11008, dtype=torch.bfloat16
    )
    expected = linear_fp32_reduce(x, weight, layer.bias.to(torch.bfloat16))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
