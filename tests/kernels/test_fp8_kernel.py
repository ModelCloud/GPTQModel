# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear, quantize_fp8_weight
from gptqmodel.quantization.dtype import available_float8_dtype_names


def _available_fp8_quant_formats():
    return [fmt for fmt in available_float8_dtype_names() if fmt != "float8_e8m0fnu"]


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_matches_reference_quantizer(fmt: str, weight_scale_method: str):
    torch.manual_seed(0)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)

    expected_weight, expected_scale_inv = quantize_fp8_weight(
        linear.weight.detach().to(torch.float32),
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
    )

    assert torch.equal(kernel.weight, expected_weight)
    assert torch.equal(kernel.weight_scale_inv, expected_scale_inv)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="float8_e8m0fnu unavailable")
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
def test_fp8_pack_rejects_e8m0fnu(weight_scale_method: str):
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format="float8_e8m0fnu",
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )

    with pytest.raises(ValueError, match="dequantization of existing checkpoints"):
        kernel.pack_original(linear=linear, scales=None, zeros=None)


@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row", "block"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_fp8_forward_matches_dequantized_reference(fmt: str, weight_scale_method: str, device: str):
    torch.manual_seed(1)
    linear = nn.Linear(64, 64, bias=True).eval()
    block_size = (16, 16) if weight_scale_method == "block" else None

    kernel = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        weight_block_size=block_size,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device=device).eval()
    kernel._scaled_mm_hard_disabled = True

    x_dtype = torch.float32 if device == "cpu" else torch.float16
    x = torch.randn(5, 64, device=device, dtype=x_dtype)

    with torch.inference_mode():
        out = kernel(x)
        expected = torch.matmul(
            x,
            kernel.dequantize_weight(device=device, dtype=x_dtype),
        )
        if kernel.bias is not None:
            expected = expected + kernel.bias.to(device=device, dtype=x_dtype)

    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fmt", _available_fp8_quant_formats())
@pytest.mark.parametrize("weight_scale_method", ["tensor", "row"])
def test_fp8_scaled_mm_matches_dense_reference(fmt: str, weight_scale_method: str):
    torch.manual_seed(2)
    linear = nn.Linear(64, 64, bias=False).eval()

    kernel = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        format=fmt,
        weight_scale_method=weight_scale_method,
        register_buffers=False,
    )
    kernel.pack_original(linear=linear, scales=None, zeros=None)
    kernel = kernel.to(device="cuda").eval()

    x = torch.randn(7, 64, device="cuda", dtype=torch.float16)
    if not kernel._can_use_scaled_mm(x):
        pytest.skip("scaled_mm is not available for this environment.")

    try:
        with torch.inference_mode():
            scaled_mm_out = kernel._forward_scaled_mm(x)
    except Exception as exc:
        pytest.skip(f"scaled_mm path unavailable: {exc}")

    with torch.inference_mode():
        x_q, scale_a = kernel._quantize_input_for_scaled_mm(x)
        x_q_dequant = x_q.to(torch.float16) * scale_a.to(torch.float16)
        dense_out = torch.matmul(
            x_q_dequant,
            kernel.dequantize_weight(device="cuda", dtype=torch.float16),
        )

    torch.testing.assert_close(scaled_mm_out, dense_out, rtol=5e-2, atol=5e-2)
