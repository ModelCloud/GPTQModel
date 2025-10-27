import torch

import pytest

from gptqmodel.quantization.dtype import (
    dequantize_f8_e4m3,
    device_supports_native_fp8,
)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_basic_conversion():
    values = torch.linspace(-1, 1, steps=8, dtype=torch.float32)
    tensor = values.to(torch.float8_e4m3fn)

    result = dequantize_f8_e4m3(tensor)

    assert result.dtype is torch.bfloat16
    assert torch.equal(result, tensor.to(torch.bfloat16))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_scale_inv():
    src = torch.randn(4, 6, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.arange(1, 5, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=0)

    expected = (fp8.to(torch.bfloat16) / scale_inv.view(-1, 1).to(torch.bfloat16)).to(torch.bfloat16)
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_scale_axis_one():
    src = torch.randn(3, 5, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale = torch.linspace(0.5, 1.5, steps=5, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale=scale, axis=1)

    expected = (fp8.to(torch.bfloat16) * scale.view(1, -1)).to(torch.bfloat16)
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_with_fractional_scale_inv():
    src = torch.randn(4, 4, dtype=torch.float32)
    fp8 = src.to(torch.float8_e4m3fn)
    scale_inv = torch.full((4,), 1 / 4, dtype=torch.float32)

    got = dequantize_f8_e4m3(fp8, scale_inv=scale_inv, axis=0)

    expected = (fp8.to(torch.bfloat16) * scale_inv.view(-1, 1).to(torch.bfloat16)).to(torch.bfloat16)
    assert torch.equal(got, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_dequantize_f8_e4m3_raises_on_both_scale_and_inverse():
    tensor = torch.zeros(2, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        dequantize_f8_e4m3(tensor, scale=torch.ones(2), scale_inv=torch.ones(2))


def test_device_supports_native_fp8_reports_capability(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (9, 0))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is True

    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 0))
    assert device_supports_native_fp8(torch.device("cuda", 0)) is False
