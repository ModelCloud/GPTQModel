# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FP8 source validation and finite-boundary parity against independent Torch math."""

import pytest
import torch
from torch import nn

import gptqmodel.nn_modules.qlinear.fp8 as fp8_module


_FORMATS = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
)


def _kwargs(method):
    return {
        "weight_scale_method": method,
        "weight_block_size": (1, 16) if method == "block" else None,
    }


@pytest.mark.parametrize("format", _FORMATS)
@pytest.mark.parametrize("method", ("tensor", "row", "block"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("invalid", (float("inf"), -float("inf"), float("nan")))
def test_fp8_quantizer_rejects_nonfinite_source_before_packing(format, method, dtype, invalid):
    if not hasattr(torch, format):
        pytest.skip(f"{format} unavailable")
    weight = torch.zeros((2, 16), dtype=dtype)
    weight[0, 5] = invalid
    assert not bool(torch.isfinite(weight).all())  # Independent input contract.
    with pytest.raises(ValueError, match="finite source weights"):
        fp8_module.quantize_fp8_weight(weight, format=format, **_kwargs(method))


def test_fp8_quantizer_rejects_finite_float64_outside_float32_range():
    weight = torch.tensor([[torch.finfo(torch.float64).max, 0.0]], dtype=torch.float64)
    assert bool(torch.isfinite(weight).all())
    with pytest.raises(ValueError, match="representable as float32"):
        fp8_module.quantize_fp8_weight(weight)


@pytest.mark.parametrize("format", _FORMATS)
@pytest.mark.parametrize("method", ("tensor", "row", "block"))
def test_fp8_valid_code_boundaries_match_independent_torch_oracle(format, method):
    if not hasattr(torch, format):
        pytest.skip(f"{format} unavailable")
    fp8_dtype = getattr(torch, format)
    maximum = torch.finfo(fp8_dtype).max
    step = 0.125 if "e4m3" in format else 0.25
    half_step = torch.tensor(1 + step / 2, dtype=torch.float32)
    below = torch.nextafter(half_step, torch.tensor(-torch.inf))
    above = torch.nextafter(half_step, torch.tensor(torch.inf))
    row = torch.tensor([
        maximum, -maximum, 0.0, -0.0, 1.0, -1.0,
        below.item(), half_step.item(), above.item(),
        -below.item(), -half_step.item(), -above.item(),
        maximum / 2, -maximum / 2, maximum, -maximum,
    ], dtype=torch.float32)
    weight = row.repeat(2, 1)
    # Each tensor/row/block has a maximum exactly equal to the FP8 maximum,
    # so the independent inverse scale is exactly one. Cast the complete
    # matrix to compare every stored FP8 byte, including signed zero.
    expected_codes = weight.to(fp8_dtype)
    actual_codes, scale_inv = fp8_module.quantize_fp8_weight(weight, format=format, **_kwargs(method))
    assert torch.equal(scale_inv, torch.ones_like(scale_inv))
    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.isfinite(actual_codes.float()).all()
    torch.testing.assert_close(actual_codes.float() / scale_inv.reshape(-1)[0],
                               expected_codes.float(), rtol=0, atol=0)


@pytest.mark.parametrize("format", _FORMATS)
@pytest.mark.parametrize("method", ("tensor", "row", "block"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_fp8_valid_source_extrema_keep_positive_finite_scales(format, method, dtype):
    if not hasattr(torch, format):
        pytest.skip(f"{format} unavailable")
    fp8_dtype = getattr(torch, format)
    fp8_max = torch.finfo(fp8_dtype).max
    source_max = torch.finfo(dtype).max
    row = torch.tensor([source_max, -source_max, 0.0, -0.0,
                        source_max / 4, -source_max / 4, source_max, -source_max,
                        source_max, -source_max, 0.0, -0.0,
                        source_max / 4, -source_max / 4, source_max, -source_max], dtype=dtype)
    weight = row.repeat(2, 1)
    # Independent FP64 scale oracle; cast once to the checkpoint's FP32 scale.
    expected_inv = torch.tensor(fp8_max / float(source_max), dtype=torch.float32)
    expected_codes = (weight.double() * expected_inv.double()).clamp(-fp8_max, fp8_max).float().to(fp8_dtype)

    actual_codes, scale_inv = fp8_module.quantize_fp8_weight(weight, format=format, **_kwargs(method))
    assert torch.isfinite(scale_inv).all()
    assert (scale_inv > 0).all()
    torch.testing.assert_close(scale_inv, expected_inv.expand_as(scale_inv), rtol=1e-6, atol=0)
    assert torch.equal(actual_codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.isfinite(actual_codes.float()).all()
    # Decode in FP64 so a finite FP32 source extremum cannot overflow the
    # test's reciprocal before the comparison.
    decoded_endpoint = actual_codes[0, 0].float().double() / scale_inv.reshape(-1)[0].double()
    assert abs(decoded_endpoint.item() - float(source_max)) / float(source_max) <= 1e-6


@pytest.mark.parametrize("invalid", (float("inf"), -float("inf"), float("nan")))
@pytest.mark.parametrize("path", ("torch", "mlx", "smoother"))
def test_fp8_pack_rejects_invalid_source_before_state_changes(monkeypatch, invalid, path):
    linear = nn.Linear(16, 2, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.weight[0, 5] = invalid
    module = fp8_module.TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False,
                            in_features=16, out_features=2, format="float8_e4m3fn",
                            weight_scale_method="row", register_buffers=True)
    before = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    if path == "mlx":
        monkeypatch.setattr(fp8_module, "_mlx_quantization_available", lambda: True)
        monkeypatch.setattr(fp8_module, "_MLX_FP8_MIN_ELEMENTS", 0)
        monkeypatch.setattr(fp8_module, "_quantize_fp8_weight_mlx_to_torch",
                            lambda *args, **kwargs: pytest.fail("invalid source reached MLX quantization"))
    elif path == "smoother":
        monkeypatch.setattr(fp8_module, "_apply_optional_smoother",
                            lambda *args, **kwargs: pytest.fail("invalid source reached smoother"))
    else:
        monkeypatch.setattr(fp8_module, "_mlx_quantization_available", lambda: False)

    with pytest.raises(ValueError, match="finite source weights"):
        module.pack_original(linear, scales=None, zeros=None,
                             smooth=object() if path == "smoother" else None)
    for name, tensor in before.items():
        assert torch.equal(module.state_dict()[name], tensor)
    assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())


def test_fp8_pack_valid_fp16_extrema_keep_only_quantized_storage():
    maximum = torch.finfo(torch.float16).max
    source = nn.Linear(16, 2, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([maximum, -maximum, 0.0, -0.0] * 4,
                                         dtype=torch.float16).repeat(2, 1))
    module = fp8_module.TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False,
                            in_features=16, out_features=2, format="float8_e4m3fn",
                            weight_scale_method="row", register_buffers=True)
    expected_inv = torch.tensor(torch.finfo(torch.float8_e4m3fn).max / maximum, dtype=torch.float32)
    expected_codes = (source.weight.detach().float() * expected_inv).to(torch.float8_e4m3fn)

    module.pack_original(source, scales=None, zeros=None)
    assert torch.equal(module.weight.view(torch.uint8), expected_codes.view(torch.uint8))
    torch.testing.assert_close(module.weight_scale_inv, expected_inv.expand(2), rtol=1e-6, atol=0)
    for _ in range(2):
        decoded = module.dequantize_weight(device="cpu", dtype=torch.float32).T
        oracle = expected_codes.float() / expected_inv
        torch.testing.assert_close(decoded, oracle, rtol=1e-6, atol=1e-6)
        assert torch.isfinite(decoded).all()
        assert not any(isinstance(value, torch.Tensor) for value in vars(module).values())
    assert set(module.state_dict()) == {"weight", "weight_scale_inv"}
