# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# CPU cases run without an accelerator; pack_gpu cases run when CUDA/ROCm is available.
"""The GPTQ packers must reject scales lost by FP16 checkpoint storage."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import _scales_for_storage
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.quantizer import SCALE_STORAGE_FLOOR, Quantizer


def test_representable_scales_are_cast_to_fp16():
    scales = torch.tensor([[1e-3], [1.0], [4.0e4]], dtype=torch.float32)
    out = _scales_for_storage(scales, name="layer")
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), scales, rtol=1e-3), out


def test_quantizer_floors_near_dead_group_scales_for_fp16_storage():
    """A near-dead group must not produce a scale lost by FP16 packing.

    OLMo Hybrid ships full-attention ``q_proj`` groups whose largest weight is
    below one FP16 scale step. The quantizer must lift the scale so the packed
    checkpoint keeps the group instead of failing at pack time.
    """

    quantizer = Quantizer(QuantizeConfig(bits=4, group_size=128, sym=True))
    quantizer.configure(perchannel=True)

    near_dead = torch.full((4, 128), 1e-7, dtype=torch.float32)
    quantizer.find_params(near_dead, weight=True)
    assert quantizer.scale.min().item() >= SCALE_STORAGE_FLOOR
    stored = _scales_for_storage(quantizer.scale, name="near_dead")
    assert stored.min().item() >= SCALE_STORAGE_FLOOR

    # Ordinary groups keep the exact symmetric scale.
    normal = torch.linspace(-0.2, 0.2, 128, dtype=torch.float32).unsqueeze(0)
    quantizer.find_params(normal, weight=True)
    expected = (normal.amax(dim=1, keepdim=True) - normal.amin(dim=1, keepdim=True)) / 15.0
    assert torch.equal(quantizer.scale, expected)

    # Exact-zero groups keep the finite sentinel path and remain representable.
    quantizer.find_params(torch.zeros((1, 128), dtype=torch.float32), weight=True)
    assert torch.isfinite(quantizer.scale).all()
    _scales_for_storage(quantizer.scale, name="zero_group")


def test_largest_representable_scale_is_accepted():
    out = _scales_for_storage(torch.tensor([[65504.0]]), name="edge")
    assert out.item() == 65504.0
    # A finite source just above the largest representable value still rounds
    # to 65504, so the stored result remains valid.
    assert _scales_for_storage(torch.tensor(65505.0), name="edge").item() == 65504.0


def test_finite_scale_at_cast_overflow_midpoint_is_rejected():
    midpoint = torch.tensor(65520.0, dtype=torch.float32)
    below = torch.nextafter(midpoint, torch.tensor(float("-inf")))
    above = torch.nextafter(midpoint, torch.tensor(float("inf")))
    assert _scales_for_storage(below, name="edge").item() == 65504.0
    for value in (midpoint, above, -midpoint):
        with pytest.raises(ValueError, match="edge.*overflow"):
            _scales_for_storage(value, name="edge")


def test_nonzero_scale_that_rounds_to_zero_is_rejected():
    midpoint = torch.tensor(2.0**-25, dtype=torch.float32)
    below = torch.nextafter(midpoint, torch.tensor(0.0))
    above = torch.nextafter(midpoint, torch.tensor(float("inf")))
    assert _scales_for_storage(above, name="tiny").item() == 2.0**-24
    assert _scales_for_storage(torch.tensor(0.0), name="zero").item() == 0.0
    for value in (below, midpoint, -midpoint):
        with pytest.raises(ValueError, match="tiny.*underflow"):
            _scales_for_storage(value, name="tiny")


def test_scale_above_fp16_range_is_rejected():
    # amax = 2e6 with 4-bit scales is 2e6 / 15 = 1.33e5.
    with pytest.raises(ValueError, match="overflow"):
        _scales_for_storage(torch.tensor([[2e6 / 15.0]]), name="layer.q_proj.scales")


def test_infinite_scale_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        _scales_for_storage(torch.tensor([[float("inf")]]), name="layer")


def test_nan_scale_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        _scales_for_storage(torch.tensor([[float("nan")]]), name="layer")


def test_error_names_the_offending_tensor_and_magnitude():
    with pytest.raises(ValueError, match="layer.q_proj.scales") as info:
        _scales_for_storage(torch.tensor([[2e6 / 15.0]]), name="layer.q_proj.scales")
    message = str(info.value)
    assert "1.333e+05" in message
    assert "65504" in message


def test_mixed_invalid_scales_report_each_failure_without_nan_magnitude():
    scales = torch.tensor([float("nan"), 65520.0, 1e-40])
    with pytest.raises(ValueError, match="layer.scales") as info:
        _scales_for_storage(scales, name="layer.scales")
    message = str(info.value)
    assert "source nonfinite=1" in message
    assert "cast overflow=1" in message
    assert "cast underflow=1" in message
    assert "largest finite source magnitude is 6.552e+04" in message


@pytest.mark.parametrize("scale", [65520.0, 1e-40])
@pytest.mark.parametrize("method", ["pack_block", "pack_original", "pack_gpu"])
def test_packer_rejects_scale_lost_at_fp16_storage_boundary(method, scale):
    if method == "pack_gpu" and not torch.cuda.is_available():
        pytest.skip("pack_gpu requires a CUDA or ROCm device")
    device = "cuda" if method == "pack_gpu" else "cpu"
    source = torch.nn.Linear(32, 32, bias=False, dtype=torch.float32, device=device)
    module = TorchLinear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=32, out_features=32, bias=False, name="layer.q_proj",
    )
    scales = torch.full((32, 1), scale, dtype=torch.float32, device=device)
    zeros = torch.zeros_like(scales)
    g_idx = torch.zeros(32, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="layer.q_proj.scales"):
        getattr(module, method)(source, scales, zeros, g_idx)


@pytest.mark.parametrize("method", ["pack_block", "pack_original", "pack_gpu"])
def test_packers_preserve_source_bfloat16_bias_bits(method):
    if method == "pack_gpu" and not torch.cuda.is_available():
        pytest.skip("pack_gpu requires a CUDA or ROCm device")
    device = "cuda" if method == "pack_gpu" else "cpu"
    source = torch.nn.Linear(
        32,
        32,
        bias=True,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.no_grad():
        source.weight.zero_()
        source.bias.fill_(100000.0)

    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=True,
        name="layer.q_proj",
    )
    scales = torch.ones((32, 1), dtype=torch.float32, device=device)
    zeros = torch.zeros_like(scales)
    g_idx = torch.zeros(32, dtype=torch.int32, device=device)

    getattr(module, method)(source, scales, zeros, g_idx)

    assert module.bias.dtype is torch.bfloat16
    assert torch.isfinite(module.bias).all()
    torch.testing.assert_close(
        module.bias,
        source.bias.detach().cpu(),
        rtol=0,
        atol=0,
    )


def _packed_words(codes, bits):
    """Independent Python-integer writer, with no production packing helpers."""
    packed = []
    for row in codes.cpu().tolist():
        words = []
        for start in range(0, len(row), 32 // bits):
            word = sum(int(code) << (bits * i) for i, code in enumerate(row[start:start + 32 // bits]))
            words.append(word - 2**32 if word >= 2**31 else word)
        packed.append(words)
    return torch.tensor(packed, dtype=torch.int32)


_SCALE_DEVICES = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


@pytest.mark.parametrize("device", _SCALE_DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("mse", [0.0, 2.4])
def test_near_dead_quantizer_codes_and_packed_bytes(device, dtype, bits, sym, mse):
    # Exact half-steps, their input-dtype neighbors, both signs, signed zero,
    # and endpoints. All candidate ranges lie below the FP16 scale floor.
    half = torch.tensor([-0.5, 0.5], dtype=dtype, device=device) * 2.0**-24
    samples = torch.cat((
        torch.nextafter(half, torch.full_like(half, -torch.inf)),
        half,
        torch.nextafter(half, torch.full_like(half, torch.inf)),
        torch.tensor([-0.75, -0.0, 0.0, 0.75], dtype=dtype, device=device) * 2.0**-24,
    ))
    row = samples.repeat(13)[:128]
    weights = torch.stack((row, row.abs(), -row.abs())).repeat(11, 1)[:32]
    qcfg = QuantizeConfig(bits=bits, group_size=128, sym=sym, mse=mse, offload_to_disk=False)
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True)
    quantizer.find_params(weights, weight=True)

    # FP64 oracle from the actual input values: neither the selected parameters
    # nor dequantized weights from the implementation supply expected values.
    reference = weights.double()
    scale = torch.full((32, 1), 2.0**-24, dtype=torch.float64, device=device)
    zero = (torch.full_like(scale, 2**(bits - 1)) if sym else
            (-reference.amin(1, keepdim=True).clamp_max(0) / scale).round())
    codes = (reference.div(scale).round() + zero).clamp(0, 2**bits - 1).long()
    expected = (codes - zero) * scale
    torch.testing.assert_close(quantizer.scale.double(), scale, rtol=0, atol=0)
    torch.testing.assert_close(quantizer.zero.double(), zero, rtol=0, atol=0)
    actual = quantizer.quantize(weights)
    torch.testing.assert_close(actual.double(), expected, rtol=0, atol=0)

    source = torch.nn.Linear(128, 32, bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        source.weight.copy_(actual)
    packed = TorchLinear(bits=bits, group_size=128, sym=sym, desc_act=False,
                         in_features=128, out_features=32, bias=False)
    groups = torch.zeros(128, dtype=torch.int32, device=device)
    if device == "cpu":
        packed.pack_block(source, quantizer.scale, quantizer.zero, groups, workers=1)
    else:
        packed.pack_gpu(source, quantizer.scale, quantizer.zero, groups, device=torch.device(device))
    assert torch.equal(packed.qweight.cpu(), _packed_words(codes, bits).T)
    assert torch.equal(packed.qzeros.cpu(), _packed_words(zero.long().T, bits))
    torch.testing.assert_close(packed.scales.double().cpu(), scale.T.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("device", _SCALE_DEVICES)
@pytest.mark.parametrize("mse", [0.0, 2.4])
def test_positive_weight_range_that_arithmetically_underflows_is_bounded(device, mse):
    quantizer = Quantizer(QuantizeConfig(bits=8, group_size=128, mse=mse, offload_to_disk=False))
    quantizer.configure(perchannel=True)
    weights = torch.full((4, 128), 1e-44, device=device)
    quantizer.find_params(weights, weight=True)
    assert torch.all(quantizer.scale == 2.0**-24)
    assert torch.equal(quantizer.quantize(weights), torch.zeros_like(weights))
    _scales_for_storage(quantizer.scale, name="underflow.scales")


@pytest.mark.parametrize("device", _SCALE_DEVICES)
def test_activation_scale_is_not_floored_by_weight_storage(device):
    for mse in (0.0, 2.4):
        quantizer = Quantizer(QuantizeConfig(bits=4, group_size=128, mse=mse, offload_to_disk=False))
        quantizer.configure(perchannel=True)
        quantizer.find_params(torch.full((128, 4), 1e-7, device=device), weight=False)
        assert torch.all(quantizer.scale < 2.0**-24)
