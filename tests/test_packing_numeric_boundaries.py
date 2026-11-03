# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent code/byte oracles for packing overflow and underflow boundaries."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.fp8 import quantize_fp8_weight
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear, pack_intweight
from gptqmodel.nn_modules.qlinear.gguf import _fallback_gguf_quantize
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq_columns
from gptqmodel.utils.marlin_scalar_type import scalar_types
from gptqmodel.utils.packing import hf_pack_layer
from gptqmodel.utils.packing_math import scaled_zero_offset


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device required")
    yield request.param
    if request.param == "cuda":
        torch.cuda.synchronize()


def _words(values, bits, word_bits=32):
    """Python integers provide an independent, arbitrary-precision bit writer."""
    rows = []
    for row in values.cpu().tolist():
        stream = sum((int(v) & ((1 << bits) - 1)) << (i * bits) for i, v in enumerate(row))
        words = []
        for i in range(math.ceil(len(row) * bits / word_bits)):
            word = (stream >> (i * word_bits)) & ((1 << word_bits) - 1)
            words.append(word - (1 << word_bits) if word >> (word_bits - 1) else word)
        rows.append(words)
    return torch.tensor(rows, dtype={8: torch.int8, 16: torch.int16, 32: torch.int32}[word_bits])


def _gptq_words(values, bits, planar):
    if not planar:
        return _words(values, bits)
    widths = {3: [2, 1], 5: [4, 1], 6: [4, 2], 7: [4, 2, 1]}[bits]
    blocks = []
    for start in range(0, values.shape[1], 32):
        offset = 0
        for width in widths:
            blocks.append(_words(values[:, start:start + 32] >> offset, width))
            offset += width
    return torch.cat(blocks, dim=1)


def _awq_words(values, bits):
    if bits == 4:
        values = values.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(values.shape)
    return _words(values, bits)


def _fast_words(values):
    """Map each original (output,input) coordinate to its packed halfword."""
    n, k = values.shape
    words = [[0] * k for _ in range(n // 4)]
    for out, row in enumerate(values.cpu().tolist()):
        for col, code in enumerate(row):
            within = col % 32
            transposed = ((within // 2) % 4) * 8 + (within // 8) * 2 + within % 2
            permuted = (transposed // 8) * 8 + (transposed % 8) // 2 + (transposed % 2) * 4
            index = (col % 64) // 32 * 32 + permuted
            word_col = col // 64 * 64 + out % 4 * 16 + index // 4
            words[out // 4][word_col] |= (int(code) & 15) << (4 * (index % 4))
    return torch.tensor(words, dtype=torch.int32).to(torch.int16)


def _bare(cls, bits=4):
    # Packing itself needs no native inference operator. Exercise the real
    # method and registered output buffers on both accelerator families.
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits, module.in_features, module.out_features, module.group_size = bits, 128, 64, 128
    return module


def _inputs(dtype, bits, device, case):
    linear = nn.Linear(128, 64, bias=False, dtype=dtype, device=device)
    maximum = (1 << bits) - 1
    scales = torch.ones(64, 1, dtype=dtype, device=device)
    zeros = torch.zeros(64, 1, dtype=torch.int32, device=device)
    if case == "affine_overflow":
        # Valid finite FP16 weights and scales: even the sum alone overflows.
        scales.fill_(5000 if bits == 4 else 300)
        zeros.fill_(8 if bits == 4 else 128)
        weights = torch.full((64, 128), 30000 if bits == 4 else 36000, dtype=dtype, device=device)
    elif case == "bf16_endpoint":
        scales.fill_(0.00054931640625)
        zeros.fill_(128)
        weights = ((255 - zeros) * scales).expand(64, 128).clone()
    else:
        half = torch.tensor([0.5, maximum - 0.5], dtype=dtype, device=device)
        samples = torch.cat((
            torch.tensor([-1, 0, maximum, maximum + 1], dtype=dtype, device=device),
            torch.nextafter(half, torch.full_like(half, -torch.inf)),
            half,
            torch.nextafter(half, torch.full_like(half, torch.inf)),
        ))
        weights = samples.repeat(math.ceil(64 * 128 / samples.numel()))[:64 * 128].reshape(64, 128)
    with torch.no_grad():
        linear.weight.copy_(weights)
    # Independent FP64 arithmetic from the actual stored source values.
    expected = ((linear.weight.detach().double() + zeros.double() * scales.double()) / scales.double())
    expected = expected.round().clamp(0, maximum).long().cpu()
    return linear, scales, zeros, expected


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", ["original", "block", "gpu"])
def test_gptq_packed_bytes_at_thresholds(monkeypatch, bits, dtype, method):
    device = "cuda" if method == "gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device required")
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    linear, scales, zeros, expected = _inputs(dtype, bits, device, "thresholds")
    module = TorchLinear(bits=bits, group_size=128, sym=False, desc_act=False,
                         in_features=128, out_features=64, bias=False)
    getattr(module, "pack_" + method)(linear, scales, zeros, torch.zeros(128, dtype=torch.int32, device=device))
    assert torch.equal(module.qweight.cpu(), _gptq_words(expected, bits, module.planar).T)
    assert torch.equal(module.qzeros.cpu(), _gptq_words(zeros.cpu().long().T, bits, module.planar))
    if device == "cuda":
        torch.cuda.synchronize()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("method", ["original", "block", "native", "gpu"])
def test_gptq_finite_affine_overflow(monkeypatch, bits, method):
    device = "cuda" if method == "gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm device required")
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "0" if method == "native" else "1")
    monkeypatch.setenv("GPTQMODEL_FORCE_PACK_EXT", "1" if method == "native" else "0")
    linear, scales, zeros, expected = _inputs(torch.float16, bits, device, "affine_overflow")
    module = TorchLinear(bits=bits, group_size=128, sym=False, desc_act=False,
                         in_features=128, out_features=64, bias=False)
    getattr(module, "pack_" + ("block" if method == "native" else method))(
        linear, scales, zeros, torch.zeros(128, dtype=torch.int32, device=device),
    )
    assert torch.equal(module.qweight.cpu(), _words(expected, bits).T)


@pytest.mark.parametrize("cls", [AwqTorchLinear, AwqGEMMLinear, AwqGEMVLinear, AwqGEMVFastLinear, LLMAwqLinear])
@pytest.mark.parametrize("case", ["thresholds", "affine_overflow"])
def test_awq_pack_saturates_without_field_spill(device, cls, case):
    linear, scales, zeros, expected = _inputs(torch.float16, 4, device, case)
    module = _bare(cls)
    module.pack(linear, scales, zeros)
    if cls in (AwqGEMVFastLinear, LLMAwqLinear):
        packed = _fast_words(expected)
    elif cls is AwqGEMVLinear:
        packed = _words(expected, 4)
    else:
        packed = _awq_words(expected.T, 4)
    assert torch.equal(module.qweight.cpu(), packed)
    assert torch.isfinite(module.scales).all()


def test_public_awq_bf16_endpoint_preserves_255(device):
    linear, scales, zeros, expected = _inputs(torch.bfloat16, 8, device, "bf16_endpoint")
    module = AwqTorchLinear(bits=8, group_size=128, sym=False, desc_act=False,
                            in_features=128, out_features=64, bias=False, register_buffers=True)
    hf_pack_layer(module, linear, scales, zeros)
    assert torch.equal(module.qweight.cpu(), _words(expected.T, 8))
    assert torch.equal(module.qzeros.cpu(), _words(zeros.T, 8))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_awq_layout_helpers_mask_each_field(device, dtype):
    values = torch.arange(-16, 32, device=device).repeat(171)[:8192].reshape(64, 128).to(dtype)
    assert torch.equal(pack_awq_columns(values, 4).cpu(), _words(values, 4))
    assert torch.equal(pack_intweight(values, 4, 64).cpu(), _fast_words(values))


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_clamps_before_integer_conversion(device, group_size):
    linear = nn.Linear(256, 64, bias=False, dtype=torch.float16, device=device)
    module = QQQTorchLinear(bits=4, group_size=group_size, sym=True, desc_act=False,
                           in_features=256, out_features=64, bias=False)
    groups = 1 if group_size == -1 else 2
    scales = torch.full((64, groups), 0.5, dtype=torch.float16, device=device)
    channels = torch.ones(64, dtype=torch.float32, device=device)
    # Uniform codes make the full packed byte oracle independent of Marlin's
    # permutation. Include both signs, extrema, and FP16 division overflow.
    for weight in [-65504, -9, -4, 0, 3.5, 8, 65504]:
        with torch.no_grad():
            linear.weight.fill_(weight)
        hf_pack_layer(module, linear, scales, scales_extra=channels)
        code = max(-8, min(7, round(weight / 0.5))) if group_size == -1 else max(0, min(15, round(weight / 0.5) + 8))
        expected = _words(torch.full((module.B.shape[0], module.B.shape[1] * 8), code), 4)
        assert torch.equal(module.B.cpu(), expected)


def test_qqq_channel_scale_widens_before_division(device):
    linear = nn.Linear(128, 64, bias=False, dtype=torch.float16, device=device)
    module = QQQTorchLinear(bits=4, group_size=-1, sym=True, desc_act=False,
                           in_features=128, out_features=64, bias=False)
    for scale in [2.0 ** -24, 2.0 ** -21, 2.0 ** -20]:
        with torch.no_grad():
            linear.weight.fill_(7 * scale)
        hf_pack_layer(module, linear, torch.full((64, 1), scale, dtype=torch.float16, device=device))
        torch.testing.assert_close(module.s_channel, torch.full_like(module.s_channel, scale / 16), rtol=0, atol=0)
        assert torch.equal(module.B.cpu(), _words(torch.full((8, 128 * 8), 7), 4))


@pytest.mark.parametrize("fmt", ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"])
@pytest.mark.parametrize("method", ["tensor", "row", "block"])
def test_fp8_tiny_groups_have_finite_scales_and_exact_codes(fmt, method):
    dtype = getattr(torch, fmt)
    maximum = torch.finfo(dtype).max
    limit = torch.finfo(torch.float32).max
    threshold = torch.tensor(maximum / limit, dtype=torch.float32)
    maxima = [0., 2.0 ** -149, 2.0 ** -126, float(torch.nextafter(threshold, torch.tensor(0.))),
              float(threshold), float(torch.nextafter(threshold, torch.tensor(torch.inf)))]
    for value in maxima:
        weight = torch.tensor([[0., -0., value, -value]]).repeat(2, 1)
        actual, scales = quantize_fp8_weight(weight, format=fmt, weight_scale_method=method,
                                            weight_block_size=(1, 4) if method == "block" else None)
        # Compute the ideal inverse in FP64 then saturate to the stored FP32
        # range. Code multiplication follows the specified FP32 format math.
        inv = torch.tensor(min(maximum / max(value, torch.finfo(torch.float32).tiny), limit)
                           if value else 1., dtype=torch.float32)
        expected = (weight * inv).clamp(-maximum, maximum).to(dtype)
        assert torch.isfinite(scales).all()
        torch.testing.assert_close(scales, inv.expand_as(scales), rtol=1e-6, atol=0)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert torch.isfinite(actual.float()).all()


def test_mxfp4_minimum_scale_and_rounding_neighbors():
    scale = 2.0 ** -128  # half of the minimum E8M0 scale; table uses doubled codes
    table = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float64)
    values = []
    for value in [0, 0.5, 1, 1.5, 2, 3, 4]:
        sample = np.float32(value * scale)
        values.extend([np.nextafter(sample, np.float32(-np.inf)), sample, np.nextafter(sample, np.float32(np.inf))])
    values.extend([-v for v in values])
    weight = np.repeat(np.array(values, dtype=np.float32)[:, None], 32, axis=1)
    actual = _fallback_gguf_quantize(weight, "MXFP4")
    source = torch.from_numpy(weight).double()
    codes = (source[..., None] - table * scale).abs().argmin(-1)
    payload = (codes[:, :16] | (codes[:, 16:] << 4)).byte().numpy()
    expected = np.concatenate((np.zeros((len(values), 1), np.uint8), payload), axis=1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cls", [MacheteLinear, AwqMacheteLinear])
def test_machete_rejects_offset_overflow_before_repacking(device, cls):
    module = _bare(cls)
    module.sym = False
    module.weight_type = scalar_types.uint4
    module.register_buffer("scales", torch.full((1, 64), 5000., dtype=torch.float16, device=device))
    module.register_buffer("qzeros", _words(torch.full((1, 64), 14), 4).to(device))
    module.register_buffer("qweight", torch.zeros((16, 64), dtype=torch.int32, device=device))
    before = {name: value.clone() for name, value in module.state_dict().items()}
    with pytest.raises(ValueError, match="Scaled zero-point offset"):
        module.post_init()
    for name, value in before.items():
        assert torch.equal(module.state_dict()[name], value)


@pytest.mark.parametrize("cls", [AwqGEMVFastLinear, LLMAwqLinear])
def test_awq_scaled_zero_overflow_rejected_before_state_changes(device, cls):
    linear, scales, zeros, _ = _inputs(torch.float16, 4, device, "affine_overflow")
    zeros.fill_(14)
    module = _bare(cls)
    with pytest.raises(ValueError, match="Scaled zero-point offset"):
        module.pack(linear, scales, zeros)
    assert not module.state_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_zero_offset_finite_boundary(device, dtype):
    scale = torch.tensor([[1., 2.0 ** -24, 4096., 5000.]], dtype=dtype, device=device)
    zeros = torch.tensor([[0, 15, 15, 14]], device=device)
    if dtype == torch.float16:
        with pytest.raises(ValueError, match="Scaled zero-point offset"):
            scaled_zero_offset(scale, zeros)
        scale = scale[:, :3]
        zeros = zeros[:, :3]
    expected = (-scale.double() * zeros.double()).to(dtype)
    torch.testing.assert_close(scaled_zero_offset(scale, zeros), expected, rtol=0, atol=0)
