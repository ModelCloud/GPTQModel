import os
import subprocess
import sys
import weakref

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_module
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import pack_hf_int4, unpack_hf_int4
from gptqmodel.utils.cpu_features import has_avx512_bf16


@pytest.fixture(scope="module")
def packed_kernel():
    if not has_avx512_bf16():
        pytest.skip("AVX512-BF16 required")
    kernel = qqq_module._qqq_cpu_kernel()
    assert kernel is not None
    return kernel


def build_case(in_features=256, out_features=128, group_size=128, dtype=torch.float16, bias=False):
    torch.manual_seed(730 + in_features + group_size)
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    scales = torch.rand(out_features, groups, dtype=torch.float16) * 0.03 + 0.01
    channels = torch.rand(out_features, dtype=torch.float32) * 0.002 + 0.001
    signed_codes = torch.randint(-8, 8, (out_features, in_features))
    linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=torch.float16)
    linear.weight.data.copy_(signed_codes * scales.repeat_interleave(effective_group, dim=1))
    codes = torch.round(linear.weight.detach() / scales.repeat_interleave(effective_group, dim=1))
    if effective_group != in_features:
        codes = (codes + 8).clamp(0, 15)
        group_scales = (scales.t() / channels).half().to(dtype).float()
        weight = ((codes.t().float() - 8) * group_scales.repeat_interleave(effective_group, dim=0)).round().clamp(-128, 127)
        channel_scales = channels.reshape(1, -1).to(dtype).float()
    else:
        codes = codes.clamp(-15, 15).to(torch.int32) & 15
        weight = torch.where(codes.t() >= 8, codes.t() - 16, codes.t()).float() * 16
        channel_scales = (scales.t() / 16).float().to(dtype).float()
    module = qqq_module.QQQTorchLinear(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=bias,
    )
    module.pack(linear, scales, channels)
    module.post_init()
    module.eval().to(dtype)
    return module, codes.t().to(torch.uint8), weight, channel_scales


def oracle(module, activation, weight, channels):
    flattened = activation.reshape(-1, module.in_features)
    scale = flattened.abs().amax(-1, keepdim=True).div(127).float()
    quantized = (flattened / scale).round().clamp(-128, 127).to(torch.int8)
    accum = quantized.float() @ weight
    raw = accum * scale * channels
    visible = raw.to(activation.dtype).reshape(*activation.shape[:-1], module.out_features)
    if module.bias is not None:
        visible = visible + module.bias.to(activation.dtype)
    return quantized, accum, raw, visible


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("tokens", [1, 2, 7, 25, 128])
def test_packed_qqq_oracle(packed_kernel, monkeypatch, dtype, group_size, tokens):
    module, codes, weight, channels = build_case(dtype=dtype, group_size=group_size)
    activation = torch.randn(tokens, module.in_features, dtype=dtype)
    quantized, expected_accum, raw, visible = oracle(module, activation, weight, channels)
    packed, scales, _ = module._packed_weight_for_cpu()
    restored = unpack_hf_int4(packed, module.out_features, module.in_features).t()
    torch.testing.assert_close(restored, codes ^ 8 if group_size == -1 else codes, rtol=0, atol=0)
    actual_accum = packed_kernel(quantized, packed, scales, module.group_size)
    torch.testing.assert_close(actual_accum, expected_accum, rtol=0, atol=0)
    with torch.inference_mode():
        output = module(activation)
    assert output.dtype == dtype
    torch.testing.assert_close(output, visible, rtol=0, atol=0)
    scale = activation.abs().amax(-1, keepdim=True).div(127).float()
    torch.testing.assert_close(actual_accum * scale * channels, raw, rtol=0, atol=0)
    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", lambda: None)
    with torch.inference_mode():
        fallback = module(activation)
    torch.testing.assert_close(fallback, visible, rtol=0, atol=0)
    assert not hasattr(module, "_torch_weight_cache")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qqq_bias_noncontiguous_and_empty(packed_kernel, dtype):
    module, _, weight, channels = build_case(dtype=dtype, bias=True)
    activation = torch.randn(2, module.in_features, 7, dtype=dtype).transpose(1, 2)
    _, _, _, expected = oracle(module, activation, weight, channels)
    torch.testing.assert_close(module(activation), expected, rtol=0, atol=0)
    empty = module(activation[:, :0])
    assert empty.shape == (2, 0, module.out_features)
    assert empty.dtype == dtype


def test_qqq_rounding_neighbors_and_saturation(packed_kernel):
    in_features, out_features = 256, 128
    thresholds = torch.tensor([0, 0.5, 1.5, 2.5, 15.5, 16, 18, 20, 22], dtype=torch.float16)
    values = torch.cat([
        torch.nextafter(thresholds, torch.full_like(thresholds, -torch.inf)),
        thresholds,
        torch.nextafter(thresholds, torch.full_like(thresholds, torch.inf)),
    ])
    scales = values.repeat((out_features + values.numel() - 1) // values.numel())[:out_features].float()
    scales = scales.repeat(2, 1).contiguous()
    codes = torch.arange(in_features).remainder(16).byte().repeat(out_features, 1)
    packed = pack_hf_int4(codes)
    activation = torch.eye(in_features, dtype=torch.int8)
    expected = ((codes.t().float() - 8) * scales.repeat_interleave(128, dim=0)).round().clamp(-128, 127)
    torch.testing.assert_close(packed_kernel(activation, packed, scales, 128), expected, rtol=0, atol=0)


@pytest.mark.parametrize("buffer", ["B", "s_group", "s_channel"])
def test_packed_qqq_mutation(packed_kernel, buffer):
    module, _, _, _ = build_case()
    activation = torch.randn(7, module.in_features, dtype=torch.float16)
    module(activation)
    first = module._torch_packed_cache
    getattr(module, buffer).add_(1)
    actual = module(activation)
    assert module._torch_packed_cache is not first
    weight, channels = module._dequantize_weight_for_torch()
    _, _, _, expected = oracle(module, activation, weight, channels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qqq_no_kernel_redequantizes_without_caching(monkeypatch):
    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", lambda: None)
    module, _, weight, channels = build_case()
    activation = torch.randn(7, module.in_features, dtype=torch.float16)
    _, _, _, expected = oracle(module, activation, weight, channels)
    scratch = []
    original_dequantize = module._dequantize_weight_for_torch
    original_matmul = module._int8_matmul

    def record_dequantize():
        expanded, scales = original_dequantize()
        scratch.append(weakref.ref(expanded))
        return expanded, scales

    def record_matmul(quantized, expanded):
        scratch.append(weakref.ref(expanded))
        return original_matmul(quantized, expanded)

    monkeypatch.setattr(module, "_dequantize_weight_for_torch", record_dequantize)
    monkeypatch.setattr(module, "_int8_matmul", record_matmul)
    for _ in range(2):
        torch.testing.assert_close(module(activation), expected, rtol=0, atol=0)
        assert all(reference() is None for reference in scratch)
        assert module._torch_packed_cache is None
        assert not hasattr(module, "_torch_weight_cache")


@pytest.mark.parametrize("in_features,out_features,group_size", [(64, 128, -1), (128, 64, 128), (512, 512, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qqq_shapes_and_zero_input(packed_kernel, in_features, out_features, group_size, dtype):
    module, _, weight, channels = build_case(in_features, out_features, group_size, dtype)
    activation = torch.randn(33, in_features, dtype=dtype)
    activation[0] = 0
    _, _, _, expected = oracle(module, activation, weight, channels)
    torch.testing.assert_close(module(activation), expected, rtol=0, atol=0)
    assert torch.isfinite(module(activation)).all()


@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("grouped", [True, False])
def test_qqq_integer_extrema_and_cancellation(packed_kernel, tokens, grouped):
    in_features, out_features = 4096, 32
    codes = torch.zeros(out_features, in_features, dtype=torch.uint8)
    codes[::2, in_features // 2:] = 15
    activation = torch.full((tokens, in_features), -128, dtype=torch.int8)
    activation[:, 1::2] = 127
    activation[:, :in_features // 2] = -128
    scales = torch.full((in_features // 128, out_features), 20.0) if grouped else torch.empty(0)
    weight = (codes.t().double() - 8) * (20 if grouped else 16)
    if grouped:
        weight = weight.round().clamp(-128, 127)
    expected = (activation.double() @ weight).float()
    actual = packed_kernel(activation, pack_hf_int4(codes), scales, 128 if grouped else in_features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("change,message", [
    ("input_dtype", "requires int8"),
    ("weight_shape", "shape mismatch"),
    ("scales_shape", "scales shape mismatch"),
    ("group_size", "group size"),
    ("noncontiguous", "contiguous"),
])
def test_qqq_kernel_rejects_invalid_inputs(packed_kernel, change, message):
    activation = torch.ones(7, 256, dtype=torch.int8)
    packed = torch.zeros(128, 128, dtype=torch.uint8)
    scales = torch.ones(2, 128)
    group_size = 128
    if change == "input_dtype":
        activation = activation.float()
    elif change == "weight_shape":
        packed = packed[:, :64].contiguous()
    elif change == "scales_shape":
        scales = scales[:1]
    elif change == "group_size":
        group_size = 3
    else:
        activation = activation.t()
    with pytest.raises(RuntimeError, match=message):
        packed_kernel(activation, packed, scales, group_size)


def test_qqq_without_onednn(packed_kernel):
    module, _, weight, channels = build_case()
    activation = torch.randn(33, module.in_features, dtype=torch.float16)
    _, _, _, expected = oracle(module, activation, weight, channels)
    with torch.backends.mkldnn.flags(enabled=False):
        torch.testing.assert_close(module(activation), expected, rtol=0, atol=0)


def test_qqq_nonfinite_scale_cast_matches_torch(packed_kernel):
    codes = torch.arange(256).remainder(16).byte().repeat(32, 1)
    scales = torch.ones(2, 32)
    scales[:, 0] = torch.inf
    scales[:, 1] = -torch.inf
    scales[:, 2] = torch.nan
    weight = ((codes.t().float() - 8) * scales.repeat_interleave(128, dim=0)).round().clamp(-128, 127).to(torch.int8)
    activation = torch.eye(256, dtype=torch.int8)
    actual = packed_kernel(activation, pack_hf_int4(codes), scales, 128)
    torch.testing.assert_close(actual, weight.float(), rtol=0, atol=0)


def test_qqq_onednn_isa_cap_falls_back(packed_kernel):
    code = """
import torch
from gptqmodel.nn_modules.qlinear.qqq import _qqq_cpu_kernel
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import pack_hf_int4
torch.set_num_threads(2)
torch.manual_seed(47)
codes = torch.randint(0, 16, (32, 256), dtype=torch.uint8)
activation = torch.randint(-128, 128, (7, 256), dtype=torch.int8)
scales = torch.ones(2, 32)
expected = activation.float() @ (codes.t().float() - 8)
actual = _qqq_cpu_kernel()(activation, pack_hf_int4(codes), scales, 128)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
"""
    environment = dict(os.environ)
    environment["ONEDNN_MAX_CPU_ISA"] = "AVX2"
    environment["DNNL_MAX_CPU_ISA"] = "AVX2"
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
