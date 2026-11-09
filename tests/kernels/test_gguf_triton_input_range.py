# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Triton preserves the range of BF16/FP32 CUDA activations."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf_triton as gguf_triton_module
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear, _dequantize_gguf_tensor_numpy
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _packed_pair(bits: str, *, bias: bool, in_features: int = 256):
    k, n = in_features, 64
    linear = torch.nn.Linear(k, n, bias=bias)
    with torch.no_grad():
        linear.weight.fill_(0.0005)
        if bias:
            linear.bias.fill_(-5000)
    options = {"bits": bits, "group_size": -1, "sym": True, "desc_act": False,
               "in_features": k, "out_features": n, "bias": bias}
    source = GGUFTorchLinear(**options, register_buffers=False)
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()
    target = GGUFTritonKernel(**options, register_buffers=True)
    target.load_state_dict(source.state_dict(), strict=True)
    return source, target.cuda().eval()


@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q1_0", "q4_k", "q5_k", "q6_k"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value", [100000.0, 1e-10])
@pytest.mark.parametrize("rows", [1, 9])
def test_wide_activation_round_trip_matches_packed_fp64_oracle(bits, dtype, value, rows):
    source, target = _packed_pair(bits, bias=False)
    packed_before = target.qweight.clone()
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        source.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    )).to(device="cuda", dtype=torch.float64)
    x = torch.full((rows, 256), value, device="cuda", dtype=dtype)
    oracle = x.double() @ dense.t()
    expected = oracle.to(dtype)

    with torch.inference_mode():
        actual = target(x)
        repeated = target(x)
    torch.cuda.synchronize()
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, repeated)
    # Compare after mandatory output rounding. BF16 can round by >0.2% at
    # some magnitudes, even when the FP32 arithmetic is exact.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0)
    assert torch.equal(target.qweight, packed_before)
    assert not any(
        torch.is_tensor(value) and value.is_floating_point() and value.numel() >= 256 * 64
        for value in vars(target).values()
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_wide_activation_bias_is_added_after_restoring_scale(dtype):
    source, target = _packed_pair("q4_k", bias=True)
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        source.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    )).to(device="cuda", dtype=torch.float64)
    storage = torch.full((3, 9, 264), 100000.0, device="cuda", dtype=dtype)
    x = storage[:, :, :256]
    assert not x.is_contiguous()
    oracle = (x.double() @ dense.t()) + target.bias.double()
    with torch.inference_mode():
        actual = target(x)
    assert actual.shape == (3, 9, 64)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, oracle.to(dtype), rtol=2e-3, atol=0)


@pytest.mark.cuda
def test_wide_activation_replays_inside_cuda_graph():
    _, target = _packed_pair("q4_k", bias=False)
    x = torch.full((1, 256), 100000.0, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = target(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = target(x)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected)


@pytest.mark.cuda
def test_wide_activation_with_padded_gguf_input_width():
    source, target = _packed_pair("q4_k", bias=False, in_features=257)
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        source.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    ))[:, :257].to(device="cuda", dtype=torch.float64)
    x = torch.full((2, 257), 100000.0, device="cuda", dtype=torch.bfloat16)
    expected = (x.double() @ dense.t()).to(x.dtype)
    with torch.inference_mode():
        actual = target(x)
    assert actual.shape == (2, 64)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("bits", ["q1_0", "q4_k", "q5_k", "q6_k"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 9])
def test_wide_biased_dot_survives_near_cancellation(bits, dtype, rows):
    source, target = _packed_pair(bits, bias=True)
    packed_before = target.qweight.clone()
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        source.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    )).to(device="cuda", dtype=torch.float64)
    x = torch.full((rows, 256), 100000.0, device="cuda", dtype=dtype)
    dot = x.double() @ dense.t()
    target.bias.fill_(-dot[0, 0].item() + 64.0)
    oracle = (dot + target.bias.double()).to(dtype)
    with torch.inference_mode():
        actual = target(x)
        repeated = target(x)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, repeated)
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(target.qweight, packed_before)
    wide_cache = target._get_triton_cache(x.device, wide_dot=True)
    if bits != "q1_0":
        assert wide_cache["scale"].dtype == torch.float32
    assert all(
        not torch.is_tensor(value) or value.numel() < 256 * 64
        for entry in target._gguf_triton_cache.values() for value in entry.values()
    )


@pytest.mark.cuda
def test_wide_biased_dot_replays_in_cuda_graph_and_invalidates_weight_cache():
    source, target = _packed_pair("q4_k", bias=True)
    x = torch.full((1, 256), 100000.0, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = target(x)
        old_cache = target._get_triton_cache(x.device, wide_dot=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = target(x)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected)

    target.qweight[0, 0] ^= 1
    new_cache = target._get_triton_cache(x.device, wide_dot=True)
    assert new_cache is not old_cache
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        target.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    )).to(device="cuda", dtype=torch.float64)
    oracle = (x.double() @ dense.t() + target.bias.double()).to(x.dtype)
    torch.testing.assert_close(target(x), oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.cuda
def test_q1_fp32_dot_matches_packed_oracle_in_all_four_kernel_variants():
    source, target = _packed_pair("q1_0", bias=True, in_features=2048)
    cache = target._get_triton_cache(torch.device("cuda"))
    sign_bytes = cache["sign_bytes"]
    words = sign_bytes.to(torch.int32)
    sign_words = (
        words[:, 0::4, :] | (words[:, 1::4, :] << 8)
        | (words[:, 2::4, :] << 16) | (words[:, 3::4, :] << 24)
    ).contiguous()
    x = torch.full((1, 2048), 100000.0, device="cuda", dtype=torch.float32)
    dense = torch.from_numpy(_dequantize_gguf_tensor_numpy(
        source.qweight.cpu().numpy(), source.gguf_tensor_qtype,
    )).to(device="cuda", dtype=torch.float64)
    oracle = x.double() @ dense.t()
    variants = (
        (gguf_triton_module.fused_q1_0_g128_matmul, sign_bytes),
        (gguf_triton_module.fused_q1_0_g128_k2048_matmul, sign_bytes),
        (gguf_triton_module.fused_q1_0_g128_u32_matmul, sign_words),
        (gguf_triton_module.fused_q1_0_g128_u32_k2048_matmul, sign_words),
    )
    outputs = [fn(x, packed, cache["scale"], output_dtype=torch.float32) for fn, packed in variants]
    for output in outputs:
        relative_error = (output.double() - oracle).abs().max() / oracle.abs().max()
        assert relative_error.item() <= 2e-3
        assert torch.equal(output, outputs[0])
