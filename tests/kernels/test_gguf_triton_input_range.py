# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Triton preserves the range of BF16/FP32 CUDA activations."""

import pytest
import torch

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
