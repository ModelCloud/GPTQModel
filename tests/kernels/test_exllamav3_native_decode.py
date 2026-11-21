# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from tests.accelerator_gates import requires_nvidia_cuda
from tests.test_exllamav3_torch_padding import _oracle_weight


pytestmark = pytest.mark.skipif(not requires_nvidia_cuda(), reason="native EXL3 decode requires NVIDIA CUDA")


@pytest.fixture(scope="module")
def native():
    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured


def _packed_projection(k, n, bits, seed=37):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, int(16 * bits)),
                           generator=generator, dtype=torch.int16).cuda()
    su = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    sv = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, su, sv


def _input(m, k, seed):
    return (torch.randn((m, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _mul1_torch_reference(weights, x):
    trellis, su, sv = weights
    model = ExllamaV3TorchLinear.from_tensors(
        in_features=su.numel(),
        out_features=sv.numel(),
        name="torch_oracle",
        tensors={"trellis": trellis, "suh": su, "svh": sv, "mul1": torch.ones((), dtype=torch.int32, device=x.device)},
    ).eval()
    dense = model.get_weight_tensor(dtype=torch.float32)
    return (x.double() @ dense.double()).float()


def _require_ampere():
    major, minor = torch.cuda.get_device_capability()
    if major != 8 or minor >= 9:
        pytest.skip("Q4NX tuned GEMV and transient prefill require NVIDIA Ampere")


def _q4nx_reference(weights, x, bias, start=0, end=None):
    trellis, su, sv = weights
    end = sv.numel() if end is None else end
    weight = _oracle_weight({
        "trellis": trellis[:, start // 16:end // 16].cpu(),
        "suh": su.cpu(),
        "svh": sv[start:end].cpu(),
        "q4nx": True,
    }, device=x.device)
    return (x.double() @ weight.double() + bias[start:end].double()).float()


@pytest.mark.parametrize("bits", (2.5, 3.0, 3.5))
@pytest.mark.parametrize(("rows", "k", "n", "dispatch"), (
    (1, 512, 256, 90),
    (8, 512, 256, 90),
    (9, 512, 256, None),
    (16, 5120, 4608, 2),
    (64, 2304, 5120, 92),
))
@torch.inference_mode()
def test_q4nx_tuned_dispatch_and_forced_packed(native, bits, rows, k, n, dispatch):
    _require_ampere()
    weights = _packed_projection(k, n, bits)
    x = _input(rows, k, seed=19)
    bias = torch.linspace(-0.031, 0.043, n, device=x.device)
    result = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    spans = [(0, n)] if n <= 256 else [(0, 128), (n - 128, n)]
    references = [(start, end, _q4nx_reference(weights, x, bias, start, end)) for start, end in spans]

    path = native(*weights, bits, bias, False, False, x, result, True, 0, False, q4nx=True)
    if dispatch is None:
        assert 1 <= path <= 7
    else:
        assert path == dispatch
    for start, end, expected in references:
        torch.testing.assert_close(result[:, start:end], expected, rtol=2e-3, atol=2e-3)

    path = native(*weights, bits, bias, False, False, x, result, False, 0, True, q4nx=True)
    assert 1 <= path <= 7
    for start, end, expected in references:
        torch.testing.assert_close(result[:, start:end], expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(("rows", "k", "n", "dispatch"), ((1, 512, 256, 90), (64, 2304, 5120, 92)))
@torch.inference_mode()
def test_q4nx_graph_replays_changed_inputs_with_stream_private_scratch(native, rows, k, n, dispatch):
    _require_ampere()
    bits = 3.5
    arms = []
    current = torch.cuda.current_stream()
    for seed in (23, 24):
        weights = _packed_projection(k, n, bits, seed)
        source = _input(rows, k, seed)
        changed = _input(rows, k, seed + 11)
        x = source.clone()
        bias = torch.linspace(-0.065, 0.042, n, device=x.device) * seed
        out = torch.empty((rows, n), device=x.device, dtype=torch.float32)
        expected = [_q4nx_reference(weights, inputs, bias, 0, 128) for inputs in (source, changed)]
        assert not torch.equal(*expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            assert native(*weights, bits, bias, False, False, x, out, True, 0, q4nx=True) == dispatch
            stream.synchronize()
            before = torch.cuda.memory_allocated(x.device)
            for _ in range(3):
                assert native(*weights, bits, bias, False, False, x, out, True, 0, q4nx=True) == dispatch
            stream.synchronize()
            assert torch.cuda.memory_allocated(x.device) == before
            with torch.cuda.graph(graph, stream=stream):
                assert native(*weights, bits, bias, False, False, x, out, True, 0, q4nx=True) == dispatch
        arms.append((stream, graph, weights, bias, x, out, (source, changed), expected))

    for state in (0, 1, 0, 1):
        for stream, graph, _, _, x, _, inputs, _ in arms:
            with torch.cuda.stream(stream):
                x.copy_(inputs[state])
                graph.replay()
        for stream, _, _, _, _, out, _, expected in arms:
            current.wait_stream(stream)
            torch.testing.assert_close(out[:, :128], expected[state], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(("rows", "k", "n"), ((1, 512, 256), (64, 2304, 5120)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@torch.inference_mode()
def test_q4nx_public_inference_with_bias_and_strided_input(rows, k, n, dtype):
    _require_ampere()
    bits = 3.5
    weights = _packed_projection(k, n, bits)
    trellis, su, sv = weights
    bias = torch.linspace(-0.062, 0.043, n, device=trellis.device).half()
    layer = ExllamaV3Linear.from_tensors(
        in_features=k, out_features=n, name="q4nx",
        tensors={"trellis": trellis, "suh": su, "svh": sv, "bias": bias,
                 "q4nx": torch.ones((), dtype=torch.int32, device=trellis.device)},
    ).eval()
    x = _input(rows, k, seed=21).to(dtype).repeat_interleave(2, dim=-1)[..., ::2]
    assert not x.is_contiguous()
    expected = _q4nx_reference(weights, x, bias, 0, 128)
    result = layer(x)
    assert result.shape == (rows, n)
    assert result.dtype == dtype
    rounded = expected.to(dtype)
    if dtype == torch.bfloat16:
        actual = result[:, :128]
        outside = ~torch.isclose(actual, rounded, rtol=2e-3, atol=5e-4)
        torch.testing.assert_close(actual[outside], torch.nextafter(rounded[outside], actual[outside]),
                                   rtol=0, atol=0)
    else:
        torch.testing.assert_close(result[:, :128], rounded, rtol=2e-3, atol=2e-3)
    retained = set(layer.state_dict())
    assert retained == {"trellis", "suh", "svh", "bias", "q4nx"}
    assert layer.trellis.data_ptr() == trellis.data_ptr()


@pytest.mark.parametrize(("m", "expected_gemv"), ((1, True), (8, True), (9, False), (24, False), (65, False)))
@torch.inference_mode()
def test_native_fused_decode_and_gemm_match_torch_oracle(native, m, expected_gemv):
    k, n, bits = 512, 256, 3
    weights = _packed_projection(k, n, bits)
    x = _input(m, k, seed=13)
    expected = _mul1_torch_reference(weights, x)
    result = torch.empty((m, n), device=x.device, dtype=torch.float32)
    path = native(*weights, bits, None, False, True, x, result, True, 0, True)
    assert (path == 90) == expected_gemv
    torch.testing.assert_close(result, expected, rtol=0.002, atol=0.002)

    forced_dense = native(*weights, bits, None, False, True, x, result, False, 0, True)
    assert 1 <= forced_dense <= 7
    torch.testing.assert_close(result, expected, rtol=0.002, atol=0.002)

    legacy = native(*weights, bits, None, False, True, x, result, False, 0, False)
    assert 1 <= legacy <= 4
    legacy_result = torch.empty_like(result)
    torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run(*weights, bits, None, False, True, x, legacy_result)
    torch.testing.assert_close(result, legacy_result, rtol=0, atol=0)


@pytest.mark.parametrize("mode", (1, 2))
@torch.inference_mode()
def test_int8_capture_changed_inputs_and_independent_workspaces(native, mode):
    k, n, bits = 512, 256, 3
    # The two projections have independent scales, inputs and outputs, while
    # their equal geometries select the same instantiation and occupy few SMs.
    arms = []
    for seed in (23, 24):
        weights = _packed_projection(k, n, bits, seed)
        source = _input(1, k, seed)
        second = _input(1, k, seed + 54)
        x = torch.empty_like(source)
        output = torch.empty((1, n), device=source.device, dtype=torch.float32)
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        expected = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for state in (source, second):
                x.copy_(state)
                assert native(*weights, bits, None, False, True, x, output, True, mode) == 91
                expected.append(output.clone())
            with torch.cuda.graph(graph, stream=stream):
                assert native(*weights, bits, None, False, True, x, output, True, mode) == 91
        current.wait_stream(stream)
        oracle = _mul1_torch_reference(weights, second)
        drift = ((expected[1] - oracle).norm() / oracle.norm()).item()
        assert drift < (0.002 if mode == 1 else 0.02)
        arms.append((stream, graph, x, output, (source, second), expected, weights))

    # Submit both graphs on their own streams before checking either result.
    for i in range(10):
        state = i % 2
        for stream, graph, x, output, inputs, expected, _weights in arms:
            with torch.cuda.stream(stream):
                x.copy_(inputs[state])
                graph.replay()
        for stream, _, _, output, _, expected, _weights in arms:
            stream.synchronize()
            torch.testing.assert_close(output, expected[state], rtol=0, atol=0)


@torch.inference_mode()
def test_integer_packed_width_cannot_be_used_as_fractional(native):
    bits, k, n = 3, 256, 256
    weights = _packed_projection(k, n, bits)
    x = _input(1, k, seed=69)
    y = torch.empty((1, n), dtype=torch.float32, device=x.device)
    with pytest.raises(RuntimeError, match="K does not match packed trellis width"):
        native(*weights, 2.5, None, False, True, x, y, True, 0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rows", (1, 33))
@pytest.mark.parametrize("with_bias", (False, True))
@torch.inference_mode()
def test_native_input_hadamard_range(native, dtype, rows, with_bias):
    """Finite projections must survive an overflowing input rotation (#5838)."""
    k, n = 512, 256
    tensors = {
        "trellis": torch.zeros((k // 16, n // 16, 64), dtype=torch.int16, device="cuda"),
        "suh": torch.ones(k, dtype=torch.float16, device="cuda"),
        "svh": torch.full((n,), 2**-10, dtype=torch.float16, device="cuda"),
        "mcg": torch.tensor(0xCBAC1FED - 2**32, dtype=torch.int32, device="cuda"),
    }
    if with_bias:
        tensors["bias"] = torch.linspace(-0.1, 0.1, n, device="cuda", dtype=torch.float32)
    layer = ExllamaV3Linear.from_tensors(
        in_features=k, out_features=n, name="input_range", tensors=tensors,
    ).eval()
    original = {name: value.clone() for name, value in layer.named_buffers()}

    # Exercise warm forwards and live checkpoint-scale changes as well as the
    # cold path. Small rows next to the large row must not inherit its scale.
    for value, sign_scale in ((6000., 1.), (4096., 2.), (-4096., -2.), (1., 1.)):
        layer.suh.fill_(sign_scale)
        x = torch.full((rows, k), 0.5, dtype=dtype, device="cuda")
        x[0] = value
        # All-zero MCG states independently decode to 2 * FP16(0x3b60).
        # The two normalized 128-point Hadamards leave only the DC entries.
        expected = torch.zeros((rows, n), dtype=torch.float64, device="cuda")
        expected[:, ::128] = x[:, :1].double() * (4 * 128 * 1.84375 * sign_scale * 2**-10)
        if with_bias:
            expected += tensors["bias"].double()
        actual = layer(x)
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-3, atol=2e-3)
        assert layer._torch_inner is None, "the regression must exercise native EXL3"

    # A warmed forward must not read the activation-dependent bound on the
    # host. Capture and replay must recompute it when the input changes.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replay_output = layer(x)
    for value in (6000., -4096.):
        x[0].fill_(value)
        graph.replay()
        expected.zero_()
        expected[:, ::128] = x[:, :1].double() * (4 * 128 * 1.84375 * 2**-10)
        if with_bias:
            expected += tensors["bias"].double()
        torch.testing.assert_close(replay_output, expected.to(dtype), rtol=2e-3, atol=2e-3)

    for name, saved in original.items():
        torch.testing.assert_close(getattr(layer, name), saved, rtol=0, atol=0)


@pytest.mark.parametrize("cancel", (False, True))
@torch.inference_mode()
def test_bf16_checkpoint_product_exceeds_fp32(native, cancel):
    """A finite x*suh must not overflow the range check itself after warmup."""
    layer = ExllamaV3Linear.from_tensors(in_features=512, out_features=256, name="wide_range", tensors={
        "trellis": torch.zeros((32, 16, 64), device="cuda", dtype=torch.int16),
        "suh": torch.full((512,), 65504., device="cuda", dtype=torch.float16),
        "svh": torch.full((256,), 2**-24, device="cuda", dtype=torch.float16),
        "mcg": torch.tensor(0xCBAC1FED - 2**32, device="cuda", dtype=torch.int32),
    }).eval()
    x = torch.ones((1, 512), device="cuda", dtype=torch.bfloat16)
    layer(x)
    if cancel:
        # Only the DC input coordinates contribute for the constant MCG
        # fixture. This non-DC impulse has zero projection, despite requiring
        # a power-of-two pre-scale larger than float32 can represent.
        x.zero_()
        x[0, 1] = 3e38
    else:
        x.fill_(1e36)
    expected = torch.zeros((1, 256), device="cuda", dtype=torch.float64)
    expected[:, ::128] = x[:, :1].double() * (4 * 128 * 1.84375 * 65504 * 2**-24)
    assert torch.isfinite(expected.to(torch.bfloat16)).all()
    actual = layer(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=2e-3, atol=2e-3)
    assert layer._torch_inner is None
