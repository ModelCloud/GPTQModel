# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from tests.accelerator_gates import requires_nvidia_cuda


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
