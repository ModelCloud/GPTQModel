# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.nn_modules.triton_utils.three_bit import (
    LAYOUT_AWQ,
    LAYOUT_GPTQ,
    pack_3bit,
    prepare_trilin_eora_3bit,
)
from gptqmodel.utils.trilin import trilin_matmul, trilin_matmul_eora


K = 4096
N = 4096
RANKS = (32, 64, 128, 256)
GROUP_SIZE = 128


def _require_sm80() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for fused TriLin+EoRA tests")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("the fused TriLin+EoRA specialization is enabled only on sm_80")


def _raw_case(dtype: torch.dtype, rank: int, *, seed: int = 73):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K // 32 * 3, N),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((K // GROUP_SIZE, N), dtype=torch.float16, device=device).mul_(0.02).add_(0.001)
    x = torch.randn((1, K), dtype=dtype, device=device)
    lora_a = torch.randn((K, rank), dtype=dtype, device=device).mul_(0.02)
    lora_b = torch.randn((rank, N), dtype=dtype, device=device).mul_(0.02)
    bias = torch.randn((N,), dtype=dtype, device=device).mul_(0.01)
    workspace = torch.full((rank,), float("nan"), dtype=torch.float32, device=device)
    return x, qweight, scales, bias, lora_a, lora_b, workspace


def _unfused_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    base = trilin_matmul(x, qweight, scales, bias)
    return Lora(rank=lora_a.shape[1], lora_A=lora_a, lora_B=lora_b).apply(x, base)


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trilin_eora_supported_rank_matches_unfused_reference_and_overwrites_workspace(
    rank: int,
    dtype: torch.dtype,
):
    _require_sm80()
    x, qweight, scales, bias, lora_a, lora_b, workspace = _raw_case(dtype, rank, seed=73 + rank)

    with torch.inference_mode():
        expected = _unfused_reference(x, qweight, scales, bias, lora_a, lora_b)
        actual = trilin_matmul_eora(x, qweight, scales, lora_a, lora_b, workspace, bias)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape == (1, N)
    assert actual.dtype == expected.dtype == dtype
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(workspace).all())
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.0625 if dtype == torch.float16 else 0.5)

    next_x = torch.randn_like(x)
    workspace.fill_(float("nan"))
    with torch.inference_mode():
        next_expected = _unfused_reference(next_x, qweight, scales, bias, lora_a, lora_b)
        next_actual = trilin_matmul_eora(next_x, qweight, scales, lora_a, lora_b, workspace, bias)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(workspace).all())
    torch.testing.assert_close(
        next_actual,
        next_expected,
        rtol=0.002,
        atol=0.0625 if dtype == torch.float16 else 0.5,
    )


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trilin_eora_supported_rank_uses_current_stream(rank: int, dtype: torch.dtype):
    _require_sm80()
    x, qweight, scales, bias, lora_a, lora_b, workspace = _raw_case(dtype, rank, seed=79 + rank)
    with torch.inference_mode():
        expected = _unfused_reference(x, qweight, scales, bias, lora_a, lora_b)

    stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.stream(stream), torch.inference_mode():
        actual = trilin_matmul_eora(x, qweight, scales, lora_a, lora_b, workspace, bias)
        completion = torch.cuda.Event()
        completion.record(stream)
    torch.cuda.current_stream(device=x.device).wait_event(completion)

    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.0625 if dtype == torch.float16 else 0.5)


@pytest.mark.parametrize("rank", RANKS)
def test_trilin_eora_supported_rank_rejects_undersized_workspace(rank: int):
    _require_sm80()
    x, qweight, scales, bias, lora_a, lora_b, _ = _raw_case(torch.float16, rank, seed=83 + rank)
    workspace = torch.empty((rank - 1,), dtype=torch.float32, device=x.device)

    with pytest.raises(RuntimeError, match=rf"at least {rank} FP32 values"):
        trilin_matmul_eora(x, qweight, scales, lora_a, lora_b, workspace, bias)


def test_trilin_eora_native_rejects_unsupported_rank():
    _require_sm80()
    x, qweight, scales, bias, lora_a, lora_b, workspace = _raw_case(torch.float16, 96, seed=181)

    with pytest.raises(RuntimeError, match="requires rank 32, 64, 128, or 256, got 96"):
        trilin_matmul_eora(x, qweight, scales, lora_a, lora_b, workspace, bias)


def _quant_linear(layout: str, dtype: torch.dtype, rank: int):
    device = torch.device("cuda")
    linear_cls = TritonV2Linear if layout == LAYOUT_GPTQ else AwqGEMMTritonLinear
    module = linear_cls(
        bits=3,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=N,
        bias=True,
        pack_dtype=torch.int32,
        adapter=Lora(rank=rank),
        register_buffers=True,
    ).to(device)
    module.qweight.random_(0, torch.iinfo(torch.int32).max)
    zeros = pack_3bit(
        torch.full((K // GROUP_SIZE, N), 4, dtype=torch.int32, device=device),
        axis=1,
    )
    module.qzeros.copy_(zeros)
    module.scales.uniform_(0.001, 0.021)
    module.bias.uniform_(-0.01, 0.01)
    module.lora_A.normal_(0.0, 0.02)
    module.lora_B.normal_(0.0, 0.02)
    if layout == LAYOUT_GPTQ:
        module.g_idx.copy_(torch.arange(K, dtype=torch.int32, device=device) // GROUP_SIZE)
    module.post_init()
    module.eval()
    if dtype == torch.bfloat16:
        module.adapter.lora_A = module.adapter.lora_A.bfloat16()
        module.adapter.lora_B = module.adapter.lora_B.bfloat16()
    return module


@pytest.mark.parametrize("layout", [LAYOUT_GPTQ, LAYOUT_AWQ])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", RANKS)
def test_trilin_eora_quant_linear_routes_decode_and_preserves_m2_fallback(
    layout: str,
    dtype: torch.dtype,
    rank: int,
):
    _require_sm80()
    pytest.importorskip("triton")
    torch.manual_seed(89)
    module = _quant_linear(layout, dtype, rank)
    runtime_qweight = (
        module.qweight if layout == LAYOUT_GPTQ else module._triton_3bit_qweight
    )
    assert module._trilin_eora_workspace.shape == (rank,)
    assert module._trilin_eora_workspace.dtype == torch.float32
    assert "_trilin_eora_workspace" not in module.state_dict()

    original_apply = module.adapter.apply
    apply_calls = 0

    def tracked_apply(*args, **kwargs):
        nonlocal apply_calls
        apply_calls += 1
        return original_apply(*args, **kwargs)

    module.adapter.apply = tracked_apply
    x_decode = torch.randn((1, 1, K), dtype=dtype, device="cuda")
    with torch.inference_mode():
        decode_base = trilin_matmul(
            x_decode.reshape(1, K),
            runtime_qweight,
            module.scales,
            module.bias,
        ).reshape(1, 1, N)
        decode_expected = original_apply(x_decode, decode_base)
        decode_actual = module(x_decode)
        decode_repeat = module(x_decode)
    assert apply_calls == 0
    torch.testing.assert_close(
        decode_actual,
        decode_expected,
        rtol=0.002,
        atol=0.0625 if dtype == torch.float16 else 0.5,
    )
    torch.testing.assert_close(decode_repeat, decode_actual, rtol=0, atol=0)

    x_fallback = torch.randn((1, 2, K), dtype=dtype, device="cuda")
    with torch.inference_mode():
        fallback_base = trilin_matmul(
            x_fallback.reshape(2, K),
            runtime_qweight,
            module.scales,
            module.bias,
        ).reshape(1, 2, N)
        fallback_expected = original_apply(x_fallback, fallback_base)
        fallback_actual = module(x_fallback)
    assert apply_calls == 1
    torch.testing.assert_close(fallback_actual, fallback_expected, rtol=0, atol=0)


def test_trilin_eora_quant_linear_uses_graph_safe_fallback(monkeypatch):
    _require_sm80()
    pytest.importorskip("triton")
    torch.manual_seed(97)
    module = _quant_linear(LAYOUT_GPTQ, torch.float16, 256)
    x = torch.randn((1, 1, K), dtype=torch.float16, device="cuda")

    # Warm the established unfused route and its allocator state before
    # capture, then re-enable fusion. Capture itself must still select that
    # graph-safe fallback rather than attempting a cooperative launch.
    monkeypatch.setenv("GPTQMODEL_TRILIN_EORA", "0")
    warmup_stream = torch.cuda.Stream(device=x.device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device=x.device))
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(3):
            expected = module(x)
    torch.cuda.current_stream(device=x.device).wait_stream(warmup_stream)
    monkeypatch.setenv("GPTQMODEL_TRILIN_EORA", "1")

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        actual = module(x)
    graph.replay()
    torch.cuda.synchronize(x.device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trilin_eora_quant_linear_uses_disjoint_workspaces_across_streams():
    _require_sm80()
    pytest.importorskip("triton")
    torch.manual_seed(101)
    module = _quant_linear(LAYOUT_GPTQ, torch.float16, 256)
    x_0 = torch.randn((1, 1, K), dtype=torch.float16, device="cuda")
    x_1 = torch.randn((1, 1, K), dtype=torch.float16, device="cuda")

    with torch.inference_mode():
        expected_0 = module.adapter.apply(
            x_0,
            trilin_matmul(x_0.reshape(1, K), module.qweight, module.scales, module.bias).reshape(1, 1, N),
        )
        expected_1 = module.adapter.apply(
            x_1,
            trilin_matmul(x_1.reshape(1, K), module.qweight, module.scales, module.bias).reshape(1, 1, N),
        )

    stream_0 = torch.cuda.Stream(device=x_0.device)
    stream_1 = torch.cuda.Stream(device=x_0.device)
    with torch.inference_mode():
        for _ in range(8):
            with torch.cuda.stream(stream_0):
                actual_0 = module(x_0)
            with torch.cuda.stream(stream_1):
                actual_1 = module(x_1)
    stream_0.synchronize()
    stream_1.synchronize()

    torch.testing.assert_close(actual_0, expected_0, rtol=0.002, atol=0.0625)
    torch.testing.assert_close(actual_1, expected_1, rtol=0.002, atol=0.0625)
    workspaces = module.adapter._trilin_eora_stream_workspaces
    assert workspaces[(x_0.get_device(), stream_0.cuda_stream)].data_ptr() != workspaces[
        (x_0.get_device(), stream_1.cuda_stream)
    ].data_ptr()


def test_trilin_eora_unsupported_rank_preserves_standard_adapter_fallback():
    _require_sm80()
    pytest.importorskip("triton")
    torch.manual_seed(103)
    module = _quant_linear(LAYOUT_GPTQ, torch.float16, 96)
    assert not hasattr(module, "_trilin_eora_workspace")

    original_apply = module.adapter.apply
    apply_calls = 0

    def tracked_apply(*args, **kwargs):
        nonlocal apply_calls
        apply_calls += 1
        return original_apply(*args, **kwargs)

    module.adapter.apply = tracked_apply
    x = torch.randn((1, 1, K), dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        expected = original_apply(
            x,
            trilin_matmul(x.reshape(1, K), module.qweight, module.scales, module.bias).reshape(1, 1, N),
        )
        actual = module(x)

    assert apply_calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_trilin_eora_rejects_cpu_without_allocating_workspace():
    adapter = Lora(rank=4, lora_A=torch.randn(8, 4), lora_B=torch.randn(4, 16))
    assert prepare_trilin_eora_3bit(
        adapter,
        device=torch.device("cpu"),
        in_features=8,
        out_features=16,
        group_size=128,
    ) is None
