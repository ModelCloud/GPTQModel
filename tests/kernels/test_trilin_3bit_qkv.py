# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.trilin import AwqTrilinLinear, TrilinLinear
from gptqmodel.nn_modules.triton_utils.three_bit import pack_3bit
from gptqmodel.nn_modules.triton_utils.trilin_qkv import install_trilin_3bit_qkv
from gptqmodel.utils.trilin import trilin_matmul, trilin_qkv


K = 4096
Q_SIZE = 4096
GROUP_SIZE = 128
ZERO = 4
_CACHE_ATTRIBUTE = "_gptqmodel_trilin_qkv_cache"
_SM80_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="the fused Trilin QKV path is profiled and enabled only on sm80",
)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = actual_fp32 - reference_fp32
    rmse = difference.square().mean().sqrt()
    reference_rms = reference_fp32.square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_rmse": (rmse / reference_rms).item(),
        "cosine": F.cosine_similarity(actual_fp32.flatten(), reference_fp32.flatten(), dim=0).item(),
    }


def _make_projection(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = torch.randint(0, 8, (K, n), dtype=torch.int8, device=device)
    qweight = pack_3bit(codes, axis=0)
    scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
    return codes, qweight, scales


def _projection_references(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense = (codes.float() - ZERO) * scales.repeat_interleave(GROUP_SIZE, dim=0).float()
    torch_output = torch.matmul(x, dense.to(x.dtype)).to(x.dtype)
    fp32_output = torch.matmul(x.float(), dense).to(x.dtype)
    return torch_output, fp32_output


@_SM80_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_size", [1024, 4096])
def test_trilin_3bit_qkv_matches_control_torch_3bit_and_fp32_references(
    dtype: torch.dtype,
    kv_size: int,
):
    torch.manual_seed(509)
    device = torch.device("cuda")
    projections = (
        _make_projection(Q_SIZE, device),
        _make_projection(kv_size, device),
        _make_projection(kv_size, device),
    )
    x = torch.randn((1, K), device=device, dtype=dtype)
    actual = trilin_qkv(
        x,
        projections[0][1],
        projections[0][2],
        projections[1][1],
        projections[1][2],
        projections[2][1],
        projections[2][2],
    )
    control = tuple(trilin_matmul(x, qweight, scales) for _, qweight, scales in projections)

    assert tuple(output.shape for output in actual) == ((1, Q_SIZE), (1, kv_size), (1, kv_size))
    assert all(output.dtype == dtype for output in actual)
    for candidate, expected in zip(actual, control):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)

    combined = actual[0]._base
    assert combined is not None
    assert all(output._base is combined for output in actual)
    assert combined.shape == (1, Q_SIZE + 2 * kv_size)
    assert [output.storage_offset() for output in actual] == [0, Q_SIZE, Q_SIZE + kv_size]

    torch_limits = {
        torch.float16: {"max_abs": 0.125, "mean_abs": 0.02, "relative_rmse": 0.001},
        torch.bfloat16: {"max_abs": 1.0, "mean_abs": 0.125, "relative_rmse": 0.004},
    }[dtype]
    oracle_limits = {
        torch.float16: {"max_abs": 0.0625, "mean_abs": 0.0005, "relative_rmse": 0.0001},
        torch.bfloat16: {"max_abs": 0.5, "mean_abs": 0.005, "relative_rmse": 0.0005},
    }[dtype]
    for candidate, (codes, _, scales) in zip(actual, projections):
        torch_reference, fp32_reference = _projection_references(x, codes, scales)
        torch_metrics = _metrics(candidate, torch_reference)
        assert torch_metrics["max_abs"] <= torch_limits["max_abs"], torch_metrics
        assert torch_metrics["mean_abs"] <= torch_limits["mean_abs"], torch_metrics
        assert torch_metrics["relative_rmse"] <= torch_limits["relative_rmse"], torch_metrics
        assert torch_metrics["cosine"] >= 0.99999, torch_metrics

        oracle_metrics = _metrics(candidate, fp32_reference)
        torch_oracle_metrics = _metrics(torch_reference, fp32_reference)
        assert oracle_metrics["max_abs"] <= oracle_limits["max_abs"], oracle_metrics
        assert oracle_metrics["mean_abs"] <= oracle_limits["mean_abs"], oracle_metrics
        assert oracle_metrics["relative_rmse"] <= oracle_limits["relative_rmse"], oracle_metrics
        assert oracle_metrics["cosine"] >= 0.999999, oracle_metrics
        assert oracle_metrics["mean_abs"] <= torch_oracle_metrics["mean_abs"] + 1e-6
        assert oracle_metrics["relative_rmse"] <= torch_oracle_metrics["relative_rmse"] + 1e-7

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_outputs = trilin_qkv(
            x,
            projections[0][1],
            projections[0][2],
            projections[1][1],
            projections[1][2],
            projections[2][1],
            projections[2][2],
        )
    torch.cuda.current_stream().wait_stream(stream)
    for stream_output, expected in zip(stream_outputs, actual):
        torch.testing.assert_close(stream_output, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = trilin_qkv(
            x,
            projections[0][1],
            projections[0][2],
            projections[1][1],
            projections[1][2],
            projections[2][1],
            projections[2][2],
        )
    graph.replay()
    for graph_output, expected in zip(graph_outputs, actual):
        torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)


def _fake_projection_forward(projection: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    qweight = getattr(projection, "_triton_3bit_qweight", projection.qweight)
    values = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    output = trilin_matmul(values, qweight, projection.scales)
    return output.reshape(*hidden_states.shape[:-1], projection.out_features)


def _fake_projection(
    projection_type: type[torch.nn.Module],
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.nn.Module:
    projection = projection_type.__new__(projection_type)
    torch.nn.Module.__init__(projection)
    projection.bits = 3
    projection.group_size = GROUP_SIZE
    projection.desc_act = False
    projection.sym = True
    projection.in_features = K
    projection.out_features = scales.shape[1]
    projection.bias = None
    projection.adapter = None
    projection._trilin_native_3bit = True
    projection.register_buffer("qweight", qweight)
    projection.register_buffer("scales", scales)
    if projection_type is AwqTrilinLinear:
        projection.register_buffer("_triton_3bit_qweight", qweight)
    projection.forward = MethodType(_fake_projection_forward, projection)
    return projection


class LlamaAttention(torch.nn.Module):
    def __init__(self, q_proj: torch.nn.Module, k_proj: torch.nn.Module, v_proj: torch.nn.Module):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj


class MistralAttention(LlamaAttention):
    pass


class _AttentionModel(torch.nn.Module):
    def __init__(self, attention: torch.nn.Module, *, model_type: str = "llama"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.attention = attention


def _make_attention(
    projection_type: type[torch.nn.Module],
    *,
    kv_size: int,
    attention_type: type[LlamaAttention] = LlamaAttention,
) -> LlamaAttention:
    device = torch.device("cuda")

    def make_projection(n: int) -> torch.nn.Module:
        qweight = torch.randint(
            -(1 << 31),
            (1 << 31) - 1,
            (K // 32 * 3, n),
            device=device,
            dtype=torch.int32,
        )
        scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
        return _fake_projection(projection_type, qweight, scales)

    return attention_type(make_projection(Q_SIZE), make_projection(kv_size), make_projection(kv_size))


@_SM80_REQUIRED
@pytest.mark.parametrize("projection_type", [TrilinLinear, AwqTrilinLinear])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trilin_3bit_qkv_installer_routes_gptq_and_awq_decode_and_falls_back_for_prefill(
    projection_type: type[torch.nn.Module],
    dtype: torch.dtype,
):
    torch.manual_seed(521)
    attention = _make_attention(projection_type, kv_size=1024)
    model = _AttentionModel(attention).eval()
    decode_input = torch.randn((1, 1, K), device="cuda", dtype=dtype)
    decode_reference = tuple(
        projection(decode_input) for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )

    assert install_trilin_3bit_qkv(model) == 1
    q_output = attention.q_proj(decode_input)
    assert getattr(decode_input, _CACHE_ATTRIBUTE) is not None
    k_output = attention.k_proj(decode_input)
    assert getattr(decode_input, _CACHE_ATTRIBUTE) is not None
    v_output = attention.v_proj(decode_input)
    assert getattr(decode_input, _CACHE_ATTRIBUTE) is None
    for candidate, expected in zip((q_output, k_output, v_output), decode_reference):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)

    prefill_input = torch.randn((1, 2, K), device="cuda", dtype=dtype)
    for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
        candidate = projection(prefill_input)
        expected = projection._gptqmodel_trilin_qkv_original_forward(prefill_input)
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)


@_SM80_REQUIRED
def test_trilin_3bit_qkv_installer_accepts_exact_mistral_mha_and_rejects_mixed_backends():
    attention = _make_attention(TrilinLinear, kv_size=4096, attention_type=MistralAttention)
    model = _AttentionModel(attention, model_type="mistral").eval()
    assert install_trilin_3bit_qkv(model) == 1

    mixed_attention = _make_attention(TrilinLinear, kv_size=1024)
    mixed_attention.v_proj = _make_attention(AwqTrilinLinear, kv_size=1024).v_proj
    mixed_model = _AttentionModel(mixed_attention).eval()
    assert install_trilin_3bit_qkv(mixed_model) == 0
