# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.trilin import AwqTrilinLinear, TrilinLinear
from gptqmodel.nn_modules.triton_utils.three_bit import pack_3bit
from gptqmodel.nn_modules.triton_utils.trilin_swiglu import install_trilin_3bit_swiglu
from gptqmodel.utils.trilin import trilin_matmul, trilin_silu_mul


K = 4096
GROUP_SIZE = 128
ZERO = 4
_SM80_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="the fused Trilin SwiGLU path is profiled and enabled only on sm80",
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


def _projection_references(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense = (codes.float() - ZERO) * scales.repeat_interleave(GROUP_SIZE, dim=0).float()
    torch_output = torch.matmul(x, dense.to(x.dtype)).to(x.dtype)
    fp32_output = torch.matmul(x.float(), dense)
    return torch_output, fp32_output


@_SM80_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [11008, 14336])
def test_trilin_3bit_swiglu_matches_torch_3bit_and_fp32_references(dtype: torch.dtype, n: int):
    torch.manual_seed(317)
    device = torch.device("cuda")
    gate_codes = torch.randint(0, 8, (K, n), dtype=torch.int8, device=device)
    up_codes = torch.randint(0, 8, (K, n), dtype=torch.int8, device=device)
    gate_qweight = pack_3bit(gate_codes, axis=0)
    up_qweight = pack_3bit(up_codes, axis=0)
    gate_scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
    up_scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
    x = torch.randn((1, K), device=device, dtype=dtype)

    gate_control = trilin_matmul(x, gate_qweight, gate_scales)
    up_control = trilin_matmul(x, up_qweight, up_scales)
    control = (F.silu(gate_control) * up_control).to(dtype)
    actual = trilin_silu_mul(x, gate_qweight, gate_scales, up_qweight, up_scales)

    gate_torch, gate_fp32 = _projection_references(x, gate_codes, gate_scales)
    up_torch, up_fp32 = _projection_references(x, up_codes, up_scales)
    torch_reference = (F.silu(gate_torch) * up_torch).to(dtype)
    fp32_reference = (F.silu(gate_fp32) * up_fp32).to(dtype)

    assert actual.shape == (1, n)
    assert actual.dtype == dtype
    control_metrics = _metrics(actual, control)
    control_limits = {
        torch.float16: {"max_abs": 0.015625, "mean_abs": 0.0001, "relative_rmse": 0.00001},
        torch.bfloat16: {"max_abs": 0.0625, "mean_abs": 0.001, "relative_rmse": 0.00001},
    }[dtype]
    assert control_metrics["max_abs"] <= control_limits["max_abs"], control_metrics
    assert control_metrics["mean_abs"] <= control_limits["mean_abs"], control_metrics
    assert control_metrics["relative_rmse"] <= control_limits["relative_rmse"], control_metrics
    assert control_metrics["cosine"] >= 0.999999, control_metrics

    torch_metrics = _metrics(actual, torch_reference)
    torch_limits = {
        torch.float16: {"max_abs": 4.0, "mean_abs": 0.1, "relative_rmse": 0.001},
        torch.bfloat16: {"max_abs": 32.0, "mean_abs": 1.0, "relative_rmse": 0.008},
    }[dtype]
    assert torch_metrics["max_abs"] <= torch_limits["max_abs"], torch_metrics
    assert torch_metrics["mean_abs"] <= torch_limits["mean_abs"], torch_metrics
    assert torch_metrics["relative_rmse"] <= torch_limits["relative_rmse"], torch_metrics
    assert torch_metrics["cosine"] >= 0.99995, torch_metrics

    actual_oracle_metrics = _metrics(actual, fp32_reference)
    torch_oracle_metrics = _metrics(torch_reference, fp32_reference)
    assert actual_oracle_metrics["mean_abs"] <= torch_oracle_metrics["mean_abs"] + 1e-6, {
        "actual": actual_oracle_metrics,
        "torch": torch_oracle_metrics,
    }
    assert actual_oracle_metrics["relative_rmse"] <= torch_oracle_metrics["relative_rmse"] + 1e-7, {
        "actual": actual_oracle_metrics,
        "torch": torch_oracle_metrics,
    }

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_output = trilin_silu_mul(x, gate_qweight, gate_scales, up_qweight, up_scales)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(stream_output, actual, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = trilin_silu_mul(x, gate_qweight, gate_scales, up_qweight, up_scales)
    graph.replay()
    torch.testing.assert_close(graph_output, actual, rtol=0, atol=0)


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


class LlamaMLP(torch.nn.Module):
    def __init__(self, gate_proj: torch.nn.Module, up_proj: torch.nn.Module):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = torch.nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MistralMLP(LlamaMLP):
    pass


class _LlamaModel(torch.nn.Module):
    def __init__(self, mlp: LlamaMLP, *, model_type: str = "llama"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, hidden_act="silu")
        self.mlp = mlp


@_SM80_REQUIRED
@pytest.mark.parametrize("projection_type", [TrilinLinear, AwqTrilinLinear])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_trilin_3bit_swiglu_installer_routes_gptq_and_awq_decode_and_falls_back_for_prefill(
    projection_type: type[torch.nn.Module],
    dtype: torch.dtype,
):
    torch.manual_seed(401)
    device = torch.device("cuda")
    n = 11008
    qweight_shape = (K // 32 * 3, n)
    gate_qweight = torch.randint(-(1 << 31), (1 << 31) - 1, qweight_shape, device=device, dtype=torch.int32)
    up_qweight = torch.randint(-(1 << 31), (1 << 31) - 1, qweight_shape, device=device, dtype=torch.int32)
    gate_scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
    up_scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
    mlp = LlamaMLP(
        _fake_projection(projection_type, gate_qweight, gate_scales),
        _fake_projection(projection_type, up_qweight, up_scales),
    )
    model = _LlamaModel(mlp).eval()

    decode_input = torch.randn((1, 1, K), device=device, dtype=dtype)
    decode_reference = mlp(decode_input)
    assert install_trilin_3bit_swiglu(model) == 1
    decode_actual = mlp(decode_input)
    decode_metrics = _metrics(decode_actual, decode_reference)
    assert decode_metrics["max_abs"] <= (0.015625 if dtype == torch.float16 else 0.0625), decode_metrics
    assert decode_metrics["mean_abs"] <= (0.0001 if dtype == torch.float16 else 0.001), decode_metrics
    assert decode_metrics["relative_rmse"] <= 0.00001, decode_metrics
    assert decode_metrics["cosine"] >= 0.999999, decode_metrics

    prefill_input = torch.randn((1, 2, K), device=device, dtype=dtype)
    prefill_actual = mlp(prefill_input)
    prefill_reference = mlp._gptqmodel_trilin_swiglu_original_forward(prefill_input)
    torch.testing.assert_close(prefill_actual, prefill_reference, rtol=0, atol=0)


@_SM80_REQUIRED
def test_trilin_3bit_swiglu_installer_accepts_exact_mistral_mlp():
    device = torch.device("cuda")
    n = 14336
    qweight_shape = (K // 32 * 3, n)
    gate_qweight = torch.zeros(qweight_shape, device=device, dtype=torch.int32)
    up_qweight = torch.zeros(qweight_shape, device=device, dtype=torch.int32)
    gate_scales = torch.ones((K // GROUP_SIZE, n), device=device, dtype=torch.float16)
    up_scales = torch.ones((K // GROUP_SIZE, n), device=device, dtype=torch.float16)
    mlp = MistralMLP(
        _fake_projection(TrilinLinear, gate_qweight, gate_scales),
        _fake_projection(TrilinLinear, up_qweight, up_scales),
    )
    model = _LlamaModel(mlp, model_type="mistral").eval()

    assert install_trilin_3bit_swiglu(model) == 1
