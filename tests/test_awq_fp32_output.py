# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ GEMM dispatch and fp32-output accuracy (issue #3984).

The JIT path switched to dequantise + matmul only above 1024 rows. From 640
rows the dequantise + fp32-output GEMM is faster and at least as accurate as
the native split-K kernel, but the fp32 output is what keeps the bias add from
rounding twice. At or below 512 rows the native kernel still wins.
"""

from __future__ import annotations

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq
from gptqmodel.utils.awq import awq_dequantize_weights, awq_runtime_available


def _patch_fake_backend(monkeypatch, weight, native):
    calls = {"dequant": 0, "gemm": 0}

    def fake_dequant(qweight, scales, qzeros, *_args):
        calls["dequant"] += 1
        return weight

    def fake_gemm(x, qweight, scales, qzeros, *_args, **_kwargs):
        calls["gemm"] += 1
        return native(x)

    monkeypatch.setattr(gemm_awq, "awq_dequantize_weights", fake_dequant)
    monkeypatch.setattr(gemm_awq, "_awq_cuda_gemm_forward", fake_gemm)
    return gemm_awq.AwqGemmFn, calls


def _dummy_quant_tensors(in_features=8, out_features=4):
    return (
        torch.zeros((in_features, max(1, out_features // 8)), dtype=torch.int32),
        torch.zeros((1, out_features), dtype=torch.float16),
        torch.zeros((1, max(1, out_features // 8)), dtype=torch.int32),
    )


def test_640_rows_uses_fp32_output_mm(monkeypatch):
    x = torch.randn((640, 8), dtype=torch.float16)
    weight = torch.randn((8, 4), dtype=torch.float16)
    bias = torch.randn((4,), dtype=torch.float16)
    qweight, scales, qzeros = _dummy_quant_tensors()

    fn, calls = _patch_fake_backend(monkeypatch, weight, lambda _x: torch.zeros((_x.shape[0], 4)))
    out = fn.apply(x, qweight, qzeros, scales, 4, 32, bias, 4)

    expected = (x.to(torch.float32) @ weight.to(torch.float32) + bias.to(torch.float32)).to(torch.float16)
    assert calls == {"dequant": 1, "gemm": 0}
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_639_rows_keeps_the_native_kernel(monkeypatch):
    x = torch.randn((639, 8), dtype=torch.float16)
    weight = torch.randn((8, 4), dtype=torch.float16)
    qweight, scales, qzeros = _dummy_quant_tensors()

    fn, calls = _patch_fake_backend(monkeypatch, weight, lambda _x: torch.zeros((_x.shape[0], 4), dtype=torch.float16))
    fn.apply(x, qweight, qzeros, scales, 4, 32, None, 4)

    assert calls == {"dequant": 0, "gemm": 1}


def test_640_rows_adds_bias_in_fp32_before_the_cast(monkeypatch):
    # x @ W == 1 + 2^-11, which fp16 rounds to 1.0. Adding the 2^-11 bias in
    # fp32 first gives 1 + 2^-10, which fp16 represents exactly; adding it
    # after an fp16 matmul rounds twice and stays at 1.0.
    x = torch.ones((640, 2), dtype=torch.float16)
    weight = torch.tensor([[1.0], [2.0**-11]], dtype=torch.float16)
    bias = torch.tensor([2.0**-11], dtype=torch.float16)
    qweight, scales, qzeros = _dummy_quant_tensors(in_features=2, out_features=1)

    def native(_x):
        rounded = (_x.to(torch.float32) @ weight.to(torch.float32)).to(torch.float16)
        return (rounded + bias).to(torch.float16)

    fn, calls = _patch_fake_backend(monkeypatch, weight, native)
    out = fn.apply(x, qweight, qzeros, scales, 4, 32, bias, 1)

    assert calls == {"dequant": 1, "gemm": 0}
    assert out[0, 0].item() == 1.0009765625


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AWQ kernel required")
@pytest.mark.skipif(not awq_runtime_available(), reason="native AWQ extension unavailable")
def test_awq_fp32_output_matches_fp64_oracle_and_native_accuracy():
    torch.manual_seed(0)
    device = torch.device("cuda")
    in_features, out_features, group_size = 256, 128, 32
    rows = 640

    x = torch.randn((rows, in_features), device=device, dtype=torch.float16)
    qweight = torch.randint(0, 16, (in_features, out_features // 8), device=device, dtype=torch.int32)
    scales = (torch.rand((in_features // group_size, out_features), device=device, dtype=torch.float16) + 0.5)
    qzeros = torch.zeros((in_features // group_size, out_features // 8), device=device, dtype=torch.int32)
    bias = torch.randn((out_features,), device=device, dtype=torch.float16)

    weight = awq_dequantize_weights(qweight, scales, qzeros, 0, 0, 0, False)
    reference = x.to(torch.float64) @ weight.to(torch.float64) + bias.to(torch.float64)

    new_out = gemm_awq.AwqGemmFn.apply(x, qweight, qzeros, scales, 4, group_size, bias, out_features, "cuda", True)
    native_out = gemm_awq._awq_cuda_gemm_forward(x, qweight, scales, qzeros, 8, fp32_accum=True)
    native_out = native_out + bias

    def error_stats(out):
        err = (out.to(torch.float64) - reference).abs()
        abs_rms = err.pow(2).mean().sqrt().item()
        rel_rms = (err.norm() / reference.norm()).item()
        max_rel = (err / reference.abs().clamp_min(1e-6)).max().item()
        return abs_rms, rel_rms, max_rel

    new_rms, new_rel, new_max = error_stats(new_out)
    native_rms, native_rel, native_max = error_stats(native_out)
    print(
        f"fp32-output abs_rms={new_rms:.6e} rel_rms={new_rel:.6e} max_rel={new_max:.6e} | "
        f"native abs_rms={native_rms:.6e} rel_rms={native_rel:.6e} max_rel={native_max:.6e}"
    )

    assert new_rel <= native_rel * 1.02, (new_rel, native_rel)
    assert new_max <= native_max * 1.02, (new_max, native_max)
    assert new_rel <= 2e-3
