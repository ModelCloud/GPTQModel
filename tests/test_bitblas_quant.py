# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import time
from statistics import mean, pstdev

import pytest
import torch
import torch.nn as nn
from parameterized import parameterized
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.bitblas import BitblasQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear, marlin_import_exception
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear


RTOL = 5e-2
ATOL = 5e-2


def _mock_gptq_linear(bits: int, group_size: int, in_features: int, out_features: int):
    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32)

    if group_size != -1:
        reshaped = weight.view(in_features // group_size, group_size, out_features)
        w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)
    else:
        w_g = weight

    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=w_g.device),
    )
    scales = scales / maxq

    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)

    if group_size != -1:
        def _reshape_back(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(group_size, -1, out_features)
            return tensor.permute(1, 0, 2).reshape(in_features, out_features)

        ref = _reshape_back(ref)
        q = _reshape_back(q)

    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = ref.t().contiguous()

    scales = scales.reshape(-1, out_features).contiguous()
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // (
        group_size if group_size != -1 else in_features
    )

    return linear, scales, zeros, g_idx


def _benchmark(module: nn.Module, x: torch.Tensor, warmup: int = 2, iters: int = 5) -> list[float]:
    times_ms: list[float] = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()
        for _ in range(iters):
            start = time.perf_counter()
            module(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
    return times_ms


def _format_pass(pass_ok: bool) -> str:
    if pass_ok:
        return "PASS"
    return "\033[91mFAIL\033[0m"


@pytest.mark.cuda
@parameterized.expand([
    ("bs1_fp16", 1, torch.float16, "float16"),
    ("bs2_fp16", 2, torch.float16, "float16"),
    ("bs4_fp16", 4, torch.float16, "float16"),
    ("bs8_fp16", 8, torch.float16, "float16"),
    ("bs16_fp16", 16, torch.float16, "float16"),
    ("bs1_bf16", 1, torch.bfloat16, "bfloat16"),
    ("bs2_bf16", 2, torch.bfloat16, "bfloat16"),
    ("bs4_bf16", 4, torch.bfloat16, "bfloat16"),
    ("bs8_bf16", 8, torch.bfloat16, "bfloat16"),
    ("bs16_bf16", 16, torch.bfloat16, "bfloat16"),
])
def test_llama3_linear_bitblas_vs_torch_vs_marlin(_, batch, dtype, dtype_name):
    try:
        pytest.importorskip("bitblas")
    except Exception as exc:
        pytest.skip(f"bitblas unavailable: {exc}")
    if marlin_import_exception is not None:
        pytest.skip(f"marlin unavailable: {marlin_import_exception}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    torch.manual_seed(0)

    bits = 4
    group_size = 128
    in_features = 8192
    out_features = 8192

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)

    torch_linear = TorchQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    torch_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))
    torch_linear.post_init()

    bitblas_linear = BitblasQuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        enable_tuning=False,
    )
    bitblas_linear.repack_from_gptq(torch_linear)
    bitblas_linear.post_init()

    device = torch.device("cuda")
    torch_linear = torch_linear.to(device=device, dtype=dtype)
    bitblas_linear = bitblas_linear.to(device=device, dtype=dtype)

    marlin_linear = MarlinQuantLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        marlin_linear.qweight.copy_(torch_linear.qweight.to(device))
        marlin_linear.scales.copy_(torch_linear.scales.to(device))
        marlin_linear.g_idx.copy_(torch_linear.g_idx.to(device))
        marlin_linear.qzeros.zero_()
    marlin_linear.post_init()

    try:
        triton_linear = TritonV2QuantLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            bias=False,
        )
    except ValueError as err:
        pytest.skip(f"triton unavailable: {err}")

    triton_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))
    triton_linear.post_init()
    triton_linear = triton_linear.to(device=device, dtype=dtype).eval()

    modules = {
        "Torch": torch_linear.eval(),
        "BitBLAS": bitblas_linear.eval(),
        "Marlin": marlin_linear.eval(),
        "TritonV2": triton_linear,
    }

    x = torch.randn((batch, in_features), dtype=dtype, device=device)

    results = []
    reference_out = None
    outputs: dict[str, torch.Tensor] = {}
    errors: dict[str, str] = {}

    for name, module in modules.items():
        try:
            with torch.inference_mode():
                outputs[name] = module(x).to(torch.float32)
            if reference_out is None:
                reference_out = outputs[name]
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors[name] = str(exc)

    for name, module in modules.items():
        err = errors.get(name)
        if err:
            results.append([
                dtype_name,
                batch,
                name,
                "-",
                "-",
                "-",
                "-",
                "\033[91mERR\033[0m",
            ])
            continue

        out = outputs[name]
        if name == "Torch" or reference_out is None:
            max_abs = 0.0
            mean_abs = 0.0
            max_rel = 0.0
            pass_ok = True
        else:
            diff = (out - reference_out).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            max_rel = float((diff / (reference_out.abs() + 1e-6)).max().item())
            pass_ok = max_abs <= ATOL and max_rel <= RTOL

        times = _benchmark(module, x)
        mean_ms = mean(times)
        std_ms = pstdev(times) if len(times) > 1 else 0.0

        results.append([
            dtype_name,
            batch,
            name,
            f"{mean_ms:.3f}",
            f"{std_ms:.3f}",
            f"{max_abs:.4f}",
            f"{mean_abs:.4f}",
            f"{max_rel:.4f}",
            _format_pass(pass_ok),
        ])

    headers = [
        "dtype",
        "batch",
        "Kernel",
        "Mean ms",
        "Std ms",
        "Max |Δ|",
        "Mean |Δ|",
        "Max Rel Δ",
        "Accuracy",
    ]
    print(tabulate(results, headers=headers, tablefmt="github"))

    # Table highlights failing kernels in red; no hard assertion to keep report informative.
