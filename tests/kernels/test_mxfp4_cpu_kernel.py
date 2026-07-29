# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the experimental MXFP4 CPU kernel FP16 fast path.

The native AVX512-FP16 path uses a 2x8 / 4x4 register tile plus scalar tails, so
the shapes below deliberately exercise partial M blocks and N columns that do
not fill a tile.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = REPO_ROOT / "scripts" / "benchmark_mxfp4_cpu_kernel.py"


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("benchmark_mxfp4_cpu_kernel", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    if not BENCH_PATH.exists():
        pytest.skip(f"{BENCH_PATH} is missing")
    return _load_bench_module()


@pytest.fixture(scope="module")
def ext(bench):
    try:
        return bench._mxfp4_extension()
    except Exception as exc:  # noqa: BLE001 - build failures should skip, not fail
        pytest.skip(f"could not build mxfp4_cpu_kernel: {exc}")


@pytest.mark.parametrize(
    ("M", "K", "N"),
    [
        (1, 128, 16),   # single row, exact tiles
        (3, 256, 18),   # partial M block and a 2-column N tail
        (8, 512, 12),   # wide M block
        (13, 96, 7),    # partial M block and an odd N tail
    ],
)
def test_fp16_path_matches_fp32_accumulation(bench, ext, M, K, N):
    torch.manual_seed(0)
    weight = torch.randn(N, K, dtype=torch.float32)
    qweight, scales = bench.quantize_mxfp4(weight)
    x = bench.make_activation(M, K, torch.float16, seed=1234)

    legacy = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 1).to(torch.float32)
    for variant in (0, 2, 3, 4):
        native = ext.mxfp4_linear_cpu(x, qweight, scales, 0, variant).to(torch.float32)
        assert native.shape == (M, N)
        torch.testing.assert_close(native, legacy, rtol=2e-2, atol=2e-2, msg=f"variant={variant}")


def test_fp16_path_falls_back_for_out_of_range_scales(bench, ext):
    """E8M0 exponents outside the FP16 normal range must not reach the FP16 tables."""
    M, K, N = 4, 128, 8
    torch.manual_seed(0)
    weight = torch.randn(N, K, dtype=torch.float32)
    qweight, scales = bench.quantize_mxfp4(weight)
    scales = scales.clone()
    scales[0, 0] = 200  # 2^73, overflows FP16
    x = bench.make_activation(M, K, torch.float16, seed=7)

    out = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 0).to(torch.float32)
    legacy = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 1).to(torch.float32)
    torch.testing.assert_close(out, legacy, rtol=0, atol=0)


def test_fp16_path_falls_back_for_overflowing_products(bench, ext):
    """Representable weights are not enough: the FP16 partial sums must fit too.

    Scale byte 130 is inside the FP16 window (114..140), so the dequantized
    weights (+-6 * 2^3 = +-48) are exact, and the true product is exactly zero:
    the first half of the reduction is +48 and the second half -48.  But an
    FP16 lane accumulating the first half alone reaches 32 * 48 * 64 = 98304 and
    saturates to inf (verified: the kernel returns inf without the guard below).
    The kernel must notice that ``flush_groups * max|w| * max|a|`` exceeds 65504
    and fall back to FP32 accumulation.
    """
    M, K, N = 4, 2048, 16
    half = K // 4  # half of the packed bytes in each row
    qweight = torch.cat(
        [
            torch.full((N, half), 0x77, dtype=torch.uint8),  # fp4 code 7 -> +6.0
            torch.full((N, half), 0xFF, dtype=torch.uint8),  # fp4 code 15 -> -6.0
        ],
        dim=1,
    )
    scales = torch.full((N, K // 32), 130, dtype=torch.uint8)  # 2^3
    x = torch.full((M, K), 64.0, dtype=torch.float16)

    out = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 0).to(torch.float32)
    torch.testing.assert_close(out, torch.zeros(M, N), rtol=0, atol=0)


def test_fp16_path_falls_back_for_non_finite_activations(bench, ext):
    M, K, N = 2, 256, 16
    torch.manual_seed(0)
    weight = torch.randn(N, K, dtype=torch.float32)
    qweight, scales = bench.quantize_mxfp4(weight)
    x = bench.make_activation(M, K, torch.float16, seed=5)
    x[0, 0] = float("inf")

    out = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 0).to(torch.float32)
    legacy = ext.mxfp4_linear_cpu(x, qweight, scales, 0, 1).to(torch.float32)
    torch.testing.assert_close(out, legacy, rtol=0, atol=0, equal_nan=True)


def test_thread_count_does_not_change_results(bench, ext):
    M, K, N = 8, 1024, 64
    torch.manual_seed(0)
    weight = torch.randn(N, K, dtype=torch.float32)
    qweight, scales = bench.quantize_mxfp4(weight)
    x = bench.make_activation(M, K, torch.float16, seed=99)

    single = ext.mxfp4_linear_cpu(x, qweight, scales, 1, 0).to(torch.float32)
    many = ext.mxfp4_linear_cpu(x, qweight, scales, 8, 0).to(torch.float32)
    torch.testing.assert_close(many, single, rtol=0, atol=0)
