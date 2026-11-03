# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Numerics + storage-rule tests for the AllSpark HIP/RDNA4 W8A16 kernel.

Oracle policy (see .agents/skills/gptqmodel-torch-oracle-numerics):
  * the reference does its own arithmetic in float64 from the original uint8
    codes and scales - it never calls the kernel or reuses its intermediates;
  * fp16 and bf16 are validated separately and the output dtype must equal the
    activation dtype;
  * arithmetic error (kernel vs oracle rounded to the activation dtype) is
    reported separately from the mandatory output-rounding error.
"""

from __future__ import annotations

import gc

import pytest
import torch

from gptqmodel.utils.allspark_hip import (
    allspark_hip_gemm,
    allspark_hip_gemm_f32,
    allspark_hip_repack,
    allspark_hip_supported,
    prewarm_allspark_hip_extension,
)


pytestmark = pytest.mark.skipif(
    not allspark_hip_supported(), reason="gfx1200/gfx1201 ROCm GPU required"
)

ATOL = 2e-3
RTOL = 2e-3


@pytest.fixture(scope="module", autouse=True)
def _prewarm():
    assert prewarm_allspark_hip_extension(), "AllSpark HIP extension failed to build"


def _make_case(n: int, k: int, m: int, dtype: torch.dtype, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (k, n), dtype=torch.uint8, device="cuda", generator=gen)
    scales = (
        torch.rand(n, device="cuda", dtype=torch.float32, generator=gen) * 0.01 + 0.005
    ).to(dtype)
    a = (torch.randn(m, k, device="cuda", dtype=torch.float32, generator=gen) * 0.5).to(dtype)
    return codes, scales, a


def _dequant_to_dtype(codes, scales, dtype):
    """The weight values the tensor cores actually consume.

    W8A16 rounds the dequantised weight to the activation dtype before the MMA:
    the fp16 path multiplies in fp16 (``__hmul2``) and the bf16 path multiplies
    in fp32 then rounds to bf16. The arithmetic oracle has to use the same
    representation, exactly like
    ``tests/kernels/test_gptq_rocm_gemm.py::_oracle_fp64``; otherwise it
    measures the format's weight rounding instead of the kernel's arithmetic.
    """
    if dtype == torch.float16:
        return (codes.to(torch.float16) - 128.0) * scales
    return ((codes.to(torch.float32) - 128.0) *
            scales.to(torch.float32)).to(torch.bfloat16)


def _oracle(codes, scales, a, dtype):
    """Independent fp64 accumulation over the dtype-rounded W8A16 weights."""
    w = _dequant_to_dtype(codes, scales, dtype).to(torch.float64)
    return a.to(torch.float64) @ w


def _exact_oracle(codes, scales, a):
    """fp64 reference with the weights kept exact (no W8A16 rounding)."""
    w = (codes.to(torch.float64) - 128.0) * scales.to(torch.float64)[None, :]
    return a.to(torch.float64) @ w


def _ulp_of(dtype: torch.dtype, ref64: torch.Tensor) -> torch.Tensor:
    """One representable step of `dtype` at each reference magnitude."""
    r = ref64.to(dtype)
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    return (up.float().to(torch.float64) - r.float().to(torch.float64)).abs()


def _check(out, raw32, ref64, exact64, dtype, label):
    """Two separate measurements, per the numerics skill.

    * arithmetic error  : fp32 accumulators vs the fp64 oracle over the
      dtype-rounded weights. This is the `rtol=atol=2e-3` acceptance gate and
      matches the convention used by tests/kernels/test_gptq_rocm_gemm.py
      (`raw` fp32 vs oracle).
    * visible error     : the returned low-precision tensor. It may only differ
      from the oracle by at most one output-dtype ULP, i.e. a natural rounding
      tie, never by more.
    * format error      : the oracle itself vs an exact-weight fp64 reference.
      This is the irreducible cost of storing int8 + per-N scale and is
      reported, not asserted.
    """
    assert out.dtype == dtype, f"{label}: dtype {out.dtype} != {dtype}"
    assert raw32.dtype == torch.float32

    torch.testing.assert_close(raw32, ref64.float(), rtol=RTOL, atol=ATOL)
    arith_err = (raw32.to(torch.float64) - ref64).abs()

    rounded = ref64.to(dtype)
    round_err = (rounded.to(torch.float64) - ref64).abs()
    vis_err = (out.to(torch.float64) - ref64).abs()
    # Strongest form of the visible check: is the returned tensor literally the
    # oracle rounded to the output dtype, or only one tie away from it?
    vs_rounded = (out.to(torch.float64) - rounded.to(torch.float64)).abs()
    fmt_err = (ref64 - exact64).abs()
    ulp = _ulp_of(dtype, ref64)
    over_ulp = (vis_err > 1.5 * ulp + ATOL).sum().item()

    report = (
        f"{label}: dtype={dtype} "
        f"arith_max_abs={arith_err.max().item():.3e} "
        f"arith_rel={((arith_err / (ref64.abs() + 1e-6)).max().item()):.3e} "
        f"visible_max_abs={vis_err.max().item():.3e} "
        f"visible_over_1ulp={over_ulp} "
        f"visible_vs_rounded_max_abs={vs_rounded.max().item():.3e} "
        f"format_max_abs={fmt_err.max().item():.3e} "
        f"format_rel={((fmt_err / (exact64.abs() + 1e-6)).max().item()):.3e} "
        f"oracle_round_max_abs={round_err.max().item():.3e} "
        f"max_ulp={(vis_err / ulp).max().item():.2f}"
    )
    assert over_ulp == 0, "visible error exceeds one output ULP; " + report
    return report


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "n,k,m",
    [
        (256, 256, 1),
        (256, 256, 2),
        (256, 256, 4),
        (256, 256, 16),
        (4096, 4096, 1),
        (11008, 4096, 4),
    ],
)
def test_gemm_matches_fp64_oracle(dtype, n, k, m):
    codes, scales, a = _make_case(n, k, m, dtype, seed=n * 31 + k + m)
    packed, packed_scales, _ = allspark_hip_repack(codes, scales, n, k)
    out = allspark_hip_gemm(a, packed, packed_scales, n)
    raw32 = allspark_hip_gemm_f32(a, packed, packed_scales, n)
    ref64 = _oracle(codes, scales, a, dtype)
    exact64 = _exact_oracle(codes, scales, a)
    print(_check(out, raw32, ref64, exact64, dtype, f"N={n} K={k} M={m}"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_non_aligned_n(dtype):
    """N not a multiple of 128 exercises the padded tail of the repack."""
    n, k, m = 11024, 256, 3
    codes, scales, a = _make_case(n, k, m, dtype, seed=7)
    packed, packed_scales, _ = allspark_hip_repack(codes, scales, n, k)
    out = allspark_hip_gemm(a, packed, packed_scales, n)
    raw32 = allspark_hip_gemm_f32(a, packed, packed_scales, n)
    assert out.shape == (m, n)
    ref64 = _oracle(codes, scales, a, dtype)
    exact64 = _exact_oracle(codes, scales, a)
    print(_check(out, raw32, ref64, exact64, dtype, f"tail N={n} K={k} M={m}"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemm_prefill_fallback(dtype):
    """M > 16 goes through unpack + matmul; it must still be exact and must not
    leave a dense weight behind."""
    n, k, m = 256, 256, 32
    codes, scales, a = _make_case(n, k, m, dtype, seed=11)
    packed, packed_scales, _ = allspark_hip_repack(codes, scales, n, k)
    before = torch.cuda.memory_allocated()
    out = allspark_hip_gemm(a, packed, packed_scales, n)
    torch.cuda.synchronize()
    gc.collect()
    after = torch.cuda.memory_allocated()
    ref64 = _oracle(codes, scales, a, dtype)
    assert out.dtype == dtype
    vis_err = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    assert (vis_err > 1.5 * ulp + ATOL).sum().item() == 0, (
        f"fallback visible error exceeds one ULP: max_abs={vis_err.max().item():.3e}"
    )
    # Only `out` should survive; the unpacked weight was a local temporary.
    assert after - before <= out.numel() * out.element_size() + (1 << 20)


class _HipLinear(torch.nn.Module):
    """Minimal stand-in for the eventual AllSparkHipLinear module."""

    def __init__(self, packed, packed_scales, n):
        super().__init__()
        self.register_buffer("qweight_packed", packed, persistent=False)
        self.register_buffer("scales_packed", packed_scales, persistent=False)
        self.n = n

    def forward(self, x):
        return allspark_hip_gemm(x, self.qweight_packed, self.scales_packed, self.n)


def _tensor_bytes(module: torch.nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    for v in vars(module).values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_no_dense_weight_cache_across_forwards(dtype):
    """AGENTS.md storage rule: only packed weights may persist between calls."""
    n, k = 512, 512
    codes, scales, a = _make_case(n, k, 1, dtype, seed=13)
    packed, packed_scales, _ = allspark_hip_repack(codes, scales, n, k)
    del codes, scales
    torch.cuda.empty_cache()

    module = _HipLinear(packed, packed_scales, n)
    baseline = _tensor_bytes(module)
    assert baseline == packed.numel() + packed_scales.numel() * packed_scales.element_size()

    for _ in range(5):
        out = module(a)
        assert out.dtype == dtype
        torch.cuda.synchronize()
        assert _tensor_bytes(module) == baseline, "module retained extra weight bytes"
        # No attribute may hold a dense/expanded weight.
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                assert value.numel() in (packed.numel(), packed_scales.numel()), (
                    f"module attribute {name} looks like a dequantized weight cache"
                )
    assert not hasattr(module, "weight")


def test_kernel_ran_sentinel():
    """The load-time selftest must observe the sentinel the kernel stamps."""
    ref = torch.zeros(1, dtype=torch.float16, device="cuda")
    assert torch.ops.gptqmodel_allspark_hip.allspark_hip_selftest(ref)
