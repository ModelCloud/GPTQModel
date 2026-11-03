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
import json

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


def _assert_visible_midpoint_neighbors(out, raw32, ref64, label, *, matmul_reference=None):
    """Keep the 2e-3 gate, adjudicating only proven adjacent BF16 near-ties.

    The extra cap is 64 FP32 ULPs at the reference, i.e. at most 1/1024
    of a BF16 interval at the same binade. This is an explicit, tight
    FP32-roundoff budget for the tested reductions, NOT a general BF16
    tolerance. Each exception must also reproduce the raw accumulator's
    rounding and straddle the exact FP64 midpoint. Larger errors, nonadjacent
    outputs, and differences away from the midpoint remain failures.

    Fallback has no exported accumulator: raw32 is an independent Torch
    FP32 matmul, and matmul_reference must reproduce the same-dtype fallback
    exactly using independently reconstructed weights.
    """
    assert raw32.dtype == torch.float32 and ref64.dtype == torch.float64
    assert torch.isfinite(out).all() and torch.isfinite(raw32).all() and torch.isfinite(ref64).all()
    torch.testing.assert_close(raw32.double(), ref64, rtol=RTOL, atol=ATOL)
    if matmul_reference is not None:
        torch.testing.assert_close(out, matmul_reference, rtol=0, atol=0)
    rounded = ref64.to(out.dtype)
    actual64, rounded64 = out.double(), rounded.double()
    visible_error = (actual64 - ref64).abs()
    assert (visible_error <= _ulp_of(out.dtype, ref64) + ATOL).all(), label
    bad = (actual64 - rounded64).abs() > ATOL + RTOL * rounded64.abs()
    # No relaxed tolerance applies to the ordinary values.
    torch.testing.assert_close(actual64[~bad], rounded64[~bad], rtol=RTOL, atol=ATOL)
    if not bad.any():
        return 0
    assert out.dtype == torch.bfloat16, "midpoint exception is BF16-only"
    for coords in bad.nonzero().tolist():
        idx = tuple(coords)
        actual, expected, ref, raw = actual64[idx], rounded64[idx], ref64[idx], raw32[idx].double()
        midpoint = (actual + expected) / 2  # Exact for these dyadic BF16 values.
        r32 = ref.float()
        fp32_ulp = (torch.nextafter(r32, torch.full_like(r32, float("inf"))).double() - r32.double()).abs()
        bound = 64 * fp32_ulp
        arithmetic_error = (raw - ref).abs()
        adjacent = torch.nextafter(rounded[idx], out[idx]) == out[idx]
        assert adjacent, f"{label} {idx}: nonadjacent BF16 outputs"
        assert arithmetic_error <= bound, f"{label} {idx}: exceeds 64-FP32-ULP bound"
        assert (ref - midpoint).abs() <= bound, f"{label} {idx}: oracle away from midpoint"
        assert (raw - midpoint) * (ref - midpoint) <= 0, f"{label} {idx}: no midpoint crossing"
        assert (ref - midpoint).abs() <= arithmetic_error, f"{label} {idx}: insufficient crossing error"
        assert raw32[idx].to(out.dtype) == out[idx], f"{label} {idx}: raw rounding disagrees"
        record = {
            "label": label,
            "index": coords,
            "actual": actual.item(),
            "rounded_oracle": expected.item(),
            "oracle_fp64": ref.item(),
            "accumulator_fp32": raw.item(),
            "midpoint": midpoint.item(),
            "oracle_minus_midpoint": (ref - midpoint).item(),
            "accumulator_minus_midpoint": (raw - midpoint).item(),
            "arithmetic_abs": arithmetic_error.item(),
            "fp32_error_bound": bound.item(),
            "reference": "Torch same-dtype and FP32 matmul" if matmul_reference is not None else "native FP32",
        }
        print("ADJUDICATED_BF16_MIDPOINT " + json.dumps(record, sort_keys=True))
    return bad.sum().item()


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
    over_ulp = (vis_err > ulp + ATOL).sum().item()

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
    _assert_visible_midpoint_neighbors(out, raw32, ref64, label)
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
    independent_weights = _dequant_to_dtype(codes, scales, dtype)
    same_dtype_matmul = a @ independent_weights
    raw32 = a.float() @ independent_weights.float()
    _assert_visible_midpoint_neighbors(out, raw32, ref64, f"prefill N={n} K={k} M={m}",
                                      matmul_reference=same_dtype_matmul)
    vis_err = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    assert (vis_err > ulp + ATOL).sum().item() == 0, (
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,k,m", [(16, 16, 0), (112, 48, 1), (128, 240, 15),
                                    (144, 256, 16), (144, 272, 17),
                                    (16, 496, 1), (16, 512, 1), (16, 528, 1),
                                    (144, 1008, 16), (144, 1024, 16),
                                    (144, 1040, 16), (144, 2048, 16)])
def test_tile_and_splitk_boundaries(dtype, n, k, m):
    codes, scales, a = _make_case(n, k, m, dtype, seed=n + k + m)
    packed, ps, _ = allspark_hip_repack(codes, scales, n, k)
    out = allspark_hip_gemm(a, packed, ps, n)
    assert out.shape == (m, n) and out.dtype == dtype
    if not m:
        assert allspark_hip_gemm_f32(a, packed, ps, n).shape == (0, n)
        return
    ref = _oracle(codes, scales, a, dtype)
    torch.testing.assert_close(out, ref.to(dtype), rtol=RTOL, atol=ATOL)
    if m <= 16:
        raw = allspark_hip_gemm_f32(a, packed, ps, n)
        print(_check(out, raw, ref, _exact_oracle(codes, scales, a), dtype,
                     f"boundary N={n} K={k} M={m}"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [16, 112, 128, 144])
def test_repack_exact_bytes_and_unpack(dtype, n):
    k = 48
    codes, scales, _ = _make_case(n, k, 1, dtype, seed=53)
    codes.flatten()[:4] = torch.tensor([0, 127, 128, 255], device="cuda", dtype=torch.uint8)
    packed, ps, _ = allspark_hip_repack(codes, scales, n, k)
    npad = (n + 127) // 128 * 128
    # Independent layout oracle: (Ktile, Khalf, Kinner, Npair, Nhalf, Ninner)
    # becomes (Npair, Ktile, lane=Khalf*16+Ninner, Nhalf, Kinner).
    padded = torch.nn.functional.pad(codes, (0, npad - n))
    expected = padded.reshape(k // 16, 2, 8, npad // 32, 2, 16).permute(3, 0, 1, 5, 4, 2)
    assert torch.equal(packed.flatten(), expected.contiguous().flatten())
    assert torch.equal(ps[:n], scales)
    assert torch.equal(ps[n:], torch.ones_like(ps[n:]))
    dense = torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(packed, ps, n, k, None)
    assert torch.equal(dense, _dequant_to_dtype(codes, scales, dtype))


def _direct_gemm(a, packed, scales, n, zeros=None, group_size=-1, has_zp=None,
                 reordered=True):
    return torch.ops.gptqmodel_allspark_hip.allspark_w8a16_gemm(
        a, packed, scales, zeros, n, group_size, 1, 0, 1024,
        zeros is not None if has_zp is None else has_zp, reordered)


@pytest.mark.parametrize("path", ["native", "fp32", "unpack"])
@pytest.mark.parametrize("bad", ["weight_dtype", "weight_short", "weight_noncontiguous",
                                 "scale_dtype", "scale_short", "scale_cpu", "zero_dtype",
                                 "zero_short", "zero_cpu", "zero_noncontiguous",
                                 "padding", "k_zero", "k_unaligned", "n_zero", "n_unaligned"])
def test_direct_ops_reject_malformed_buffers(path, bad):
    n, k = 16, 32
    codes, scales, a = _make_case(n, k, 2, torch.float16, seed=62)
    p, s, _ = allspark_hip_repack(codes, scales, n, k)
    z = torch.full_like(s, 128)
    if bad == "weight_dtype": p = p.float()
    if bad == "weight_short": p = p.flatten()[:-1].reshape(1, -1)
    if bad == "weight_noncontiguous": p = p.t()
    if bad == "scale_dtype": s = s.float()
    if bad == "scale_short": s = s[:-1]
    if bad == "scale_cpu": s = s.cpu()
    if bad == "zero_dtype": z = z.float()
    if bad == "zero_short": z = z[:-1]
    if bad == "zero_cpu": z = z.cpu()
    if bad == "zero_noncontiguous": z = torch.ones(z.numel() * 2, device="cuda", dtype=z.dtype)[::2]
    if bad == "padding": p, s, z = p[:32], s[:32], z[:32]
    if bad == "k_zero": k, a = 0, a[:, :0].contiguous()
    if bad == "k_unaligned": k, a = 31, a[:, :31].contiguous()
    if bad == "n_zero": n = 0
    if bad == "n_unaligned": n = 15
    with pytest.raises(RuntimeError):
        if path == "native": _direct_gemm(a, p, s, n, z)
        elif path == "fp32":
            torch.ops.gptqmodel_allspark_hip.allspark_hip_gemm_f32(a, p, s, n, 1, z)
        else: torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(p, s, n, k, z)


@pytest.mark.parametrize("bad", ["shape", "output_short", "output_dtype", "scale_output_short",
                                 "zero_output_short", "zero_dtype", "missing_zero", "padding"])
def test_direct_repack_rejects_malformed_buffers(bad):
    n, k, npad = 16, 32, 128
    codes, scales, _ = _make_case(n, k, 1, torch.float16, seed=63)
    p = torch.empty((npad, k), device="cuda", dtype=torch.uint8)
    s = torch.empty(npad, device="cuda", dtype=scales.dtype)
    z, zo = torch.full_like(scales, 128), torch.empty_like(s)
    if bad == "shape": codes = codes.reshape(n, k)
    if bad == "output_short": p = p[:-1]
    if bad == "output_dtype": p = p.float()
    if bad == "scale_output_short": s = s[:-1]
    if bad == "zero_output_short": zo = zo[:-1]
    if bad == "zero_dtype": z = z.float()
    if bad == "missing_zero": zo = None
    if bad == "padding": npad, p, s, zo = 32, p[:32], s[:32], zo[:32]
    with pytest.raises(RuntimeError):
        torch.ops.gptqmodel_allspark_hip.rearrange_kn_weight_as_n32k16_order(
            codes, scales, z, True, p, s, zo, k, n, npad)


@pytest.mark.parametrize("m", [1, 17])
@pytest.mark.parametrize("bad", ["group", "shape", "dtype", "zero_k", "noncontiguous", "scale_dtype"])
def test_helper_rejects_invalid_inputs_on_native_and_fallback(m, bad):
    codes, scales, a = _make_case(128, 32, m, torch.float16, seed=64)
    p, s, _ = allspark_hip_repack(codes, scales, 128, 32)
    group = 16 if bad == "group" else -1
    if bad == "shape": a = a.unsqueeze(0)
    if bad == "dtype": a = a.float()
    if bad == "zero_k": a = a[:, :0]
    if bad == "noncontiguous": a = torch.randn(m, 64, device="cuda", dtype=a.dtype)[:, ::2]
    if bad == "scale_dtype": s = s.bfloat16()
    with pytest.raises(ValueError):
        allspark_hip_gemm(a, p, s, 128, group_size=group)


def test_native_rejects_inconsistent_flags_and_unaligned_pointers():
    codes, scales, a = _make_case(128, 32, 1, torch.float16, seed=65)
    p, s, _ = allspark_hip_repack(codes, scales, 128, 32)
    with pytest.raises(RuntimeError, match="has_zp"):
        _direct_gemm(a, p, s, 128, has_zp=True)
    with pytest.raises(RuntimeError, match="reordered"):
        _direct_gemm(a, p, s, 128, reordered=False)
    for which in ("a", "p"):
        source = a if which == "a" else p
        shifted = torch.empty(source.numel() + 1, device="cuda", dtype=source.dtype)[1:].view(source.shape)
        shifted.copy_(source)
        with pytest.raises(RuntimeError, match="aligned"):
            _direct_gemm(shifted if which == "a" else a,
                         shifted if which == "p" else p, s, 128)


@pytest.mark.parametrize("arch,expected", [("gfx1200", True), ("gfx1201:sramecc-", True),
                                          ("gfx12010", False), ("gfx1100", False),
                                          ("gfx1250", False)])
def test_arch_allowlist(monkeypatch, arch, expected):
    import gptqmodel.utils.allspark_hip as hip
    monkeypatch.setattr(hip, "_device_arch", lambda device: arch)
    assert hip.allspark_hip_supported() is expected
    assert not hip.allspark_hip_supported(torch.device("cpu"))
    if not expected:
        codes, scales, a = _make_case(128, 32, 17, torch.float16, seed=66)
        p, s, _ = allspark_hip_repack(codes, scales, 128, 32)
        with pytest.raises(ValueError, match="gfx1200/gfx1201"):
            allspark_hip_gemm(a, p, s, 128)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 17])
def test_repeated_calls_release_dense_and_splitk_scratch(dtype, m):
    n, k = 1024, 2048
    codes, scales, a = _make_case(n, k, m, dtype, seed=68)
    p, s, _ = allspark_hip_repack(codes, scales, n, k)
    del codes, scales
    # Warm up BLAS/dispatch so allocator accounting measures only live tensors.
    out = allspark_hip_gemm(a, p, s, n)
    torch.cuda.synchronize()
    del out
    gc.collect()
    baseline = torch.cuda.memory_allocated()
    for _ in range(3):
        out = allspark_hip_gemm(a, p, s, n)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - baseline == out.numel() * out.element_size()
        del out
        assert torch.cuda.memory_allocated() == baseline
    print(f"storage dtype={dtype} M={m}: packed={p.numel() + s.numel()*s.element_size()} "
          f"retained_dense=0 transient_dense={k*n*2 if m > 16 else 0}")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_device_guards_and_operand_device_mismatch(dtype):
    codes, scales, a = _make_case(144, 272, 4, dtype, seed=78)
    c1, s1, a1 = (t.to("cuda:1") for t in (codes, scales, a))
    torch.cuda.set_device(0)
    p, s, _ = allspark_hip_repack(c1, s1, 144, 272)
    out = allspark_hip_gemm(a1, p, s, 144)
    raw = allspark_hip_gemm_f32(a1, p, s, 144)
    dense = torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(p, s, 144, 272, None)
    assert torch.cuda.current_device() == 0
    assert out.device == raw.device == dense.device == torch.device("cuda:1")
    ref = _oracle(c1, s1, a1, dtype)
    torch.testing.assert_close(out, ref.to(dtype), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(raw.double(), ref, rtol=RTOL, atol=ATOL)
    for path in ("native", "fp32", "unpack"):
        for operand in ("weight", "scale", "zero"):
            mixed_p = p.to("cuda:0") if operand == "weight" else p
            mixed_s = s.to("cuda:0") if operand == "scale" else s
            mixed_z = torch.full_like(s, 128).to("cuda:0" if operand == "zero" else "cuda:1")
            with pytest.raises(RuntimeError, match="same CUDA device"):
                if path == "native": _direct_gemm(a1, mixed_p, mixed_s, 144, mixed_z)
                elif path == "fp32":
                    torch.ops.gptqmodel_allspark_hip.allspark_hip_gemm_f32(
                        a1, mixed_p, mixed_s, 144, 1, mixed_z)
                else:
                    torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(
                        mixed_p, mixed_s, 144, 272, mixed_z)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nondefault_stream_repack_gemm_and_fallback(dtype):
    codes, scales, a = _make_case(144, 528, 17, dtype, seed=79)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        p, s, _ = allspark_hip_repack(codes, scales, 144, 528)
        out = allspark_hip_gemm(a[:16], p, s, 144)
        raw = allspark_hip_gemm_f32(a[:16], p, s, 144)
        prefill = allspark_hip_gemm(a, p, s, 144)
    stream.synchronize()
    ref = _oracle(codes, scales, a, dtype)
    torch.testing.assert_close(out, ref[:16].to(dtype), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(raw.double(), ref[:16], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(prefill, ref.to(dtype), rtol=RTOL, atol=ATOL)


def test_prewarm_rejects_unsupported_arch_without_build(monkeypatch):
    import gptqmodel.utils.allspark_hip as hip
    monkeypatch.setattr(hip, "allspark_hip_supported", lambda: False)
    monkeypatch.setattr(hip._EXTENSION, "load", lambda: pytest.fail("unsupported arch reached build"))
    assert hip.prewarm_allspark_hip_extension() is False


@pytest.mark.parametrize("invalid", [None, "nonadjacent", "large_error", "away_from_midpoint",
                                     "wrong_raw_rounding", "fallback_mismatch"])
def test_midpoint_adjudication_rejects_non_ties(invalid):
    lower = torch.tensor([1.0], dtype=torch.bfloat16)
    upper = torch.nextafter(lower, torch.full_like(lower, float("inf")))
    midpoint = (lower.double() + upper.double()) / 2
    ref = midpoint - 2.0 ** -22
    raw = (midpoint + 2.0 ** -22).float()
    out = upper.clone()
    kwargs = {}
    if invalid == "nonadjacent":
        out = torch.nextafter(upper, torch.full_like(upper, float("inf")))
    elif invalid == "large_error":
        raw = (midpoint + 2.0 ** -12).float()
    elif invalid == "away_from_midpoint":
        ref = midpoint - 2.0 ** -15
    elif invalid == "wrong_raw_rounding":
        raw = ref.float()
    elif invalid == "fallback_mismatch":
        kwargs["matmul_reference"] = lower
    if invalid is None:
        assert _assert_visible_midpoint_neighbors(out, raw, ref, "synthetic midpoint", **kwargs) == 1
    else:
        with pytest.raises(AssertionError):
            _assert_visible_midpoint_neighbors(out, raw, ref, f"negative control {invalid}", **kwargs)
