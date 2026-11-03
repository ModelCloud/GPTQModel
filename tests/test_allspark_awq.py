# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ 8-bit end-to-end validation for the AllSpark HIP/RDNA4 kernel.

Scope is RDNA4 only (gfx1200/gfx1201). The tests start from real AWQ tensors
produced by the repo's own packer (``pack_awq``), so they cover the on-disk
layout, not just the kernel:

* AWQ packs along N (``qweight`` is ``(K, ceil(N*bits/32))`` while GPTQ-8bit
  packs along K), so a wrong layout shows up immediately as garbage.
* ``qzeros`` is per K-group, so grouping is exercised explicitly.
* Oracle policy follows the repo convention: fp64 arithmetic from the original
  codes/zeros/scales over weights rounded to the activation dtype. Arithmetic
  error (fp32 accumulators) and visible output rounding are reported apart.
"""

from __future__ import annotations

import gc

import pytest
import torch

from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.allspark_hip import (
    allspark_hip_gemm,
    allspark_hip_gemm_f32,
    allspark_hip_repack_awq,
    allspark_hip_supported,
    prewarm_allspark_hip_extension,
)
from tests.test_allspark_hip import _assert_visible_midpoint_neighbors


pytestmark = pytest.mark.skipif(
    not allspark_hip_supported(), reason="gfx1200/gfx1201 ROCm GPU required"
)

ATOL = 2e-3
RTOL = 2e-3


@pytest.fixture(scope="module", autouse=True)
def _prewarm():
    assert prewarm_allspark_hip_extension(), "AllSpark HIP extension failed to build"


def _awq_case(n: int, k: int, m: int, group_size: int, dtype: torch.dtype, seed: int):
    """Random AWQ-8bit tensors. Never all-zero: a zero tensor would let the
    hardware skip work and hide layout/dequant bugs."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = k if group_size in (-1, k) else group_size
    assert k % g == 0
    groups = k // g
    codes = torch.randint(0, 256, (k, n), dtype=torch.int32, device="cuda", generator=gen)
    zeros = torch.randint(0, 256, (groups, n), dtype=torch.int32, device="cuda", generator=gen)
    # A zero point far from 128 proves the path is not silently assuming GPTQ's
    # neutral zero; the explicit spread makes a mixed-up row obvious.
    zeros[::2] = torch.randint(0, 32, (zeros[::2].shape[0], n), dtype=torch.int32,
                               device="cuda", generator=gen)
    scales = (torch.rand(groups, n, dtype=torch.float32, device="cuda", generator=gen) * 0.01
              + 0.005).to(dtype)
    qweight, qzeros = pack_awq(codes, zeros, 8)
    a = (torch.randn(m, k, dtype=torch.float32, device="cuda", generator=gen) * 0.5).to(dtype)
    return qweight, qzeros, scales, codes, zeros, a, g


def _round_weights(codes, zeros, scales, g, dtype):
    """The weight values the tensor cores consume (see test_allspark_hip.py)."""
    zk = zeros.repeat_interleave(g, 0)
    sk = scales.repeat_interleave(g, 0)
    if dtype == torch.float16:
        return ((codes.to(torch.float16) - zk.to(torch.float16)) * sk).to(torch.float64)
    return ((codes.to(torch.float32) - zk.to(torch.float32)) * sk.to(torch.float32)).to(
        torch.bfloat16
    ).to(torch.float64)


def _ulp_of(dtype: torch.dtype, ref64: torch.Tensor) -> torch.Tensor:
    r = ref64.to(dtype)
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    return (up.float().to(torch.float64) - r.float().to(torch.float64)).abs()


def _check(out, raw32, ref64, dtype, label):
    assert out.dtype == dtype, f"{label}: dtype {out.dtype} != {dtype}"
    assert raw32.dtype == torch.float32
    _assert_visible_midpoint_neighbors(out, raw32, ref64, label)
    torch.testing.assert_close(raw32, ref64.float(), rtol=RTOL, atol=ATOL)
    arith = (raw32.to(torch.float64) - ref64).abs().max().item()
    vis = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    over = (vis > ulp + ATOL).sum().item()
    report = (
        f"{label}: dtype={dtype} arith_max_abs={arith:.3e} "
        f"visible_max_abs={vis.max().item():.3e} visible_over_1ulp={over} "
        f"max_ulp={(vis / ulp).max().item():.2f}"
    )
    assert over == 0, "visible error exceeds one output ULP; " + report
    print(report)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 4, 16])
def test_awq_8bit_matches_fp64_oracle(dtype, m):
    # group_size=-1 only: AWQ's qzeros are per output channel, which is the one
    # layout the kernel can load outside the k-loop.
    group_size = -1
    n, k = 256, 256
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, group_size, dtype,
                                                            seed=m)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, group_size
    )
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, group_size)
    raw32 = allspark_hip_gemm_f32(a, packed, packed_scales, n, packed_zeros, group_size)
    ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
    _check(out, raw32, ref64, dtype, f"awq N={n} K={k} M={m} gs={group_size}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_zero_point_is_not_assumed_128(dtype):
    """A constant zero of 200 must fail loudly if the kernel ignored qzeros."""
    n, k, m = 128, 256, 4
    codes = torch.full((k, n), 210, dtype=torch.int32, device="cuda")
    zeros = torch.full((1, n), 200, dtype=torch.int32, device="cuda")
    scales = torch.full((1, n), 0.01, dtype=dtype, device="cuda")
    qweight, qzeros = pack_awq(codes, zeros, 8)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    a = torch.ones(m, k, dtype=dtype, device="cuda")
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, -1)
    # Every weight is (210 - 200) * 0.01 = 0.1; K=256 rows of activation 1.0.
    expected = 0.1 * k
    torch.testing.assert_close(out.float(), torch.full_like(out.float(), expected),
                               rtol=RTOL, atol=ATOL)
    # If the kernel had assumed the GPTQ neutral zero (128), the result would be
    # 0.82 * K = 210.
    assert (out.float() - expected).abs().max().item() < 0.1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_non_aligned_n(dtype):
    n, k, m = 11024, 128, 3
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, -1, dtype, seed=99)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, -1)
    raw32 = allspark_hip_gemm_f32(a, packed, packed_scales, n, packed_zeros, -1)
    assert out.shape == (m, n)
    ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
    _check(out, raw32, ref64, dtype, f"awq tail N={n} K={k} M={m}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_prefill_fallback(dtype):
    """M > 16 goes through the unpack + matmul fallback and must not cache."""
    n, k, m = 256, 256, 32
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, -1, dtype, seed=17)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    before = torch.cuda.memory_allocated()
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, -1)
    torch.cuda.synchronize()
    gc.collect()
    after = torch.cuda.memory_allocated()
    ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
    independent_weights = _round_weights(codes, zeros, scales, g, dtype).to(dtype)
    raw32 = a.float() @ independent_weights.float()
    _assert_visible_midpoint_neighbors(out, raw32, ref64, "AWQ prefill",
                                      matmul_reference=a @ independent_weights)
    vis = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    assert (vis > ulp + ATOL).sum().item() == 0, (
        f"fallback visible error exceeds one ULP: max={vis.max().item():.3e}"
    )
    assert after - before <= out.numel() * out.element_size() + (1 << 20)


class _AwqHipLinear(torch.nn.Module):
    def __init__(self, packed, packed_scales, packed_zeros, n, group_size):
        super().__init__()
        self.register_buffer("qweight_packed", packed, persistent=False)
        self.register_buffer("scales_packed", packed_scales, persistent=False)
        self.register_buffer("zeros_packed", packed_zeros, persistent=False)
        self.n = n
        self.group_size = group_size

    def forward(self, x):
        return allspark_hip_gemm(
            x, self.qweight_packed, self.scales_packed, self.n,
            self.zeros_packed, self.group_size,
        )


def _tensor_bytes(module: torch.nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    for v in vars(module).values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_no_dense_weight_cache(dtype):
    n, k = 512, 512
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, 1, -1, dtype, seed=23)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    del codes, zeros, qweight, qzeros
    torch.cuda.empty_cache()

    module = _AwqHipLinear(packed, packed_scales, packed_zeros, n, -1)
    baseline = _tensor_bytes(module)
    packed_bytes = (packed.numel() + packed_scales.numel() * packed_scales.element_size()
                    + packed_zeros.numel() * packed_zeros.element_size())
    assert baseline == packed_bytes
    for _ in range(5):
        out = module(a)
        assert out.dtype == dtype
        torch.cuda.synchronize()
        assert _tensor_bytes(module) == baseline, "module retained extra weight bytes"
    assert not hasattr(module, "weight")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_second_gpu_matches_first(dtype):
    """A custom op gets no dispatcher device guard, so running on cuda:1 while
    cuda:0 is current must still use cuda:1's pointers. Without the guard the
    kernel reads another device's memory and the output is NaN."""
    n, k, m, gsz = 256, 256, 8, -1
    qweight, qzeros, scales, codes, zeros, a, _ = _awq_case(
        n, k, m, gsz, dtype, seed=41
    )
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, gsz
    )
    a1 = a.to("cuda:0")
    out0 = allspark_hip_gemm(a1, packed.to("cuda:0"), packed_scales.to("cuda:0"), n,
                             packed_zeros.to("cuda:0"), gsz)

    # Move the operands first, then force cuda:0 to stay current: the op must
    # switch to the tensors' device itself.
    p1, s1, z1, a2 = (packed.to("cuda:1"), packed_scales.to("cuda:1"),
                      packed_zeros.to("cuda:1"), a.to("cuda:1"))
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    out1 = allspark_hip_gemm(a2, p1, s1, n, z1, gsz)
    assert out1.device.index == 1
    assert torch.isfinite(out1).all(), "second-GPU output is not finite"
    torch.testing.assert_close(out1.to("cuda:0"), out0, rtol=RTOL, atol=ATOL)


def test_awq_module_runs_on_rdna4():
    """Module-level path: real packer -> post_init -> forward, fp16 and bf16."""
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear

    for dtype in (torch.float16, torch.bfloat16):
        k, n, m, group = 512, 512, 8, -1
        qweight, qzeros, scales, codes, zeros, a, g = _awq_case(
            n, k, m, group, dtype, seed=31
        )
        mod = AllSparkAwqLinear(
            bits=8, group_size=group, sym=True, desc_act=False,
            in_features=k, out_features=n, bias=False,
            dtype=torch.bfloat16 if dtype == torch.float16 else torch.float16,
        ).cuda().to(dtype)
        mod.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        mod.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        mod.scales = torch.nn.Parameter(scales, requires_grad=False)
        mod.post_init()
        assert mod.qweight is None and mod.qzeros is None
        assert mod.scales_packed.dtype == dtype
        a = torch.stack((a, a), dim=-1).reshape(m, k * 2)[:, ::2]
        assert not a.is_contiguous()
        out = mod(a)
        assert out.dtype == dtype
        ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
        torch.testing.assert_close(out, ref64.to(dtype), rtol=RTOL, atol=ATOL)
        vis = (out.to(torch.float64) - ref64).abs()
        ulp = _ulp_of(dtype, ref64)
        assert (vis > ulp + ATOL).sum().item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_pack_roundtrip_from_dense_weight(dtype):
    """Full quantisation path: dense weight -> pack() -> post_init() -> forward.

    This is the only test that exercises the module's own AWQ packer against a
    known weight, so a packer that silently used the GPTQ (input-axis) layout
    would fail here rather than only in a checkpoint load.
    """
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear

    # group_size=-1 is one group over all of K.
    group_size, k, n, g = -1, 512, 512, 512
    torch.manual_seed(5)
    weight = (torch.randn(n, k, dtype=torch.float32, device="cuda") * 0.02)
    groups = k // g
    # Affine per-group quantisation over the real [min, max] range: this keeps
    # both the scale and the zero point inside 8-bit range (an absmax/255 scale
    # with ``-min/scale`` saturates the zero point at 255 and reconstructs garbage).
    wg = weight.reshape(n, groups, g)
    wmin, wmax = wg.amin(dim=-1), wg.amax(dim=-1)
    scales = ((wmax - wmin) / 255).clamp_min(1e-8)                 # (n, groups)
    zeros = (-wmin / scales).round().clamp(0, 255).to(torch.int32)  # (n, groups)

    mod = AllSparkAwqLinear(
        bits=8, group_size=group_size, sym=False, desc_act=False,
        in_features=k, out_features=n, bias=False,
        dtype=torch.bfloat16 if dtype == torch.float16 else torch.float16,
        register_buffers=True,
    ).cuda().to(dtype)
    with torch.no_grad():
        mod.pack(_Dense(weight), scales, zeros)
    assert mod.scales.dtype == dtype
    # post_init drops the on-disk tensors, so keep copies for the oracle.
    qw, qz, sc = mod.qweight.clone(), mod.qzeros.clone(), mod.scales.clone()
    mod.post_init()
    assert mod.qweight is None and mod.qzeros is None

    x = (torch.randn(4, k, dtype=torch.float32, device="cuda") * 0.5).to(dtype)
    out = mod(x)
    assert out.dtype == dtype

    # Truth is the quantised weight AWQ actually stores, not the fp16 original.
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

    wq = dequantize_gemm(qw, qz, sc, 8, g)
    if dtype == torch.float16:
        wq = wq.half()
    else:
        wq = wq.to(torch.bfloat16)
    ref = x.to(torch.float64) @ wq.to(torch.float64)
    vis = (out.to(torch.float64) - ref).abs()
    ulp = _ulp_of(dtype, ref)
    assert (vis > ulp + ATOL).sum().item() == 0, (
        f"visible error exceeds one ULP: max={vis.max().item():.3e}"
    )
    # The stored codes must reproduce the dense weight to within 8-bit noise.
    # ``wq`` is (K, N) - AWQ packs along N - while ``weight`` is (N, K).
    rel = (wq.float() - weight.t()).abs().max().item() / weight.abs().max().item()
    assert rel < 0.01, f"quantised weight drifted from the dense weight: {rel:.4f}"


class _Dense(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.bias = None


def test_awq_backend_registered():
    """AWQ shares the AllSpark backend and is offered on RDNA4 and Ampere+ CUDA."""
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import get_kernel_backends

    # One shared AllSpark backend; the checkpoint method picks the AWQ module.
    assert BACKEND.ALLSPARK in get_kernel_backends(AllSparkAwqLinear)
    assert set(AllSparkAwqLinear.SUPPORTS_DEVICES) == {DEVICE.CUDA, DEVICE.ROCM}
    # Per-output-channel zeros only.
    assert AllSparkAwqLinear.SUPPORTS_GROUP_SIZE == [-1]


def test_awq_selectable_through_registry():
    """The loader path, not just the class: AWQ + GEMM + allspark resolves."""
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    cls = select_quant_linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        device="cuda",
        backend=BACKEND.ALLSPARK,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert cls.__name__ == "AllSparkAwqLinear"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 16, 17])
def test_awq_explicit_full_k_group(dtype, m):
    n, k = 144, 272
    qw, qz, scales, codes, zeros, a, g = _awq_case(n, k, m, k, dtype, seed=71)
    p, s, z = allspark_hip_repack_awq(qw, qz, scales, n, k, k)
    out = allspark_hip_gemm(a, p, s, n, z, k)
    ref = a.double() @ _round_weights(codes, zeros, scales, g, dtype)
    torch.testing.assert_close(out, ref.to(dtype), rtol=RTOL, atol=ATOL)
    if m <= 16:
        raw = allspark_hip_gemm_f32(a, p, s, n, z, k)
        _check(out, raw, ref, dtype, f"full-K AWQ M={m}")


@pytest.mark.parametrize("group_size", [0, -2, 16])
def test_awq_rejects_group_before_unpack(monkeypatch, group_size):
    from gptqmodel.quantization.awq.utils import packing_utils
    def forbidden(*args, **kwargs):
        pytest.fail("invalid grouping reached unpack_awq")
    monkeypatch.setattr(packing_utils, "unpack_awq", forbidden)
    with pytest.raises(ValueError, match="group_size=-1 or K"):
        allspark_hip_repack_awq(None, None, None, 128, 32, group_size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_zero_endpoints_and_padded_unpack(dtype):
    n, k = 144, 48
    qw, qz, scales, codes, zeros, _, _ = _awq_case(n, k, 1, -1, dtype, seed=72)
    codes.flatten()[:4] = torch.tensor([0, 127, 128, 255], device="cuda")
    zeros.flatten()[:4] = torch.tensor([255, 128, 127, 0], device="cuda")
    qw, qz = pack_awq(codes, zeros, 8)
    p, s, z = allspark_hip_repack_awq(qw, qz, scales, n, k, -1)
    assert torch.equal(z[:n], zeros.flatten().to(dtype))
    assert torch.equal(z[n:], torch.zeros_like(z[n:]))
    unpacked = torch.ops.gptqmodel_allspark_hip.allspark_hip_unpack(p, s, n, k, z)
    assert torch.equal(unpacked, _round_weights(codes, zeros, scales, k, dtype).to(dtype))


@pytest.mark.parametrize("bad", ["weight_dtype", "zero_dtype", "weight_shape", "zero_rows",
                                 "scale_rows", "scale_dtype", "zero_device"])
def test_awq_repack_rejects_invalid_format(bad):
    n, k = 128, 32
    qw, qz, scales, *_ = _awq_case(n, k, 1, -1, torch.float16, seed=80)
    if bad == "weight_dtype": qw = qw.long()
    if bad == "zero_dtype": qz = qz.long()
    if bad == "weight_shape": qw = qw[:-1]
    if bad == "zero_rows": qz = qz.repeat(2, 1)
    if bad == "scale_rows": scales = scales.repeat(2, 1)
    if bad == "scale_dtype": scales = scales.float()
    if bad == "zero_device": qz = qz.cpu()
    with pytest.raises(ValueError):
        allspark_hip_repack_awq(qw, qz, scales, n, k, -1)
