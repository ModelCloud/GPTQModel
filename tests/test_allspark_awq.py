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
    torch.testing.assert_close(raw32, ref64.float(), rtol=RTOL, atol=ATOL)
    arith = (raw32.to(torch.float64) - ref64).abs().max().item()
    vis = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    over = (vis > 1.5 * ulp + ATOL).sum().item()
    report = (
        f"{label}: dtype={dtype} arith_max_abs={arith:.3e} "
        f"visible_max_abs={vis.max().item():.3e} visible_over_1ulp={over} "
        f"max_ulp={(vis / ulp).max().item():.2f}"
    )
    assert over == 0, "visible error exceeds one output ULP; " + report
    print(report)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 16, 32, 64, 128])
@pytest.mark.parametrize("m", [1, 4, 16])
def test_awq_8bit_matches_fp64_oracle(dtype, group_size, m):
    n, k = 256, 256
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, group_size, dtype,
                                                            seed=group_size + m)
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
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, 32, dtype, seed=99)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, 32
    )
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, 32)
    raw32 = allspark_hip_gemm_f32(a, packed, packed_scales, n, packed_zeros, 32)
    assert out.shape == (m, n)
    ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
    _check(out, raw32, ref64, dtype, f"awq tail N={n} K={k} M={m}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_prefill_fallback(dtype):
    """M > 16 goes through the grouped unpack + matmul and must not cache."""
    n, k, m = 256, 256, 32
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, m, 32, dtype, seed=17)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, 32
    )
    before = torch.cuda.memory_allocated()
    out = allspark_hip_gemm(a, packed, packed_scales, n, packed_zeros, 32)
    torch.cuda.synchronize()
    gc.collect()
    after = torch.cuda.memory_allocated()
    ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
    vis = (out.to(torch.float64) - ref64).abs()
    ulp = _ulp_of(dtype, ref64)
    assert (vis > 1.5 * ulp + ATOL).sum().item() == 0, (
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
    qweight, qzeros, scales, codes, zeros, a, g = _awq_case(n, k, 1, 128, dtype, seed=23)
    packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
        qweight, qzeros, scales, n, k, 128
    )
    del codes, zeros, qweight, qzeros
    torch.cuda.empty_cache()

    module = _AwqHipLinear(packed, packed_scales, packed_zeros, n, 128)
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
    n, k, m, gsz = 256, 256, 8, 128
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
        k, n, m, group = 512, 512, 8, 128
        qweight, qzeros, scales, codes, zeros, a, g = _awq_case(
            n, k, m, group, dtype, seed=31
        )
        mod = AllSparkAwqLinear(
            bits=8, group_size=group, sym=True, desc_act=False,
            in_features=k, out_features=n, bias=False, dtype=dtype,
        ).cuda().to(dtype)
        mod.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        mod.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        mod.scales = torch.nn.Parameter(scales, requires_grad=False)
        mod.post_init()
        assert mod.qweight is None and mod.qzeros is None
        out = mod(a)
        assert out.dtype == dtype
        ref64 = a.to(torch.float64) @ _round_weights(codes, zeros, scales, g, dtype)
        vis = (out.to(torch.float64) - ref64).abs()
        ulp = _ulp_of(dtype, ref64)
        assert (vis > 1.5 * ulp + ATOL).sum().item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 128])
def test_awq_pack_roundtrip_from_dense_weight(dtype, group_size):
    """Full quantisation path: dense weight -> pack() -> post_init() -> forward.

    This is the only test that exercises the module's own AWQ packer against a
    known weight, so a packer that silently used the GPTQ (input-axis) layout
    would fail here rather than only in a checkpoint load.
    """
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear

    k, n = 512, 512
    g = k if group_size == -1 else group_size
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
        in_features=k, out_features=n, bias=False, dtype=dtype,
        register_buffers=True,
    ).cuda().to(dtype)
    with torch.no_grad():
        mod.pack(_Dense(weight), scales, zeros)
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
    assert (vis > 1.5 * ulp + ATOL).sum().item() == 0, (
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
    """AWQ is RDNA4-only and must not be selectable on CUDA."""
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import get_kernel_backends

    assert BACKEND.AWQ_ALLSPARK in get_kernel_backends(AllSparkAwqLinear)
    assert AllSparkAwqLinear.SUPPORTS_DEVICES == [DEVICE.ROCM]


def test_awq_selectable_through_registry():
    """The loader path, not just the class: AWQ + GEMM + awq_allspark resolves."""
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear

    for group_size in (-1, 128):
        cls = select_quant_linear(
            bits=8,
            group_size=group_size,
            desc_act=False,
            sym=True,
            device="cuda",
            backend=BACKEND.AWQ_ALLSPARK,
            format=FORMAT.GEMM,
            quant_method=METHOD.AWQ,
            pack_dtype=torch.int32,
        )
        assert cls.__name__ == "AllSparkAwqLinear"
