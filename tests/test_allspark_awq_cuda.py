# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ 8-bit validation for the AllSpark CUDA (Ampere+) kernel.

The HIP AWQ path lives in ``test_allspark_awq.py`` and is ROCm-gated. This file
covers the CUDA side of the same AWQ contract, which differs in exactly one
place worth testing:

* the CUDA kernel applies the zero point AFTER its bias-128 dequant, i.e. it
  computes ``((code - 128) - zero_buffer) * scale``, so the repack has to store
  ``zero_awq - 128``. Storing the raw AWQ zero instead shifts every weight by
  128 LSBs. ``test_awq_zero_point_convention_is_load_bearing`` pins that down
  with a negative control, because a plain accuracy assertion would still pass
  if both the repack and the oracle drifted the same way.

Oracle policy matches the rest of the repo: fp64 arithmetic from the original
codes/zeros/scales, weights rounded to the activation dtype.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.allspark import (
    allspark_gemm,
    allspark_repack_awq,
    allspark_supported,
    prewarm_allspark_extension,
)
from gptqmodel.utils.rocm import IS_ROCM


pytestmark = pytest.mark.skipif(
    IS_ROCM or not allspark_supported(),
    reason="AllSpark CUDA (Ampere+) GPU required",
)

DTYPES = [torch.float16, torch.bfloat16]


def _case(n, k, dtype, seed=0):
    """Real AWQ on-disk tensors from the repo's own packer."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 256, (k, n), generator=g, dtype=torch.int32).cuda()
    # Non-128 zeros on purpose: a neutral 128 zero would hide an off-by-128 bug.
    zeros = torch.randint(1, 256, (1, n), generator=g, dtype=torch.int32).cuda()
    scales = (
        (torch.rand(1, n, generator=g, dtype=torch.float32) * 0.01 + 0.001)
        .to(dtype)
        .cuda()
    )
    qweight, qzeros = pack_awq(codes, zeros, 8)
    return codes, zeros, scales, qweight, qzeros


def _oracle(a, codes, zeros, scales):
    # Independent dequantization, including the weight rounding performed
    # before tensor-core multiplication. MMA and split reduction use FP32.
    weight = ((codes.double() - zeros.double()) * scales.double()).to(a.dtype)
    return a.double() @ weight.double()


def _check(a, codes, zeros, scales, qweight, qzeros, m, dtype):
    n, k = codes.shape[1], codes.shape[0]
    packed, pscales, pzeros = allspark_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    out = allspark_gemm(a, packed, pscales, n, b_qzeros=pzeros)
    assert out.dtype == dtype
    assert out.shape == (m, n)

    ref64 = _oracle(a, codes, zeros, scales)
    _assert_rounded_close(out, ref64)


def _assert_rounded_close(out, ref64):
    assert torch.isfinite(out).all()
    if out.numel() == 0:
        return
    rounded = ref64.to(out.dtype)
    step = (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).double()
            - rounded.double()).abs()
    # Permit one output ULP at a rounding boundary. Residual arithmetic error
    # beyond that budget must meet the repository's 2e-3 absolute/relative gate.
    excess = ((out.double() - ref64).abs() - step).clamp_min(0)
    assert torch.all(excess <= 2e-3 + 2e-3 * ref64.abs()), float(excess.max())
    visible = (out.double() - rounded.double()).abs()
    assert torch.all(visible <= step + 2e-3), float(visible.max())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,k,m", [
    (256, 256, 0), (256, 256, 1), (256, 256, 8), (4864, 896, 4),
    (11008, 4096, 1), (272, 80, 64), (272, 80, 1024), (272, 80, 1025),
])
def test_awq_matches_fp64_oracle(dtype, n, k, m):
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    codes, zeros, scales, qweight, qzeros = _case(n, k, dtype, seed=n + k + m)
    gen = torch.Generator(device="cuda:0").manual_seed(n + k + m)
    a = torch.randn(m, k, device="cuda:0", dtype=dtype, generator=gen)
    _check(a, codes, zeros, scales, qweight, qzeros, m, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_zero_point_convention_is_load_bearing(dtype):
    """The stored zero buffer must be ``zero_awq - 128``, not the raw code.

    Feeds the raw zero through the same kernel and requires a visibly worse
    result, so a future "simplification" of the repack fails here instead of
    silently degrading every AWQ checkpoint.
    """
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    n, k, m = 256, 256, 4
    codes, zeros, scales, qweight, qzeros = _case(n, k, dtype, seed=99)
    gen = torch.Generator(device="cuda:0").manual_seed(99)
    a = torch.randn(m, k, device="cuda:0", dtype=dtype, generator=gen)

    packed, pscales, pzeros = allspark_repack_awq(qweight, qzeros, scales, n, k, -1)
    good = allspark_gemm(a, packed, pscales, n, b_qzeros=pzeros).float()

    # Negative control: undo the -128 adjustment.
    raw_zeros = (pzeros.float() + 128.0).to(dtype)
    bad = allspark_gemm(a, packed, pscales, n, b_qzeros=raw_zeros).float()

    ref = _oracle(a, codes, zeros, scales)
    ref_max = ref.abs().max().item()
    good_rel = (good.double() - ref).abs().max().item() / ref_max
    bad_rel = (bad.double() - ref).abs().max().item() / ref_max
    # Measured: good 2.6e-4..2.6e-3, bad 6.7e-1..7.3e-1.
    assert good_rel < 1e-2, f"correct convention should match, got rel={good_rel:.3e}"
    assert bad_rel > 0.05 and bad_rel > 10 * good_rel, (
        "raw-zero control did not degrade: the kernel may be ignoring b_qzeros "
        f"(good={good_rel:.3e}, bad={bad_rel:.3e})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_rejects_grouped_zeros(dtype):
    """Only group_size=-1 is supported; anything else must raise, not mis-scale."""
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    n, k = 256, 256
    _, _, scales, qweight, qzeros = _case(n, k, dtype, seed=5)
    with pytest.raises(ValueError, match="group_size=-1"):
        allspark_repack_awq(qweight, qzeros, scales, n, k, 128)


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_full_k_group_alias_and_zero_endpoints(dtype):
    assert prewarm_allspark_extension()
    codes, zeros, scales, qw, qz = _case(272, 80, dtype, seed=7)
    zeros[:, ::2] = 0
    zeros[:, 1::2] = 255
    qw, qz = pack_awq(codes, zeros, 8)
    p, s, z = allspark_repack_awq(qw, qz, scales, 272, 80, 80)
    a = torch.randn(8, 80, device="cuda", dtype=dtype)
    _assert_rounded_close(allspark_gemm(a, p, s, 272, z), _oracle(a, codes, zeros, scales))


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_cuda_streams_graph_and_second_device(dtype):
    assert prewarm_allspark_extension()
    codes, zeros, scales, qw, qz = _case(512, 1024, dtype, seed=81)
    # Repack on device 1 while device 0 is current exercises the device guard.
    for ordinal in range(min(torch.cuda.device_count(), 2)):
        device = torch.device("cuda", ordinal)
        p, s, z = allspark_repack_awq(qw.to(device), qz.to(device), scales.to(device), 512, 1024, -1)
        with torch.cuda.device(device):
            a = torch.randn(4, 1024, device=device, dtype=dtype)
            b = torch.randn_like(a)
            stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
            stream1.wait_stream(torch.cuda.current_stream())
            stream2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream1):
                for _ in range(10):
                    out1 = allspark_gemm(a, p, s, 512, z)
            with torch.cuda.stream(stream2):
                for _ in range(10):
                    out2 = allspark_gemm(b, p, s, 512, z)
            torch.cuda.current_stream().wait_stream(stream1)
            torch.cuda.current_stream().wait_stream(stream2)
            _assert_rounded_close(out1, _oracle(a, codes.to(device), zeros.to(device), scales.to(device)))
            _assert_rounded_close(out2, _oracle(b, codes.to(device), zeros.to(device), scales.to(device)))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream1):
                captured = allspark_gemm(a, p, s, 512, z)
            a.neg_()
            graph.replay()
            _assert_rounded_close(captured, _oracle(a, codes.to(device), zeros.to(device), scales.to(device)))


def test_cuda_rejects_invalid_buffers():
    assert prewarm_allspark_extension()
    _, _, scales, qw, qz = _case(256, 256, torch.float16)
    p, s, z = allspark_repack_awq(qw, qz, scales, 256, 256, -1)
    a = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    for bad_scales in (s.float(), s[:-1]):
        with pytest.raises(RuntimeError, match="b_scales"):
            allspark_gemm(a, p, bad_scales, 256, z)
    with pytest.raises(RuntimeError, match="float16 or bfloat16"):
        allspark_gemm(a.float(), p, s, 256, z)
    unaligned = torch.randn(257, device="cuda", dtype=a.dtype)[1:].view(1, 256)
    with pytest.raises(RuntimeError, match="aligned"):
        allspark_gemm(unaligned, p, s, 256, z)
    if torch.cuda.device_count() > 1:
        with pytest.raises(RuntimeError, match="same CUDA device"):
            allspark_gemm(a, p.to("cuda:1"), s, 256, z)


def test_cuda_fallback_releases_dequantized_storage():
    assert prewarm_allspark_extension()
    _, _, scales, qw, qz = _case(512, 1024, torch.float16, seed=61)
    p, s, z = allspark_repack_awq(qw, qz, scales, 512, 1024, -1)
    a = torch.randn(1025, 1024, device="cuda", dtype=torch.float16)
    # Initialize cuBLAS before measuring live tensor storage, not allocator reserves.
    warm = a @ a.t()
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        result = allspark_gemm(a, p, s, 512, z)
        del result
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= before + 4096


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_module_pack_boundaries_and_dtype_moves(dtype):
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear

    k, n = 80, 272
    # Power-of-two scales make half-code ties and their neighbours exact.
    values = torch.tensor([-129., -128., -127.5, -0.5, 0., 0.5, 1.5, 126.5, 127., 128.], dtype=dtype)
    below = torch.nextafter(values, torch.full_like(values, -float("inf")))
    above = torch.nextafter(values, torch.full_like(values, float("inf")))
    weight = torch.cat([below, values, above]).repeat((k * n + 29) // 30)[:k*n].reshape(n, k).cuda()
    dense = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=dtype)
    dense.weight.data.copy_(weight)
    mod = AllSparkAwqLinear(bits=8, group_size=-1, sym=False, desc_act=False,
                           in_features=k, out_features=n, dtype=dtype, register_buffers=True).cuda()
    scales = torch.ones(n, 1, device="cuda", dtype=dtype)
    zeros = torch.full((n, 1), 128, device="cuda", dtype=torch.int32)
    mod.pack(dense, scales, zeros)
    # Reproduce the format's input-dtype add/divide before ties-to-even rounding.
    codes = (weight.t() + 128).round().clamp(0, 255).to(torch.int32)
    shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
    expected = (codes.long().reshape(k, n // 4, 4) << shifts).sum(-1).to(torch.int32)
    torch.testing.assert_close(mod.qweight, expected, rtol=0, atol=0)
    torch.testing.assert_close(mod.qzeros, torch.full_like(mod.qzeros, -2139062144), rtol=0, atol=0)
    other = torch.bfloat16 if dtype == torch.float16 else torch.float16
    mod.to(other)
    mod.post_init()
    x = torch.randn(4, k * 2, device="cuda", dtype=other)[:, ::2]
    out = mod(x)
    _assert_rounded_close(out, _oracle(x, codes, zeros.t(), scales.t().to(other)))
    assert mod.qweight is None and mod.scales is None and mod.qzeros is None
    for _ in range(2):
        mod(x)
    assert set(dict(mod.named_buffers())) == {"qweight_packed", "scales_packed", "zeros_packed"}
    assert mod.qweight_packed.dtype == torch.uint8
    # Packed scales are authoritative, so mutation and dtype/device moves do
    # not leave an expanded weight cache with stale values.
    mod.scales_packed.mul_(0.5)
    _assert_rounded_close(mod(x), _oracle(x, codes, zeros.t(), scales.t().to(other) * 0.5))
