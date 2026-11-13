# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Build and numerically check the AllSpark W8A16 kernel through the JIT path."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AllSpark needs a CUDA device"
)


def _quantize_per_column(weight: torch.Tensor):
    """Symmetric int8 per output column (group_size = -1) in GPTQ uint8b128 form."""
    scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-8)
    qweight = (weight / scale).round().add(128).clamp(0, 255).to(torch.uint8)
    return qweight.contiguous(), scale.to(weight.dtype).contiguous()


@pytest.mark.parametrize("m", [1, 16, 256, 2048])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_gemm_matches_dequant_reference(m, dtype):
    from gptqmodel.utils.allspark import (
        allspark_gemm,
        allspark_repack,
        allspark_supported,
        prewarm_allspark_extension,
    )

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")

    assert prewarm_allspark_extension(), "AllSpark extension failed to build"

    n, k = 4096, 4096
    torch.manual_seed(42 + m)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    weight = torch.randn(k, n, dtype=dtype, device="cuda")

    qweight, scales = _quantize_per_column(weight)
    qweight_reordered, scales_reordered, _ = allspark_repack(qweight, scales, n, k)
    assert qweight_reordered.shape == (n, k)

    out = allspark_gemm(a, qweight_reordered, scales_reordered, n)
    reference = a @ ((qweight.float() - 128) * scales.float()).to(dtype)

    rel = (out.float() - reference.float()).abs().mean() / reference.float().abs().mean()
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert rel < 2e-3, f"m={m} relative error {rel.item():.5f}"


def _allspark_bias_case(dtype, m, n=4096, k=4096, value=-32768.0):
    """Weight == 1.0 per column, activations 16: the product is 65536."""
    from gptqmodel.utils.allspark import allspark_repack

    codes = torch.full((k, n), 0xFF, dtype=torch.uint8, device="cuda")
    scales = torch.full((n,), 1.0 / 127.0, dtype=dtype, device="cuda")
    packed, packed_scales, _ = allspark_repack(codes, scales, n, k)
    a = torch.full((m, k), 16.0, dtype=dtype, device="cuda")
    bias = torch.full((n,), value, dtype=torch.float32, device="cuda")
    return a, packed, packed_scales, bias, codes, scales


@pytest.mark.parametrize("m", [1, 32, 64, 1025])
@pytest.mark.parametrize("bias_dtype", [None, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_bias_added_before_output_rounding(m, bias_dtype, dtype):
    """FP16 `65536 + (-32768)` is `32768`; a narrowed partial sum is `inf`.

    ``m`` spans the fused split-K reduce (m <= Mtile), the multi-block MMA
    path, and the cuBLAS dense fallback (m > 1024).
    """
    from gptqmodel.utils.allspark import (
        allspark_gemm,
        allspark_supported,
        prewarm_allspark_extension,
    )

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    assert prewarm_allspark_extension(), "AllSpark extension failed to build"

    a, packed, packed_scales, wide_bias, _, _ = _allspark_bias_case(dtype, m)
    bias = wide_bias if bias_dtype is torch.float32 else wide_bias.to(dtype)
    out = allspark_gemm(a, packed, packed_scales, packed.shape[0], bias=bias)
    assert out.dtype == dtype
    assert torch.isfinite(out).all(), "bias was added after the FP16/BF16 rounding"
    torch.testing.assert_close(
        out, torch.full_like(out, 16.0 * a.shape[1] - 32768.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("m", [1, 32, 64, 1025])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_allspark_bias_matches_fp64_oracle(m, dtype):
    """Ordinary biases must match the oracle on every execution path."""
    from gptqmodel.utils.allspark import (
        allspark_gemm,
        allspark_repack,
        allspark_supported,
        prewarm_allspark_extension,
    )

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    assert prewarm_allspark_extension(), "AllSpark extension failed to build"

    n, k = 1024, 4096
    gen = torch.Generator(device="cuda").manual_seed(700 + m)
    weight = torch.randn(k, n, dtype=dtype, device="cuda", generator=gen) * 0.02
    scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-8)
    codes = (weight / scale).round().add(128).clamp(0, 255)
    packed, packed_scales, _ = allspark_repack(
        codes.to(torch.uint8).contiguous(), scale.to(dtype), n, k
    )
    a = torch.randn(m, k, dtype=dtype, device="cuda", generator=gen) * 0.02
    bias = (torch.randn(n, dtype=torch.float32, device="cuda", generator=gen) * 2)
    out = allspark_gemm(a, packed, packed_scales, n, bias=bias)
    dense = ((codes.double() - 128.0) * scale.double()).to(dtype)
    ref = a.double() @ dense.double() + bias.double()[None, :]
    assert out.dtype == dtype
    # Permit one output-dtype ULP (the mandatory rounding of the exact FP64
    # sum); residual arithmetic error must meet the 2e-3 gate. The cuBLAS
    # fallback used for m > 1024 accumulates BF16 operands in FP32, so its
    # last-place rounding can differ from the exact sum by a single ULP.
    rounded = ref.to(dtype)
    ulp = (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))).double()
           - rounded.double()).abs()
    excess = ((out.double() - ref).abs() - ulp).clamp_min(0)
    assert torch.all(excess <= 2e-3 + 2e-3 * ref.abs()), float(excess.max())


def test_allspark_bias_rejected_for_wrong_shape_and_device():
    from gptqmodel.utils.allspark import allspark_gemm, allspark_supported

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    a, packed, packed_scales, bias, _, _ = _allspark_bias_case(torch.float16, 2, n=128, k=256)
    n = packed.shape[0]
    with pytest.raises(ValueError, match="one element per output channel"):
        allspark_gemm(a, packed, packed_scales, n, bias=bias[:-1])
    with pytest.raises(ValueError, match="activation device"):
        allspark_gemm(a, packed, packed_scales, n, bias=bias.cpu())
    with pytest.raises(ValueError, match="float16, bfloat16 or float32"):
        allspark_gemm(a, packed, packed_scales, n, bias=bias.double())
    # A 16-bit bias of the other dtype is widened losslessly to FP32 instead of
    # being read with the wrong element size.
    out = allspark_gemm(a, packed, packed_scales, n, bias=bias.to(torch.bfloat16))
    assert out.dtype == torch.float16 and torch.isfinite(out).all()
