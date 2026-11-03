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
