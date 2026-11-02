# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSparkLinear: GPTQ 8-bit layout -> repack -> GEMM."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AllSpark needs a CUDA device"
)


def _pack_gptq_8bit(weight: torch.Tensor):
    """Pack a (K, N) fp16 weight the way a GPTQ 8-bit checkpoint stores it."""
    k, n = weight.shape
    scale = (weight.abs().amax(dim=0) / 127).clamp_min(1e-8)
    codes = (weight / scale).round().add(128).clamp(0, 255).to(torch.int32)
    shifts = (torch.arange(4, device=weight.device, dtype=torch.int32) * 8).view(1, 4, 1)
    qweight = (codes.view(k // 4, 4, n) << shifts).sum(dim=1).to(torch.int32)
    return qweight.contiguous(), scale.view(1, n).to(weight.dtype), codes


@pytest.mark.parametrize("m", [1, 16, 64])
def test_allspark_linear_matches_dequant(m):
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
    from gptqmodel.utils.allspark import allspark_supported, prewarm_allspark_extension

    if not allspark_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0")
    assert prewarm_allspark_extension()

    k, n = 256, 512
    weight = torch.randn(k, n, dtype=torch.float16, device="cuda")
    qweight, scales, codes = _pack_gptq_8bit(weight)

    mod = AllSparkLinear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=False,
    ).cuda().half()
    mod.qweight.data.copy_(qweight)
    mod.scales.data.copy_(scales)
    mod.post_init()

    x = torch.randn(m, k, dtype=torch.float16, device="cuda") * 0.02
    out = mod(x)
    ref = x @ ((codes.float() - 128) * scales.float()).half()

    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel < 0.02, f"m={m} relative error {rel.item():.5f}"
