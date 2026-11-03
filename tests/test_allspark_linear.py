# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSparkLinear: GPTQ 8-bit layout -> repack -> GEMM.

Runs on both device paths of the ``allspark`` backend: CUDA (Ampere+) and
ROCm RDNA4 (gfx1200/gfx1201, WMMA port). The module must pick the matching
repack/kernel pair from the runtime device, so this is also the guard against
silently pairing a CUDA-packed weight with the HIP kernel.
"""

import pytest
import torch

from gptqmodel.utils.rocm import IS_ROCM


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


def _kernel_supported() -> bool:
    if IS_ROCM:
        from gptqmodel.utils.allspark_hip import allspark_hip_supported

        return allspark_hip_supported()
    from gptqmodel.utils.allspark import allspark_supported

    return allspark_supported()


def _prewarm() -> bool:
    if IS_ROCM:
        from gptqmodel.utils.allspark_hip import prewarm_allspark_hip_extension

        return prewarm_allspark_hip_extension()
    from gptqmodel.utils.allspark import prewarm_allspark_extension

    return prewarm_allspark_extension()


def test_gptq_load_only_packing_contract():
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
    from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
    from gptqmodel.utils.importer import _supports_pack_api

    assert not _supports_pack_api(AllSparkLinear)
    assert _supports_pack_api(AllSparkAwqLinear)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 16, 64])
def test_allspark_linear_matches_dequant(m, dtype):
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear

    if not _kernel_supported():
        pytest.skip(
            "AllSpark requires compute capability >= 8.0 (CUDA) or gfx1200/gfx1201 (ROCm)"
        )
    assert _prewarm()

    k, n = 256, 512
    torch.manual_seed(42 + m)
    # Random, non-zero weights: a zero weight would let the hardware skip work
    # and would hide both repack bugs and dequant bugs.
    weight = torch.randn(k, n, dtype=torch.float32, device="cuda")
    weight = (weight * 0.02).to(dtype)
    qweight, scales, codes = _pack_gptq_8bit(weight)

    mod = AllSparkLinear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=False, dtype=dtype,
    ).cuda().to(dtype)
    mod.qweight.data.copy_(qweight)
    mod.scales.data.copy_(scales)
    mod.post_init()

    x = (torch.randn(m, k, dtype=torch.float32, device="cuda") * 0.02).to(dtype)
    out = mod(x)
    assert out.dtype == dtype
    # Reference in fp64 from the original codes: no reuse of kernel state.
    ref = x.double() @ ((codes.double() - 128) * scales.double()).to(dtype).double()

    rel = (out.double() - ref).abs().mean() / ref.abs().mean()
    assert torch.isfinite(out).all()
    assert rel < 2e-3, f"m={m} relative error {rel.item():.5f}"
