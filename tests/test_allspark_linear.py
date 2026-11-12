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


def _load_gptq_module(dtype, bias=True, k=256, n=512):
    """A GPTQ AllSpark layer carrying real checkpoint tensors."""
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear

    torch.manual_seed(7)
    weight = (torch.randn(k, n, dtype=torch.float32, device="cuda") * 0.02).to(dtype)
    qweight, scales, codes = _pack_gptq_8bit(weight)
    module = AllSparkLinear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=bias, dtype=dtype,
    ).cuda().to(dtype)
    module.qweight.data.copy_(qweight)
    module.scales.data.copy_(scales)
    if bias:
        module.bias.data.uniform_(-0.01, 0.01)
    return module, qweight, scales, codes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_module_export_keeps_checkpoint_layout(dtype):
    """``post_init()`` must not drop the checkpoint tensors from the export.

    The kernel layout is derived state; deleting ``qweight``/``scales`` left
    ``state_dict()``/``get_state_dict_for_save()`` with only ``bias`` and made a
    strict reload of the original checkpoint fail.
    """
    from gptqmodel.utils.model import get_state_dict_for_save

    if not _kernel_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0 or gfx1200/gfx1201")
    assert _prewarm()

    module, qweight, scales, codes = _load_gptq_module(dtype)
    pristine = {name: tuple(t.shape) for name, t in module.state_dict().items()}
    assert set(pristine) == {"qweight", "scales", "bias"}

    module.post_init()
    assert {name: tuple(t.shape) for name, t in module.state_dict().items()} == pristine
    assert {name: tuple(entry.shape) for name, entry in get_state_dict_for_save(module).items()} == pristine
    # The kernel layout is derived, not part of the checkpoint.
    assert set(dict(module.named_buffers())) == {"bias", "qweight_packed", "scales_packed"}
    assert "qweight_packed" not in dict(module.state_dict())

    x = (torch.randn(4, module.in_features, dtype=torch.float32, device="cuda") * 0.02).to(dtype)
    baseline = module(x).clone()

    # A strict reload of the original checkpoint keeps working after a forward.
    checkpoint = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    module.load_state_dict(checkpoint, strict=True)
    restored = module(x)
    torch.testing.assert_close(restored.float(), baseline.float(), rtol=2e-3, atol=2e-3)

    # The exported checkpoint must load into a fresh module without drop/extra
    # keys: the reported failure mode was a strict load of a bias-only dict.
    fresh, _, _, _ = _load_gptq_module(dtype)
    fresh.load_state_dict(checkpoint, strict=True)
    fresh.post_init()
    torch.testing.assert_close(fresh(x).float(), baseline.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_runtime_state_dict_roundtrip_and_cache_invalidation(dtype):
    """A self round-trip stays valid and stale kernel layout is never served."""
    if not _kernel_supported():
        pytest.skip("AllSpark requires compute capability >= 8.0 or gfx1200/gfx1201")
    assert _prewarm()

    module, qweight, scales, codes = _load_gptq_module(dtype)
    module.post_init()
    x = (torch.randn(2, module.in_features, dtype=torch.float32, device="cuda") * 0.02).to(dtype)
    baseline = module(x).clone()

    runtime_state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    module.load_state_dict(runtime_state, strict=True)
    # ``load_state_dict`` retires the cache; the next forward re-derives it.
    assert not module.packed_ready()
    torch.testing.assert_close(module(x), baseline, rtol=2e-3, atol=2e-3)
    assert module.packed_ready()

    # Mutating the checkpoint scales must not leave a stale kernel layout.
    with torch.no_grad():
        module.scales.mul_(0.5)
    assert not module.packed_ready()
    ref = (
        x.double() @ ((codes.double() - 128) * (scales.double() * 0.5)).to(dtype).double()
        + module.bias.double()
    )
    torch.testing.assert_close(module(x), ref.to(dtype), rtol=2e-3, atol=2e-3)

    # A device/dtype move keeps the checkpoint layout and rebuilds the cache.
    module.to(torch.bfloat16 if dtype is torch.float16 else torch.float16)
    assert {name: tensor.shape for name, tensor in module.state_dict().items()} == {
        "qweight": torch.Size([64, 512]),
        "scales": torch.Size([1, 512]),
        "bias": torch.Size([512]),
    }
