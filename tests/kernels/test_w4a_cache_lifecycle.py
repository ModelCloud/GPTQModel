# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hardware weight caches retain their formats across model lifecycle operations."""

import pytest
import torch

from tests.kernels.test_w4a_nvfp4_gb10 import _module
from tests.kernels.test_w4a_stream import _independent_nvfp4_qdq
from tests.kernels.test_w4afp8_gb10 import _packed_linear
from tests.w4a_hardware_marks import FP8_HARDWARE, NVFP4_HARDWARE


def _staged(mode):
    factory = _module if mode == "w4a_nvfp4" else _packed_linear
    return factory(k=256, n=128, device="cpu")


def _cache_names(mode: str) -> list[str]:
    return ["_weight_e4m3", "_weight_both", "_unit_weight_scales"] if mode == "w4a_nvfp4" else ["_weight_e4m3"]


@pytest.mark.parametrize("mode", [
    pytest.param("w4afp8", marks=FP8_HARDWARE),
    pytest.param("w4a_nvfp4", marks=NVFP4_HARDWARE),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("method", ["to", "cast"])
def test_model_dtype_conversion_preserves_hardware_cache_bytes(mode, dtype, method):
    module = _staged(mode)
    caches = {name: (getattr(module, name).dtype, getattr(module, name).view(torch.uint8).clone())
              for name in _cache_names(mode)}
    native = module.qweight.clone()
    parent = torch.nn.Sequential(module)
    if method == "to":
        parent.to(dtype=dtype)
    elif dtype == torch.bfloat16:
        parent.bfloat16()
    else:
        parent.half()
    assert module.scales.dtype == dtype
    assert torch.equal(module.qweight, native)
    for name, (cache_dtype, cache_bytes) in caches.items():
        actual = getattr(module, name)
        assert actual.dtype == cache_dtype
        assert torch.equal(actual.view(torch.uint8), cache_bytes)
        assert name not in module.state_dict()


@pytest.mark.parametrize("mode", [
    pytest.param("w4afp8", marks=FP8_HARDWARE),
    pytest.param("w4a_nvfp4", marks=NVFP4_HARDWARE),
])
@pytest.mark.parametrize("operation", ["pack", "load"])
def test_native_weight_changes_invalidate_every_derived_cache(mode, operation):
    module = _staged(mode)
    if operation == "pack":
        dense = torch.nn.Linear(256, 128, bias=False)
        dense.weight.data.zero_()
        module.pack(dense, torch.ones(128, 2), torch.full((128, 2), 8.), module.g_idx.clone())
        expected_code = 0
    else:
        state = {name: value.clone() for name, value in module.state_dict().items()}
        state["qweight"].zero_()
        module.load_state_dict(state)
        expected_code = -8
    assert all(getattr(module, name).numel() == 0 for name in _cache_names(mode))
    with pytest.raises(RuntimeError, match="cache is absent"):
        module(torch.zeros(1, 256, dtype=torch.float16))
    module.post_init()
    assert all(getattr(module, name).numel() for name in _cache_names(mode))
    assert torch.equal(module._weight_e4m3.float(), torch.full((256, 128), float(expected_code)))


@pytest.mark.parametrize("mode", [
    pytest.param("w4afp8", marks=FP8_HARDWARE),
    pytest.param("w4a_nvfp4", marks=NVFP4_HARDWARE),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cast_and_device_move_consume_native_operands_without_restaging(monkeypatch, mode, dtype):
    module = _staged(mode)

    def unexpected_post_init():
        pytest.fail("A dtype/device move must preserve caches without restaging the weights")

    monkeypatch.setattr(module, "post_init", unexpected_post_init)
    module.to(device="cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(2415)
    x = torch.randn((7, 256), device="cuda", dtype=dtype, generator=generator)
    x[0].zero_()
    if mode == "w4afp8":
        maximum = x.float().abs().amax(-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 448., torch.ones_like(maximum))
        operand = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    else:
        operand = _independent_nvfp4_qdq(x, module.activation_global_scale)
    codes = torch.arange(256, device="cuda", dtype=torch.float32).remainder(16) - 8
    expected = torch.zeros((7, 128), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        expected += (operand[:, sl] @ codes[sl, None].expand(128, 128)) * ((group + 1) / 8.)
    actual = module(x)
    module.cpu().cuda()
    moved = module(x)
    torch.cuda.synchronize()
    assert actual.dtype == moved.dtype == dtype
    torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(moved, actual, rtol=0, atol=0)
