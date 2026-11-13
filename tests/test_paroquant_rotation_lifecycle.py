# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ParoQuant rotation buffers must govern the identity shortcut."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear


def _module() -> ParoLinear:
    module = ParoLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=8,
        bias=False,
        register_buffers=True,
    ).eval()
    with torch.no_grad():
        # Every packed 4-bit weight code is 3, every zero code is 2, and
        # each scale is 1. The independently known dense weight is all ones.
        module.qweight.fill_(0x33333333)
        module.qzeros.fill_(0x22222222)
        module.scales.fill_(1)
    module.post_init()
    return module


def _input(dtype: torch.dtype = torch.float16) -> torch.Tensor:
    x = torch.zeros((2, 128), dtype=dtype)
    x[0, 0] = 5
    x[1, 0] = -3
    return x


def _torch_oracle(module: ParoLinear, x: torch.Tensor) -> torch.Tensor:
    """Direct pair arithmetic and known packed-code weight, independent of forward."""
    rotated = x.float() * module.channel_scales.float()
    a, b = rotated[:, 0].clone(), rotated[:, 1].clone()
    angle = module.theta[0, 0].float()
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated[:, 0] = a * cosine + b * sine
    rotated[:, 1] = -a * sine + b * cosine
    return rotated.sum(dim=1, keepdim=True).expand(-1, module.out_features).to(x.dtype)


def _assert_matches_oracle(module: ParoLinear, x: torch.Tensor) -> None:
    actual = module(x)
    torch.testing.assert_close(actual, _torch_oracle(module, x), rtol=2e-3, atol=2e-3)
    assert module.qweight.shape == (128, 1)
    assert not any(
        isinstance(value, torch.Tensor) and value.shape == (128, 8)
        for value in vars(module).values()
    )


@pytest.mark.parametrize("changed_buffer", ["theta", "channel_scales"])
def test_clear_weight_cache_rechecks_rotation_buffers(changed_buffer):
    module = _module()
    x = _input()
    _assert_matches_oracle(module, x)
    assert module._rotation_identity

    with torch.no_grad():
        if changed_buffer == "theta":
            module.theta[0, 0] = 0.5
        else:
            module.channel_scales[0, 0] = 1.25
    module.clear_weight_cache()

    _assert_matches_oracle(module, x)
    _assert_matches_oracle(module, x)
    assert not module._rotation_identity


def test_clear_weight_cache_restores_identity_shortcut():
    module = _module()
    x = _input()
    with torch.no_grad():
        module.theta[0, 0] = 0.5
        module.channel_scales[0, 0] = 1.25
    module.clear_weight_cache()
    assert not module._rotation_identity

    with torch.no_grad():
        module.theta.zero_()
        module.channel_scales.fill_(1)
    module.clear_weight_cache()

    assert module._rotation_identity
    assert module._rotate_inputs(x) is x
    _assert_matches_oracle(module, x)


def test_explicit_clear_rechecks_inference_tensor_without_version_counter():
    module = _module()
    x = _input()
    with torch.no_grad():
        module.theta[0, 0] = 0.25
    module.post_init()
    _assert_matches_oracle(module, x)

    with torch.inference_mode():
        replacement = module.theta.clone()
        replacement[0, 0] = 0.5
        module.theta = replacement
    module.clear_weight_cache()

    _assert_matches_oracle(module, x)
    assert not module._rotation_identity


@pytest.mark.parametrize("source_rotated", [False, True])
@pytest.mark.parametrize("assign", [False, True])
def test_state_dict_load_rechecks_identity_in_both_directions(source_rotated, assign):
    source, target = _module(), _module()
    x = _input()
    with torch.no_grad():
        changed = source if source_rotated else target
        changed.theta[0, 0] = 0.5
        changed.channel_scales[0, 0] = 1.25
    changed.post_init()
    target.load_state_dict(source.state_dict(), assign=assign)

    _assert_matches_oracle(target, x)
    assert target._rotation_identity is not source_rotated


def test_dtype_move_rechecks_identity_after_rotation_mutation():
    module = _module().to(dtype=torch.bfloat16)
    x = _input()
    assert module._rotation_identity
    with torch.no_grad():
        module.theta[0, 0] = 0.5
    module.to(dtype=torch.float16)

    assert module.theta.dtype == torch.float16
    _assert_matches_oracle(module, x)
    assert not module._rotation_identity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP runtime unavailable")
def test_device_move_rechecks_identity_after_rotation_mutation():
    module = _module()
    with torch.no_grad():
        module.theta[0, 0] = 0.5
    module.to("cuda")
    assert not module._rotation_identity

    module.to("cpu")
    _assert_matches_oracle(module, _input())


def test_meta_offload_then_state_dict_load_rechecks_rotation():
    source, target = _module(), _module()
    with torch.no_grad():
        source.theta[0, 0] = 0.5
    source.post_init()

    target.to_empty(device="meta")
    assert not target._rotation_identity
    target.load_state_dict(source.state_dict(), assign=True)

    _assert_matches_oracle(target, _input())
    assert not target._rotation_identity
