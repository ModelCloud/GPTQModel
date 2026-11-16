# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4395 item 1: the CPU int4pack checkpoint snapshot keeps no duplicate.

The int4pack transform (TorchFused, TorchAten and their AWQ variants) used to
clone ``scales`` even though it leaves that buffer byte-identical, and kept a
standalone ``qzeros`` copy of the zero column already stored inside
``scales_and_zeros``. Both copies are removed without changing the forward
numerics or a single checkpoint byte.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file


def _int4pack_cases():
    from tests.test_checkpoint_state_invariant import _awq_cases, _gptq_cases

    cases = []
    for name, build, _device in _gptq_cases():
        if name in ("torch_aten", "torch_fused"):
            dtype = torch.bfloat16 if name == "torch_fused" else torch.float16
            cases.append(pytest.param(name, build, dtype, id=name))
    for name, build, _device, dtype in _awq_cases():
        if name in ("awq_torch_aten", "awq_torch_fused"):
            cases.append(pytest.param(name, build, dtype, id=name))
    return cases


CASES = _int4pack_cases()


def _run(module, x):
    with torch.inference_mode():
        return module(x).clone()


def _tensor_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


@pytest.mark.parametrize(("name", "build", "dtype"), CASES)
def test_int4pack_snapshot_drops_duplicate_operands(name, build, dtype):
    module = build().to("cpu").eval()
    checkpoint = _tensor_state(module)
    # ``transform_cpu`` must not rebind or clone the checkpoint scales buffer.
    live_scales = module.scales
    x = torch.randn(2, module.in_features, dtype=dtype)
    _run(module, x)

    snapshot = module._fused_train_state
    assert snapshot is not None, f"{name}: the int4pack transform did not run"
    # Only the two operands the transform actually rewrites are snapshotted.
    assert set(snapshot) == {"qweight", "qzeros"}, name
    assert sorted(
        buffer for buffer in module._buffers if buffer.startswith("_fused_train_state_")
    ) == ["_fused_train_state_qweight", "_fused_train_state_qzeros"], name

    # ``scales`` stays the same tensor object: no byte-identical second copy.
    assert module.scales is live_scales, name

    # ``qzeros`` is a view of the zero column of the packed affine metadata.
    packed = module.scales_and_zeros
    assert module.qzeros.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr(), name
    assert module.qzeros.data_ptr() != packed.data_ptr(), name
    assert torch.equal(module.qzeros, packed[..., 1]), name
    # The qzeros view is inside the metadata allocation, so it adds no bytes.
    assert module.qzeros.untyped_storage().nbytes() == packed.untyped_storage().nbytes(), name

    # The emitted checkpoint is bitwise identical to the pre-transform one.
    emitted = _tensor_state(module)
    assert set(emitted) == set(checkpoint), name
    for key, expected in checkpoint.items():
        actual = emitted[key]
        assert actual.dtype == expected.dtype, f"{name}: {key} dtype changed"
        assert actual.shape == expected.shape, f"{name}: {key} shape changed"
        assert torch.equal(actual, expected), f"{name}: {key} is not bitwise identical"


@pytest.mark.parametrize(("name", "build", "dtype"), CASES)
def test_int4pack_save_reload_round_trip_is_byte_identical(name, build, dtype, tmp_path):
    module = build().to("cpu").eval()
    x = torch.randn(2, module.in_features, dtype=dtype)
    expected = _run(module, x)

    first = tmp_path / "transformed.safetensors"
    save_file(
        {
            key: value.contiguous()
            for key, value in module.state_dict().items()
            if torch.is_tensor(value)
        },
        first,
    )

    fresh = build().to("cpu").eval()
    result = fresh.load_state_dict(module.state_dict(), strict=True)
    assert not result.missing_keys, f"{name}: {result.missing_keys}"
    assert not result.unexpected_keys, f"{name}: {result.unexpected_keys}"
    assert torch.equal(_run(fresh, x), expected), f"{name}: reloaded forward differs"

    second = tmp_path / "reloaded.safetensors"
    save_file(
        {
            key: value.contiguous()
            for key, value in fresh.state_dict().items()
            if torch.is_tensor(value)
        },
        second,
    )
    assert second.read_bytes() == first.read_bytes(), f"{name}: saved bytes differ"
