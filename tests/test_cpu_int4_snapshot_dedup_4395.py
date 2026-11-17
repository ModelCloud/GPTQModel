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

from gptqmodel.nn_modules.qlinear._cpu_int4pack_layout import (
    int4pack_block_size,
    int4pack_to_codes,
)
from gptqmodel.nn_modules.qlinear._fused_train_state import (
    RebuildCandidate,
    RebuiltCheckpointTensor,
)


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


def _unique_storage_bytes(module):
    """Resident bytes owned by the module's tensors, counting each storage once."""

    seen = set()
    total = 0

    def add(value):
        nonlocal total
        if not isinstance(value, torch.Tensor):
            return
        storage = value.untyped_storage()
        if storage.data_ptr() in seen:
            return
        seen.add(storage.data_ptr())
        total += storage.nbytes()

    for value in (*module._parameters.values(), *module._buffers.values()):
        add(value)
    for name in ("scales_and_zeros", "ret_idx", "torch_fused_op"):
        value = getattr(module, name, None)
        if isinstance(value, torch.Tensor):
            add(value)
        elif isinstance(value, torch.nn.Module):
            for buffer in value.buffers():
                add(buffer)
    return total


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
    # ``qweight`` is described, not stored: the verified inverse rebuilds it.
    assert isinstance(snapshot["qweight"], RebuiltCheckpointTensor), name
    assert snapshot["qweight"].shape == tuple(checkpoint["qweight"].shape), name
    assert snapshot["qweight"].dtype == checkpoint["qweight"].dtype, name
    assert sorted(
        buffer for buffer in module._buffers if buffer.startswith("_fused_train_state_")
    ) == ["_fused_train_state_qzeros"], name

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
def test_int4pack_qweight_snapshot_is_not_resident(name, build, dtype):
    """The resident state grows by the affine metadata, not a second qweight."""

    module = build().to("cpu").eval()
    checkpoint_qweight = module.qweight.detach().clone()
    qweight_bytes = checkpoint_qweight.numel() * checkpoint_qweight.element_size()
    before = _unique_storage_bytes(module)
    x = torch.randn(2, module.in_features, dtype=dtype)
    _run(module, x)
    after = _unique_storage_bytes(module)

    # A retained checkpoint-layout qweight would add its full size. The
    # transform itself keeps the packed bytes the same size, so the only new
    # resident bytes are the affine metadata (and the qzeros snapshot).
    assert after - before < qweight_bytes, (
        f"{name}: resident bytes grew by {after - before}, "
        f"which includes a {qweight_bytes}-byte qweight duplicate"
    )

    # Exporting rebuilds the checkpoint tensor transiently and must not cache
    # a second copy on the module.
    resident = _unique_storage_bytes(module)
    module.state_dict()
    assert _unique_storage_bytes(module) == resident, name


@pytest.mark.parametrize(
    ("rows", "columns"),
    [(64, 16), (48, 32), (64, 64), (80, 16), (128, 128), (256, 2048), (2048, 256)],
)
def test_cpu_int4pack_inverse_round_trips(rows, columns):
    """The rebuild inverse matches the local op on full and partial blocks."""

    if int4pack_block_size() is None:
        pytest.skip("aten CPU int4pack op is unavailable")
    torch.manual_seed(4395 + rows + columns)
    codes = torch.randint(0, 16, (rows, columns), dtype=torch.int32)
    packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, 1)
    assert torch.equal(int4pack_to_codes(packed).to(torch.int32), codes)


@pytest.mark.parametrize(("name", "build", "dtype"), CASES)
def test_int4pack_keeps_clone_when_inverse_disagrees(name, build, dtype, monkeypatch):
    """A wrong inverse must never reach a save: the clone stays authoritative."""

    module = build().to("cpu").eval()
    checkpoint = _tensor_state(module)
    real_rebuild = type(module)._rebuild_checkpoint_qweight

    def corrupted_rebuild(self):
        tensor = real_rebuild(self).clone()
        tensor[0, 0] ^= 1
        return tensor

    monkeypatch.setattr(type(module), "_rebuild_checkpoint_qweight", corrupted_rebuild)
    _run(module, torch.randn(2, module.in_features, dtype=dtype))

    snapshot = module._fused_train_state
    assert isinstance(snapshot["qweight"], RebuildCandidate), name
    assert "_fused_train_state_qweight" in module._buffers, name

    emitted = _tensor_state(module)
    for key, expected in checkpoint.items():
        assert torch.equal(emitted[key], expected), f"{name}: {key} changed"


def test_finalize_drops_clone_only_after_a_match():
    """``finalize_fused_train_state`` reports whether it removed a duplicate."""

    from gptqmodel.nn_modules.qlinear._fused_train_state import (
        finalize_fused_train_state,
    )

    _name, build, dtype = CASES[0].values
    module = build().to("cpu").eval()
    assert finalize_fused_train_state(module) is False
    _run(module, torch.randn(2, module.in_features, dtype=dtype))
    assert "_fused_train_state_qweight" not in module._buffers
    # A second call has nothing left to verify.
    assert finalize_fused_train_state(module) is False


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
