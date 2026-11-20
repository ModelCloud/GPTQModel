# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4805: the QQQ CPU packed cache must not duplicate the checkpoint.

``QQQTorchLinear`` is AUTO for QQQ on CPU (FP16 and BF16). Its first forward
builds an INT4 cache for the AVX512 ``qqq_int4`` kernel; the checkpoint
``B``/``s_group``/``s_channel`` then stayed resident next to it, so the packed
weight was stored twice and the group scales were widened to FP32.

The cache now rebuilds the checkpoint tensors bit for bit, verifies them, and
only then replaces the checkpoint layout. ``state_dict()``/save/load and moves
rebuild the checkpoint tensors on demand. The AVX512 kernel itself is x86-only,
so these tests drive the cache with a deterministic stand-in kernel.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear import qqq as qqq_module
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear


def _build(
    *, group_size: int = 128, dtype: torch.dtype = torch.float16
) -> QQQTorchLinear:
    from tests.test_qqq_kernel_memory_4387 import _build_torch_module

    return _build_torch_module(
        in_features=256,
        out_features=128,
        group_size=group_size,
        dtype=dtype,
    )


class _FakePackedKernel:
    """Deterministic ``qqq_int4`` stand-in that reads every operand."""

    def __init__(self):
        self.calls = 0

    def __call__(self, activation, packed, scales, group_size):
        assert activation.dtype == torch.int8
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.float32
        self.calls += 1
        weight = packed.to(torch.float32).sum(dim=1)
        scale = scales.to(torch.float32).sum() if scales.numel() else 1.0
        return activation.to(torch.float32).sum(dim=1, keepdim=True) + weight.unsqueeze(0) * scale


def _unique_tensor_bytes(module) -> int:
    """Resident bytes owned by the module, counting each storage once."""

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
    for value in getattr(module, "_torch_packed_cache", None) or ():
        add(value)
    return total


@pytest.fixture
def packed_kernel(monkeypatch):
    kernel = _FakePackedKernel()
    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", lambda: kernel)
    return kernel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [128, -1])
def test_qqq_cpu_cache_replaces_the_checkpoint_layout(packed_kernel, dtype, group_size):
    module = _build(group_size=group_size, dtype=dtype)
    checkpoint_bytes = _unique_tensor_bytes(module)
    activation = torch.randn(4, module.in_features, dtype=dtype)

    with torch.inference_mode():
        first = module(activation).clone()

    resident = _unique_tensor_bytes(module)
    assert packed_kernel.calls == 1
    assert resident <= checkpoint_bytes * 1.1, (
        f"{dtype} group_size={group_size}: {resident} resident bytes after the "
        f"first forward vs {checkpoint_bytes} checkpoint bytes"
    )
    assert module._torch_dropped_checkpoint is not None
    for name in ("B", "s_group", "s_channel"):
        assert module._buffers[name] is None, name
    # The cache keeps the checkpoint scale dtype; FP32 would double them.
    packed, s_group, s_channel = module._torch_packed_cache
    assert s_group.dtype == s_channel.dtype == dtype
    assert packed.dtype == torch.uint8

    with torch.inference_mode():
        assert torch.equal(module(activation), first)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [128, -1])
def test_qqq_cpu_state_dict_rebuilds_checkpoint_bit_identically(packed_kernel, dtype, group_size):
    module = _build(group_size=group_size, dtype=dtype)
    original = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    activation = torch.randn(3, module.in_features, dtype=dtype)

    with torch.inference_mode():
        module(activation)
    assert module._buffers["B"] is None

    rebuilt = module.state_dict()
    assert set(rebuilt) == set(original)
    for name in ("B", "s_group", "s_channel"):
        assert rebuilt[name].dtype == original[name].dtype, name
        assert rebuilt[name].shape == original[name].shape, name
        assert torch.equal(rebuilt[name], original[name]), name
    # Serializing must not reintroduce the checkpoint copy.
    assert module._buffers["B"] is None


def test_qqq_cpu_load_and_move_restore_the_checkpoint_layout(packed_kernel):
    module = _build()
    original = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    activation = torch.randn(2, module.in_features, dtype=torch.float16)

    with torch.inference_mode():
        module(activation)
    assert module._buffers["B"] is None

    module.load_state_dict(original)
    assert module._torch_dropped_checkpoint is None
    assert module._torch_packed_cache is None
    for name in ("B", "s_group", "s_channel"):
        assert torch.equal(module._buffers[name], original[name]), name

    # Drop again, then move: a device move must restore the checkpoint layout
    # and discard the cache built for the old placement. A dtype cast after the
    # first forward is rejected (contracts/kernel/forward-dtype.md).
    with torch.inference_mode():
        module(activation)
    assert module._buffers["B"] is None
    with pytest.raises(RuntimeError, match="frozen after the first forward"):
        module.to(dtype=torch.bfloat16)
    # The rejected cast may restore the checkpoint layout (same decoded
    # weights), but no tensor dtype changes.
    assert all(tensor.dtype != torch.bfloat16 for tensor in module.buffers())
    moved = module.to("cpu")
    assert moved is module
    assert module._torch_dropped_checkpoint is None
    assert module._torch_packed_cache is None
    assert module._buffers["B"] is not None
    with torch.inference_mode():
        output = module(activation)
    assert output.dtype == torch.float16
    assert output.shape == (2, module.out_features)


def test_qqq_cpu_fallback_after_drop_rebuilds_transiently(monkeypatch, packed_kernel):
    """The dequantized fallback needs ``B``/scales without re-residing them."""

    module = _build()
    with torch.inference_mode():
        module(torch.randn(2, module.in_features, dtype=torch.float16))
    assert module._buffers["B"] is None

    monkeypatch.setattr(qqq_module, "_qqq_cpu_kernel", lambda: None)
    with torch.inference_mode():
        output = module(torch.randn(2, module.in_features, dtype=torch.float16))
    assert output.shape == (2, module.out_features)
    assert output.dtype == torch.float16
    # The fallback rebuilt the checkpoint weight for one call only.
    assert module._buffers["B"] is None


def test_qqq_cpu_failed_inverse_keeps_both_layouts(monkeypatch):
    """Fail closed: an unverified inverse must never drop the checkpoint."""

    module = _build()
    monkeypatch.setattr(
        QQQTorchLinear,
        "_rebuild_checkpoint_B",
        lambda self: torch.zeros(
            (self.in_features // self.tile, self.out_features * 2), dtype=torch.int32
        ),
    )
    assert module._packed_weight_for_cpu() is not None
    assert module._torch_dropped_checkpoint is None
    assert module._buffers["B"] is not None
    assert module._torch_packed_cache is not None
