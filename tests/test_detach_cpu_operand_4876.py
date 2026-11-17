# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4876: ``detach_cpu_operand`` copies only a checkpoint-backed tensor.

The helper was added for #4785 so a CPU int4 repack does not keep a
``safetensors`` mapping pinned through ``g_idx``/``scales`` views. It cloned
unconditionally, which duplicated buffers the module already owns and broke the
#4395 identity assertion. A file-backed storage is not resizable; an ordinary
allocation is, which is the discriminator pinned here.
"""

from __future__ import annotations

import mmap

import pytest
import torch

from gptqmodel.nn_modules.qlinear import detach_cpu_operand


def _mmap_tensor(values: torch.Tensor):
    """Return ``values`` as a read-only, file-backed ``mmap`` tensor."""

    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    handle.write(values.numpy().tobytes())
    handle.flush()
    handle.close()
    file_obj = open(handle.name, "rb")  # noqa: SIM115 - released by the caller
    mapping = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
    return torch.frombuffer(mapping, dtype=values.dtype), file_obj, mapping


def test_module_owned_tensor_keeps_its_identity():
    tensor = torch.arange(8, dtype=torch.int32)

    assert tensor.untyped_storage().resizable()
    assert detach_cpu_operand(tensor) is tensor


def test_module_owned_view_keeps_its_identity():
    base = torch.arange(16, dtype=torch.int32)
    view = base[::2]

    assert detach_cpu_operand(view) is view


def test_file_backed_tensor_is_copied():
    source = torch.arange(8, dtype=torch.int32)
    mapped, file_obj, mapping = _mmap_tensor(source)
    try:
        assert not mapped.untyped_storage().resizable()
        detached = detach_cpu_operand(mapped)
        assert detached is not mapped
        assert detached.untyped_storage().data_ptr() != mapped.untyped_storage().data_ptr()
        assert detached.untyped_storage().resizable()
        assert torch.equal(detached, mapped)
    finally:
        mapping.close()
        file_obj.close()


def test_non_owning_buffer_is_copied():
    buffer = bytearray(torch.arange(8, dtype=torch.int32).numpy().tobytes())
    mapped = torch.frombuffer(buffer, dtype=torch.int32)

    assert not mapped.untyped_storage().resizable()
    detached = detach_cpu_operand(mapped)
    assert detached is not mapped
    assert torch.equal(detached, mapped)


def test_none_and_non_tensors_pass_through():
    assert detach_cpu_operand(None) is None
    sentinel = object()
    assert detach_cpu_operand(sentinel) is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_tensor_passes_through_unchanged():
    tensor = torch.arange(8, dtype=torch.int32, device="cuda")

    assert detach_cpu_operand(tensor) is tensor


def test_int4pack_keeps_scales_identity():
    """The #4395 item-1 contract: the repack must not rebind ``scales``."""

    from tests.test_cpu_int4_snapshot_dedup_4395 import CASES, _run, _tensor_state

    if not CASES:
        pytest.skip("no int4pack cases available")

    param = CASES[0]
    module = param.values[1]()
    live_scales = module.scales
    state = _tensor_state(module)
    _run(module, torch.randn(2, module.in_features, dtype=torch.float32))
    assert module.scales is live_scales
    assert module.scales is state["scales"] or torch.equal(module.scales, state["scales"])
