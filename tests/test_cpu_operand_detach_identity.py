# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``detach_cpu_operand`` must only copy tensors that alias a checkpoint (#4395, #4785).

The CPU int4 repack detaches the small per-column operands from the shard's
``safetensors`` mapping so a single surviving view cannot pin the whole file
(#4785). That copy must not touch tensors the module owns: ``scales`` staying
the loader's object is an explicit invariant of the int4pack snapshot, which
removed the byte-identical second copy (#4395).
"""

from __future__ import annotations

import mmap
import os

import torch

from gptqmodel.nn_modules.qlinear import detach_cpu_operand


def test_private_tensor_keeps_identity():
    tensor = torch.randn(4, 8)
    assert detach_cpu_operand(tensor) is tensor


def test_none_and_non_tensor_pass_through():
    assert detach_cpu_operand(None) is None
    assert detach_cpu_operand("scales") == "scales"


def test_mapped_checkpoint_tensor_is_copied(tmp_path):
    source = torch.randn(4, 8)
    path = tmp_path / "scales.bin"
    path.write_bytes(source.view(torch.uint8).reshape(-1).numpy().tobytes())

    fd = os.open(path, os.O_RDWR)
    mapping = mmap.mmap(fd, 0)
    os.close(fd)
    try:
        view = torch.frombuffer(mapping, dtype=source.dtype, count=source.numel()).reshape(
            source.shape
        )
        # A file-backed storage is non-resizable; that is what marks it as a
        # checkpoint alias rather than a module-owned allocation.
        assert not view.untyped_storage().resizable()

        detached = detach_cpu_operand(view)

        assert detached is not view
        assert detached.untyped_storage().resizable()
        assert detached.data_ptr() != view.data_ptr()
        assert torch.equal(detached, view)
    finally:
        mapping.close()
