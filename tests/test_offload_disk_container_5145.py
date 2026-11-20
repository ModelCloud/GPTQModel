# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``offload_to_disk`` must not silently skip a container module (#5145).

The skip check looked at ``module.parameters(recurse=False)`` and
``module.buffers(recurse=False)``. A container such as a multimodal
``model.visual`` tower owns no direct tensors -- everything lives in its
children -- so the call returned without moving anything, and the tower stayed
on the GPU for the whole layer loop. The check now uses the recursive state
that is actually bundled to disk.
"""

import tempfile

import pytest
import torch
import torch.nn as nn

from gptqmodel.utils.offload import offload_to_disk


def _cuda_count(module: nn.Module) -> int:
    return sum(
        1
        for tensor in list(module.parameters()) + list(module.buffers())
        if tensor.device.type == "cuda"
    )


class _Container(nn.Module):
    """A module that owns no tensors directly, like ``model.visual``."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(64, 64)])
        self.merger = nn.Linear(64, 64)


class _Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Container()


def test_container_module_without_direct_tensors_is_offloaded():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to observe an offload")

    model = _Root().to("cuda")
    container = model.visual
    # The shape that made the old check skip: no direct state, recursive state.
    assert list(container.parameters(recurse=False)) == []
    assert list(container.buffers(recurse=False)) == []
    assert _cuda_count(container) == 4

    with tempfile.TemporaryDirectory() as offload_dir:
        offload_to_disk(model=model, module=container, disk_path=offload_dir)

    assert _cuda_count(container) == 0, "container module was left on the GPU"
