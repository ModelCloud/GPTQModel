# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXL3's per-stream tile scratch must be released when a run ends."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as implementation
from gptqmodel.looper import module_looper
from gptqmodel.looper.exllamav3_processor import EXL3Processor
from gptqmodel.looper.module_looper import ModuleLooper


pytestmark = [pytest.mark.cpu]


def _populate_workspace_registry():
    """Install a real CPU scratch pair and return weakrefs to its tensors."""

    implementation.clear_tile_workspace()
    device = torch.device("cpu")
    buffers = implementation._allocate_temp_buffers(device, K=4, batch=1)
    key = (device, 4, object(), 1)
    with implementation._workspace_registry_lock:
        implementation._workspace_buffers[key] = buffers
    return [weakref.ref(tensor) for tensor in buffers]


def test_clear_tile_workspace_drops_registry_and_tensors():
    refs = _populate_workspace_registry()

    assert implementation._workspace_buffers
    assert all(ref() is not None for ref in refs)

    implementation.clear_tile_workspace()

    assert implementation._workspace_buffers == {}
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("raises", [False, True])
def test_looper_releases_tile_workspace_when_run_ends(monkeypatch, raises):
    refs = _populate_workspace_registry()
    assert implementation._workspace_buffers

    looper = object.__new__(ModuleLooper)
    looper.processors = [object.__new__(EXL3Processor)]
    looper.gptq_model = SimpleNamespace()
    looper._restore_offloaded_base_modules = lambda **kwargs: None
    monkeypatch.setattr(module_looper, "release_private_offload_root", lambda model: None)

    if raises:
        def fail(**kwargs):
            raise RuntimeError("boom")

        looper._loop_impl = fail
        with pytest.raises(RuntimeError, match="boom"):
            looper.loop()
    else:
        looper._loop_impl = lambda **kwargs: "done"
        assert looper.loop() == "done"

    assert implementation._workspace_buffers == {}
    assert all(ref() is None for ref in refs)
