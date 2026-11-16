# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The process-wide Hessian workspace pool must be releasable after a run."""

import threading
import types

import pytest
import torch

from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.quantization import gptq as gptq_module


pytestmark = [pytest.mark.cpu]


def test_release_workspace_cache_drops_pooled_slots():
    gptq_module.release_workspace_cache()
    device = torch.device("cpu")
    key = gptq_module._workspace_cache_key(device)

    with gptq_module._lease_workspace(device, torch.float32, cols=4, required_rows=8) as (workspace, reused):
        assert workspace.shape == (8, 4)
        assert reused is False

    assert key in gptq_module._WORKSPACE_CACHE
    assert any(slot_key[0] == key for slot_key in gptq_module._WORKSPACE_SLOTS)

    gptq_module.release_workspace_cache()

    assert key not in gptq_module._WORKSPACE_CACHE
    assert all(slot_key[0] != key for slot_key in gptq_module._WORKSPACE_SLOTS)


def test_release_workspace_cache_drops_an_in_flight_lease_instead_of_repopulating():
    gptq_module.release_workspace_cache()
    device = torch.device("cpu")
    key = gptq_module._workspace_cache_key(device)
    entered = threading.Event()
    release_lease = threading.Event()

    def hold():
        with gptq_module._lease_workspace(device, torch.float32, cols=4, required_rows=8):
            entered.set()
            assert release_lease.wait(10)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert entered.wait(10), "the lease never started"

    gptq_module.release_workspace_cache()
    release_lease.set()
    worker.join(10)
    assert not worker.is_alive()

    assert key not in gptq_module._WORKSPACE_CACHE
    assert all(slot_key[0] != key for slot_key in gptq_module._WORKSPACE_SLOTS)


def test_loop_processor_finalize_releases_workspace_pool(monkeypatch):
    called = []
    monkeypatch.setattr(
        gptq_module, "release_workspace_cache", lambda: called.append(True)
    )

    processor = object.__new__(LoopProcessor)
    processor._close_device_smi_handles = lambda: None
    processor.inputs_cache = {}
    processor._results = {}
    model = types.SimpleNamespace(
        quantize_config=types.SimpleNamespace(keep_processor_logs=False)
    )

    processor.finalize(model)

    assert called == [True]
