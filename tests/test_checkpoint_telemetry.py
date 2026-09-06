# SPDX-License-Identifier: Apache-2.0
"""Restore success requires exact bits and placement, not tolerance checks."""

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gptqmodel.looper.checkpoint import CheckpointConfig, CheckpointExtension
from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.extension import LoopExtensions, LoopPlan, LoopStep
from gptqmodel.looper.gptq_checkpoint import GPTQCheckpointAdapter
from gptqmodel.utils.device_telemetry import (
    capture_device_telemetry,
    clear_device_telemetry_records,
    device_telemetry_scope,
    get_device_telemetry_records,
)


@pytest.fixture(autouse=True)
def telemetry():
    clear_device_telemetry_records()
    with device_telemetry_scope(True):
        yield
    clear_device_telemetry_records()


def records(event):
    return [
        record for record in get_device_telemetry_records() if record["event"] == event
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_exact_tensor_and_metadata_telemetry(device):
    if (
        device.startswith("cuda")
        and torch.cuda.device_count() <= torch.device(device).index
    ):
        pytest.skip(f"requires {device}")
    tensor = torch.tensor([0.0, -0.0, float("nan"), float("inf"), 3.5], device=device)
    expected = {"cache": [tensor], "metadata": {"device": torch.device(device)}}
    actual = ContinuationCodec.loads(ContinuationCodec.dumps(expected))
    assert torch.equal(actual["cache"][0].view(torch.uint8), tensor.view(torch.uint8))
    assert actual["cache"][0].device == torch.device(device)
    assert actual["metadata"] == expected["metadata"]
    (captured,) = records("checkpoint_tensor_captured")
    (restored,) = records("checkpoint_tensor_restored")
    assert (
        captured["source_device"]
        == restored["expected_device"]
        == restored["actual_device"]
        == device
    )
    assert (
        restored["matched"]
        and restored["state_matched"]
        and restored["placement_matched"]
    )
    (meta,) = records("checkpoint_device_metadata_restored")
    assert meta["expected_device"] == meta["actual_device"] == device
    assert meta["matched"]


@pytest.mark.parametrize("in_place", [False, True])
def test_corrupted_transfer_fails_even_with_telemetry_disabled(monkeypatch, in_place):
    data = ContinuationCodec.dumps(torch.ones(2))
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda tensor, **kwargs: tensor.add_(1) if in_place else tensor + 1,
    )
    with (
        device_telemetry_scope(False),
        pytest.raises(ValueError, match="state or device placement restore mismatch"),
    ):
        ContinuationCodec.loads(data)


def test_corrupted_transfer_reports_actual_failure(monkeypatch):
    data = ContinuationCodec.dumps(torch.ones(2))
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, **kwargs: tensor + 1)
    with pytest.raises(ValueError, match="restore mismatch"):
        ContinuationCodec.loads(data)
    (restored,) = records("checkpoint_tensor_restored")
    assert restored["placement_matched"] and not restored["state_matched"]
    assert not restored["matched"]


def test_wrong_gpu_index_reports_failure(monkeypatch):
    from test_checkpoint_devices import test_codec_uses_saved_logical_index

    test_codec_uses_saved_logical_index(monkeypatch)
    (restored,) = records("checkpoint_tensor_restored")
    assert restored["expected_device"] == "cuda:1"
    assert restored["actual_device"] == "cpu"
    assert restored["state_matched"] and not restored["placement_matched"]
    assert not restored["matched"]


class Adapter:
    def capture(self):
        return {"tensor": torch.ones(1)}, {}

    def restore(self, state, artifacts):
        self.state = state


PLAN = LoopPlan((LoopStep("layer", 0, "model.layers.0"),))


def commit(root):
    with CheckpointExtension(CheckpointConfig(root), Adapter()) as extension:
        extension.prepare(PLAN, {"device_topology": {"gpu_count": 0}})
        LoopExtensions([extension]).publish(PLAN.steps[0])


def test_lifecycle_order_and_commit(tmp_path):
    commit(tmp_path)
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        assert extension.prepare(PLAN, {"device_topology": {"gpu_count": 0}}) == 1
    names = [record["event"] for record in get_device_telemetry_records()]
    ordered = [
        "checkpoint_capture_begin",
        "checkpoint_tensor_captured",
        "checkpoint_committed",
        "checkpoint_topology_validated",
        "checkpoint_restore_begin",
        "checkpoint_tensor_restored",
        "checkpoint_restore_complete",
    ]
    assert [names.index(name) for name in ordered] == sorted(
        names.index(name) for name in ordered
    )


def test_topology_rejection_has_no_restore_success(tmp_path):
    commit(tmp_path)
    clear_device_telemetry_records()
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(CheckpointError, match="device_topology"),
    ):
        extension.prepare(PLAN, {"device_topology": {"gpu_count": 2}})
    assert records("checkpoint_prepare_rejected")
    assert not records("checkpoint_restore_begin")
    assert not records("checkpoint_restore_complete")


def test_restore_failure_never_reports_success(tmp_path, monkeypatch):
    commit(tmp_path)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, **kwargs: tensor + 1)
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(ValueError, match="restore mismatch"),
    ):
        extension.prepare(PLAN, {"device_topology": {"gpu_count": 0}})
    assert records("checkpoint_restore_failed")
    assert not records("checkpoint_restore_complete")


def test_failed_publication_never_reports_commit(tmp_path, monkeypatch):
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        extension.prepare(PLAN, {})

        def fail(**kwargs):
            raise OSError("injected publication failure")

        monkeypatch.setattr(extension.store, "commit", fail)
        with pytest.raises(OSError):
            LoopExtensions([extension]).publish(PLAN.steps[0])
    assert records("checkpoint_capture_begin")
    assert not records("checkpoint_committed")


def test_scheduler_mismatch_is_fatal_before_module_restore():
    adapter = GPTQCheckpointAdapter()
    adapter.processors = ()
    adapter.model = SimpleNamespace(quantize_config=None)
    adapter.execution = SimpleNamespace(
        load_execution_state_dict=lambda state: None,
        execution_state_dict=lambda: {
            "next_device": 0,
            "module_devices": {"a": "cuda:0"},
        },
    )
    with pytest.raises(CheckpointError, match="execution placement restore mismatch"):
        adapter.restore(
            {
                "version": adapter.VERSION,
                "specs": {},
                "processors": [],
                "execution": {"next_device": 1, "module_devices": {"a": "cuda:1"}},
            },
            {},
        )
    (event,) = records("checkpoint_execution_restored")
    assert not event["matched"] and event["expected"] != event["actual"]
    assert not records("checkpoint_adapter_restored")


def test_rng_mismatch_is_fatal_before_success(tmp_path, monkeypatch):
    adapter = GPTQCheckpointAdapter()
    adapter.processors = ()
    adapter.model = SimpleNamespace(quantize_config=None)
    adapter.execution = SimpleNamespace(
        load_execution_state_dict=lambda state: None,
        execution_state_dict=dict,
    )
    adapter.processor = SimpleNamespace(
        lock=threading.Lock(), receive_input_cache=lambda cache: None
    )
    adapter.shared_state = {}
    adapter.offload = tmp_path
    numpy_state = np.random.get_state()
    expected_rng = torch.Generator().manual_seed(19463).get_state()
    if torch.equal(expected_rng, torch.random.get_rng_state()):
        expected_rng = torch.Generator().manual_seed(19464).get_state()
    state = {
        "version": adapter.VERSION,
        "specs": {},
        "processors": [],
        "dense": {},
        "execution": {},
        "cache": {},
        "log": [],
        "shared_state": {},
        "rng": expected_rng,
        "cuda_rng": [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    monkeypatch.setattr(torch.random, "set_rng_state", lambda state: None)
    with pytest.raises(CheckpointError, match="RNG state restore mismatch"):
        adapter.restore(state, {})
    (event,) = records("checkpoint_rng_restored")
    assert not event["matched"]
    assert not records("checkpoint_adapter_restored")


def test_concurrent_codec_telemetry_keeps_all_events():
    def roundtrip(index):
        expected = torch.tensor([index])
        actual = ContinuationCodec.loads(ContinuationCodec.dumps(expected))
        assert torch.equal(actual, expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(capture_device_telemetry(roundtrip), range(16)))
    assert len(records("checkpoint_tensor_captured")) == 16
    restored = records("checkpoint_tensor_restored")
    assert len(restored) == 16 and all(record["matched"] for record in restored)
