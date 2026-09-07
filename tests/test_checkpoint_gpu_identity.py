# SPDX-License-Identifier: Apache-2.0
import copy
import subprocess
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper import checkpoint_devices
from gptqmodel.looper.checkpoint import CheckpointConfig, CheckpointExtension
from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.extension import LoopExtensions, LoopPlan, LoopStep
from gptqmodel.utils.device_telemetry import (
    clear_device_telemetry_records,
    device_telemetry_scope,
    get_device_telemetry_records,
)


def test_serial_query_maps_uuid_not_physical_index(monkeypatch):
    def query(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=uuid,serial",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 5 and kwargs["check"]
        return SimpleNamespace(
            stdout="GPU-b, SERIAL-B\nGPU-a, SERIAL-A\nGPU-c, [N/A]\n"
        )

    monkeypatch.setattr(checkpoint_devices.subprocess, "run", query)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(
            uuid=["a", "b"][i], name="RTX 4090", major=8, minor=9
        ),
    )
    topology = checkpoint_devices.checkpoint_device_topology(
        {"quantization": ["cuda:0", "cuda:1"]}
    )
    assert [gpu["serial"] for gpu in topology["visible_cuda"]] == [
        "SERIAL-A",
        "SERIAL-B",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired("nvidia-smi", 5),
        subprocess.CalledProcessError(1, "nvidia-smi"),
    ],
)
def test_unavailable_serial_query_falls_back_to_uuid(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(checkpoint_devices.subprocess, "run", fail)
    assert checkpoint_devices._gpu_serials_by_uuid() == {}


@pytest.mark.parametrize("invalid", [0, 1, "false", "true", None])
def test_strict_device_check_requires_explicit_bool(tmp_path, invalid):
    with pytest.raises(ValueError, match="must be a boolean"):
        CheckpointConfig(tmp_path, strict_device_check=invalid)


def identity():
    return {
        "model": "test",
        "device_topology": {
            "version": 2,
            "gpu_count": 2,
            "pools": {"quantization": ["cuda:0", "cuda:1"]},
            "visible_cuda": [
                {
                    "index": i,
                    "uuid": f"GPU-{i}",
                    "serial": f"SERIAL-{i}",
                    "name": "RTX 4090",
                    "capability": [8, 9],
                }
                for i in range(2)
            ],
        },
    }


class Adapter:
    def capture(self):
        return {"value": torch.ones(1)}, {}

    def restore(self, state, artifacts):
        assert torch.equal(state["value"], torch.ones(1))


PLAN = LoopPlan((LoopStep("layer", 0, "model.layers.0"),))


def publish(root, saved):
    with CheckpointExtension(CheckpointConfig(root), Adapter()) as extension:
        extension.prepare(PLAN, saved)
        LoopExtensions([extension]).publish(PLAN.steps[0])


@pytest.mark.parametrize(
    "change", ["serial", "uuid", "serial_disappeared", "reordered"]
)
def test_strict_physical_identity_rejected_before_decode(tmp_path, monkeypatch, change):
    saved = identity()
    publish(tmp_path, saved)
    actual = copy.deepcopy(saved)
    devices = actual["device_topology"]["visible_cuda"]
    if change == "reordered":
        devices[0]["serial"], devices[1]["serial"] = (
            devices[1]["serial"],
            devices[0]["serial"],
        )
    else:
        devices[0]["serial" if change == "serial_disappeared" else change] = (
            None if change == "serial_disappeared" else "replacement"
        )

    def fail_decode(data):
        pytest.fail("physical identity must gate before tensor decode")

    monkeypatch.setattr(ContinuationCodec, "loads", fail_decode)
    assert CheckpointConfig(tmp_path).strict_device_check
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(CheckpointError, match="device_topology"),
    ):
        extension.prepare(PLAN, actual)


def test_override_allows_physical_identity_change_and_reports_bypass(
    tmp_path, monkeypatch, caplog
):
    saved = identity()
    publish(tmp_path, saved)
    actual = copy.deepcopy(saved)
    actual["device_topology"]["visible_cuda"][0].update(
        uuid="replacement", serial="replacement"
    )
    clear_device_telemetry_records()
    with (
        device_telemetry_scope(True),
        CheckpointExtension(
            CheckpointConfig(tmp_path, strict_device_check=False), Adapter()
        ) as extension,
    ):
        assert extension.prepare(PLAN, actual) == 1
    (event,) = [
        record
        for record in get_device_telemetry_records()
        if record["event"] == "checkpoint_topology_validated"
    ]
    assert (
        event["matched"]
        and not event["strict_device_check"]
        and not event["physical_identity_matched"]
    )
    assert event["expected_topology"] != event["actual_topology"]
    assert "explicitly disabled" in caplog.text
    assert (
        saved == identity()
    )  # Comparison must not strip identity from the saved state.
    clear_device_telemetry_records()


@pytest.mark.parametrize(
    "change", ["count", "pool", "index", "capability", "name", "model", "version"]
)
def test_override_still_requires_exact_logical_topology_and_model(tmp_path, change):
    saved = identity()
    publish(tmp_path, saved)
    actual = copy.deepcopy(saved)
    topology = actual["device_topology"]
    if change == "count":
        topology["gpu_count"] = 1
        topology["visible_cuda"].pop()
    elif change == "pool":
        topology["pools"]["quantization"].reverse()
    elif change == "index":
        topology["visible_cuda"][0]["index"] = 1
    elif change == "capability":
        topology["visible_cuda"][0]["capability"] = [9, 0]
    elif change == "name":
        topology["visible_cuda"][0]["name"] = "different GPU model"
    elif change == "version":
        topology["version"] = 1
    else:
        actual["model"] = "different model"
    with (
        CheckpointExtension(
            CheckpointConfig(tmp_path, strict_device_check=False), Adapter()
        ) as extension,
        pytest.raises(CheckpointError, match="incompatible checkpoint identity"),
    ):
        extension.prepare(PLAN, actual)


def test_uuid_gate_remains_when_serial_unavailable(tmp_path):
    saved = identity()
    for gpu in saved["device_topology"]["visible_cuda"]:
        gpu["serial"] = None
    publish(tmp_path, saved)
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        assert extension.prepare(PLAN, saved) == 1
    changed = copy.deepcopy(saved)
    changed["device_topology"]["visible_cuda"][0]["uuid"] = "replacement"
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(CheckpointError, match="device_topology"),
    ):
        extension.prepare(PLAN, changed)
