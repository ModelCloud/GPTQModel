# SPDX-License-Identifier: Apache-2.0
import json
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save

from gptqmodel.looper.checkpoint import CheckpointConfig, CheckpointExtension
from gptqmodel.looper.checkpoint_devices import checkpoint_device_topology
from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.extension import LoopExtensions, LoopPlan, LoopStep
from gptqmodel.looper.module_looper import ModuleLooper


def mock_cuda(monkeypatch, identities):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(identities))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            uuid=identities[index],
            name="Test GPU",
            major=9,
            minor=0,
        ),
    )


def pools(devices):
    return {
        "quantization": devices,
        "dense": devices,
        "moe": devices,
        "forward": devices,
    }


class Adapter:
    def capture(self):
        return {"value": torch.ones(1)}, {}

    def restore(self, state, artifacts):
        pytest.fail("topology mismatch must be rejected before restoration")


@pytest.mark.parametrize(
    "change", ["fewer", "more", "reordered", "replaced", "pool", "cpu"]
)
def test_changed_topology_rejected_before_tensor_restore(tmp_path, monkeypatch, change):
    mock_cuda(monkeypatch, ["GPU-A", "GPU-B"])
    original_pools = pools(["cuda:0", "cuda:1"])
    original = checkpoint_device_topology(original_pools)
    assert original["gpu_count"] == 2
    plan = LoopPlan((LoopStep("layer", 0, "model.layers.0"),))
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension:
        extension.prepare(plan, {"device_topology": original})
        LoopExtensions([extension]).publish(plan.steps[0])
    changed_pools = pools(["cuda:0", "cuda:1"])
    if change == "fewer":
        mock_cuda(monkeypatch, ["GPU-A"])
        changed_pools = pools(["cuda:0"])
    elif change == "more":
        mock_cuda(monkeypatch, ["GPU-A", "GPU-B", "GPU-C"])
        changed_pools = pools(["cuda:0", "cuda:1", "cuda:2"])
    elif change == "reordered":
        mock_cuda(monkeypatch, ["GPU-B", "GPU-A"])
    elif change == "replaced":
        mock_cuda(monkeypatch, ["GPU-A", "GPU-C"])
    elif change == "pool":
        changed_pools["dense"] = ["cuda:1", "cuda:0"]
    else:
        changed_pools = pools(["cpu"])
    changed = checkpoint_device_topology(changed_pools)

    def fail_decode(data):
        pytest.fail("topology must be checked before decoding/moving tensors")

    monkeypatch.setattr(ContinuationCodec, "loads", fail_decode)
    with (
        CheckpointExtension(CheckpointConfig(tmp_path), Adapter()) as extension,
        pytest.raises(
            CheckpointError,
            match="device_topology",
        ),
    ):
        extension.prepare(plan, {"device_topology": changed})


def test_gpu_identity_requires_uuid(monkeypatch):
    mock_cuda(monkeypatch, [None])
    with pytest.raises(CheckpointError, match="UUID"):
        checkpoint_device_topology(pools(["cuda:0"]))


def scheduler():
    looper = ModuleLooper.__new__(ModuleLooper)
    looper._quant_devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    looper._quant_device_lock = threading.Lock()
    looper._quant_device_rr = 0
    looper._module_device_map = {}
    return looper


def assign(looper, name):
    module = SimpleNamespace(full_name=name, name=name, state={})
    return looper._assign_quant_device_for_module(module, torch.device("cpu"))


def test_scheduler_resumes_same_device_assignment():
    uninterrupted = scheduler()
    assert [str(assign(uninterrupted, f"completed.{i}")) for i in range(3)] == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
    ]
    uninterrupted._module_device_map["completed.0"] = torch.device("cpu")
    state = uninterrupted.execution_state_dict()
    resumed = scheduler()
    resumed.load_execution_state_dict(state)
    assert resumed.execution_state_dict() == state
    assert (
        assign(resumed, "next")
        == assign(uninterrupted, "next")
        == torch.device("cuda:1")
    )
    assert assign(resumed, "completed.0") == torch.device("cpu")


def test_execution_pools_record_forward_only_gpus(monkeypatch):
    import gptqmodel.looper.module_looper as module

    looper = scheduler()
    looper._primary_quant_device = torch.device("cuda:0")
    looper._dense_quant_devices = looper._moe_quant_devices = looper._quant_devices
    looper.gptq_model = SimpleNamespace(
        quantize_config=SimpleNamespace(auto_forward_data_parallel=False)
    )
    monkeypatch.setattr(
        module,
        "select_forward_devices",
        lambda primary: [torch.device(f"cuda:{index}") for index in range(3)],
    )
    mock_cuda(monkeypatch, ["GPU-A", "GPU-B", "GPU-C"])
    topology = checkpoint_device_topology(looper.execution_device_pools())
    assert topology["pools"]["forward"] == ["cuda:0"] and topology["gpu_count"] == 2
    looper.gptq_model.quantize_config.auto_forward_data_parallel = True
    topology = checkpoint_device_topology(looper.execution_device_pools())
    assert (
        topology["pools"]["forward"] == ["cuda:0", "cuda:1", "cuda:2"]
        and topology["gpu_count"] == 3
    )


def test_scheduler_rejects_missing_device_without_mutation():
    looper = scheduler()
    prior = looper.execution_state_dict()
    with pytest.raises(ValueError, match="unavailable device"):
        looper.load_execution_state_dict(
            {"version": 1, "next_device": 3, "module_devices": {"a": "cuda:2"}}
        )
    assert looper.execution_state_dict() == prior


def test_codec_uses_saved_logical_index(monkeypatch):
    # Simulate a cuda:1 tensor on a CPU test runner, then inspect the requested
    # transfer destination. This is a contract test, not multi-GPU E2E evidence.
    data = save(
        {"0": torch.ones(2)},
        metadata={
            "continuation": json.dumps(
                {
                    "version": ContinuationCodec.VERSION,
                    "tree": ["tensor", {"name": "0", "device": "cuda:1"}],
                }
            )
        },
    )
    destinations = []

    def record(tensor, *, device):
        destinations.append(str(device))
        return tensor

    monkeypatch.setattr(torch.Tensor, "to", record)
    with pytest.raises(ValueError, match="device placement restore mismatch"):
        ContinuationCodec.loads(data)
    assert destinations == ["cuda:1"]


@pytest.mark.parametrize("count", [1, 2])
def test_real_cuda_tensor_and_metadata_placement(count):
    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} physical CUDA devices")
    state = {
        index: {
            "tensor": torch.arange(6, device=f"cuda:{index}"),
            "device": torch.device(f"cuda:{index}"),
        }
        for index in range(count)
    }
    restored = ContinuationCodec.loads(ContinuationCodec.dumps(state))
    for index in range(count):
        assert restored[index]["tensor"].device == state[index]["device"]
        assert restored[index]["device"] == state[index]["device"]
        assert torch.equal(restored[index]["tensor"], state[index]["tensor"])


@pytest.mark.parametrize("family", ["llama", "qwen3_moe"])
@pytest.mark.parametrize(
    "mode",
    [
        "term",
        "int",
        "kill-before",
        "kill-after",
        "error",
        "kill-hessian",
        "kill-hessian-early",
    ],
)
def test_real_multigpu_partial_hessian_recovery(tmp_path, monkeypatch, family, mode):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two physical CUDA devices for quantization/recovery")
    # Preserve visible physical identity if the caller already selected GPUs.
    # The subprocess uses the same visible set, never a CPU simulation.
    import os

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    from test_checkpoint_quantization import test_subprocess_recovery

    test_subprocess_recovery(tmp_path, family, mode, device="cuda:0", required_gpus=2)
