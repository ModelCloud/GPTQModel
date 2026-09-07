# SPDX-License-Identifier: Apache-2.0
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.looper.checkpoint_store import CheckpointConfig
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper import checkpoint as checkpoint_module
from gptqmodel.looper import gptq_checkpoint
from gptqmodel.looper.gptq_checkpoint import QuantizationCheckpointAdapter


def adapter_for(model, tmp_path):
    adapter = QuantizationCheckpointAdapter()
    adapter.model = SimpleNamespace(model=model, quantize_config=None)
    adapter.execution = SimpleNamespace(
        execution_state_dict=dict, load_execution_state_dict=lambda state: None
    )
    adapter.processors = ()
    adapter.processor = SimpleNamespace(lock=threading.Lock())
    adapter.shared_state = {"placement": torch.device("cpu")}
    adapter.offload = tmp_path
    adapter._artifacts = {}
    return adapter


def test_auto_checkpoint_path_uses_offload_root(tmp_path):
    offload_root = tmp_path / "offload"
    offload_root.mkdir()
    captured = {}

    class DummyExtension:
        def __init__(self, config, adapter):
            captured["path"] = config.path
            self.store = SimpleNamespace(root=Path(config.path))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    model = SimpleNamespace(
        quantize_config=SimpleNamespace(
            offload_to_disk=True,
            offload_to_disk_path=str(offload_root),
        )
    )
    with patch.object(checkpoint_module, "CheckpointExtension", DummyExtension):
        with gptq_checkpoint.checkpoint_session(CheckpointConfig(), model):
            pass
    assert captured["path"] == str(offload_root)
    assert model.quantize_config.offload_to_disk_path != str(offload_root)


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
def test_dense_scaling_state_and_aliases_roundtrip(tmp_path, device):
    if (
        device.startswith("cuda")
        and torch.cuda.device_count() <= torch.device(device).index
    ):
        pytest.skip(f"requires {device}")
    model = torch.nn.ModuleDict(
        {"a": torch.nn.Linear(4, 4), "b": torch.nn.Linear(4, 4)}
    ).to(device)
    model.b.weight = model.a.weight
    model.a.register_buffer("scale", torch.tensor([0.0, -0.0, 2.0], device=device))
    adapter = adapter_for(model, tmp_path)
    state, artifacts = adapter.capture()
    state = ContinuationCodec.loads(ContinuationCodec.dumps(state))
    expected = state["dense"]["a"]["weight"].clone()
    with torch.no_grad():
        model.a.weight.zero_()
        model.a.scale.fill_(3)
    adapter.shared_state.clear()
    adapter.restore(state, artifacts)
    assert model.a.weight is model.b.weight
    assert model.a.weight.device == torch.device(device)
    assert torch.equal(model.a.weight, expected)
    assert torch.equal(
        model.a.scale.view(torch.uint8), state["dense"]["a"]["scale"].view(torch.uint8)
    )
    assert adapter.shared_state == {"placement": torch.device("cpu")}


@pytest.mark.parametrize("owner", [None, 5, ["a"], ["a", 7], ["missing", "weight"]])
def test_bad_dense_alias_rejected_before_weight_mutation(tmp_path, owner):
    model = torch.nn.ModuleDict(
        {"a": torch.nn.Linear(4, 4), "b": torch.nn.Linear(4, 4)}
    )
    adapter = adapter_for(model, tmp_path)
    state, artifacts = adapter.capture()
    state["dense_aliases"] = {"b.weight": owner}
    original = model.a.weight
    with pytest.raises(CheckpointError, match="alias"):
        adapter.restore(state, artifacts)
    assert model.a.weight is original
