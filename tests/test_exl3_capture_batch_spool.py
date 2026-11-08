from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from gptqmodel.utils.exl3_capture_batch_spool import (
    EXL3CaptureBatchSpool,
    EXL3CaptureBatchSpoolError,
)


def _open(tmp_path, *, checkpoint_interval=1):
    return EXL3CaptureBatchSpool(
        tmp_path / "spool",
        layer_index=4,
        subset_index=0,
        subset_total=2,
        expected_batches=3,
        payload_contract="test.exl3-capture-payload-v1",
        phase="gate-up",
        module_names=["model.layers.4.mlp.experts.0.gate_proj"],
        provenance={"plan_sha256": "a" * 64},
        ownership={"0": "cuda:0"},
        checkpoint_interval=checkpoint_interval,
    )


def test_capture_batch_spool_round_trip_and_resume(tmp_path) -> None:
    spool = _open(tmp_path)
    spool.commit(
        1,
        tensors={
            "router_input": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "top_indices": torch.tensor([[0], [1], [0]]),
        },
        metadata={"corpus_batch_sha256": "b" * 64},
    )
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset({1})
    tensors, metadata = resumed.load(1)
    assert torch.equal(tensors["router_input"], torch.arange(12).reshape(3, 4))
    assert metadata == {"corpus_batch_sha256": "b" * 64}


def test_capture_batch_spool_checkpoints_bounded_groups(tmp_path) -> None:
    spool = _open(tmp_path, checkpoint_interval=2)
    spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    assert spool.committed_indices == frozenset()
    assert spool.pending_indices == frozenset({0})

    spool.commit(
        1,
        tensors={"router_input": torch.ones(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 1},
    )
    assert spool.committed_indices == frozenset({0, 1})
    assert spool.pending_indices == frozenset()

    # The final short group is forced durable even though it does not fill the
    # configured interval.
    spool.commit(
        2,
        tensors={"router_input": torch.full((2, 4), 2, dtype=torch.bfloat16)},
        metadata={"batch": 2},
    )
    assert spool.committed_indices == frozenset({0, 1, 2})
    resumed = _open(tmp_path, checkpoint_interval=3)
    assert resumed.committed_indices == frozenset({0, 1, 2})


def test_capture_batch_spool_discards_uncheckpointed_group(tmp_path) -> None:
    spool = _open(tmp_path, checkpoint_interval=3)
    spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    assert spool.pending_indices == frozenset({0})

    resumed = _open(tmp_path, checkpoint_interval=3)
    assert resumed.committed_indices == frozenset()
    assert resumed.pending_indices == frozenset()
    assert not (resumed.directory / "batch-000000000.safetensors").exists()


def test_capture_batch_spool_drops_interrupted_temporary(tmp_path) -> None:
    spool = _open(tmp_path)
    partial = spool.directory / ".batch-000000000.safetensors.dead.tmp"
    partial.write_bytes(b"partial")
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset()
    assert not partial.exists()


def test_capture_batch_spool_drops_unmanifested_renamed_payload(tmp_path) -> None:
    spool = _open(tmp_path)
    partial = spool.directory / "batch-000000000.safetensors"
    save_safetensors_file({"router_input": torch.zeros(1, 4)}, partial)
    resumed = _open(tmp_path)
    assert resumed.committed_indices == frozenset()
    assert not partial.exists()


def test_capture_batch_spool_rejects_payload_tampering(tmp_path) -> None:
    spool = _open(tmp_path)
    record = spool.commit(
        0,
        tensors={"router_input": torch.zeros(2, 4, dtype=torch.bfloat16)},
        metadata={"batch": 0},
    )
    (spool.directory / record["file"]).write_bytes(b"corrupt")
    with pytest.raises(EXL3CaptureBatchSpoolError, match="payload differs"):
        _open(tmp_path)


def test_capture_batch_spool_rejects_identity_drift(tmp_path) -> None:
    spool = _open(tmp_path)
    progress_path = spool.directory / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["capture_key"]["phase"] = "down"
    progress_path.write_text(json.dumps(progress))
    with pytest.raises(EXL3CaptureBatchSpoolError, match="identity differs"):
        _open(tmp_path)


def test_capture_batch_spool_payload_contract_changes_storage_identity(tmp_path) -> None:
    original = _open(tmp_path)
    replacement = EXL3CaptureBatchSpool(
        tmp_path / "spool",
        layer_index=4,
        subset_index=0,
        subset_total=2,
        expected_batches=3,
        payload_contract="test.exl3-capture-payload-v2",
        phase="gate-up",
        module_names=["model.layers.4.mlp.experts.0.gate_proj"],
        provenance={"plan_sha256": "a" * 64},
        ownership={"0": "cuda:0"},
    )

    assert replacement.directory != original.directory
    assert replacement.committed_indices == frozenset()
    assert not original.directory.exists()
