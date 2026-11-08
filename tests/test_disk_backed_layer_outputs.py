from __future__ import annotations

import gc
import json

import pytest
import torch

from gptqmodel.looper.input_cache import (
    DiskBackedLayerOutputSequence,
    DiskBackedLayerOutputWriter,
    TensorLifetimeDiagnostic,
)


def test_disk_backed_layer_outputs_are_sharded_and_random_access(tmp_path):
    root = tmp_path / "activations"
    provenance = {"plan_sha256": "a" * 64, "replay_batch_size": 4}
    writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=1,
        expected_batches=5,
        provenance=provenance,
        shard_batches=2,
    )
    expected = {
        index: torch.full((1 if index == 4 else 4, 3, 2), index, dtype=torch.bfloat16)
        for index in range(5)
    }
    # Exercise out-of-order writes as used by two-device forward replay.
    for index in (0, 2, 1, 4, 3):
        writer.put(index, [expected[index]])
    sequence = writer.finalize()

    assert len(sequence) == 5
    assert sequence.row_counts == [4, 4, 4, 4, 1]
    for index in range(5):
        assert torch.equal(sequence[index][0], expected[index])

    manifest = json.loads((root / "layer-000001" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["hash_algorithm"] == "xxh3-128"
    assert len(manifest["shards"]) == 3
    reopened = DiskBackedLayerOutputSequence.open(root / "layer-000001")
    assert torch.equal(reopened[3][0], expected[3])


def test_disk_backed_layer_outputs_report_issued_tensor_lifetime(tmp_path):
    root = tmp_path / "activations"
    writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=1,
        provenance={"plan": "lifetime"},
    )
    writer.put(0, [torch.zeros(2, 4)])
    sequence = writer.finalize()

    retained = sequence[0]
    live = sequence.lifetime_diagnostic()
    assert live["issued"] == 1
    assert live["alive_tensors"] == 1
    assert live["alive_storage_bytes"] == 2 * 4 * 4

    del retained
    gc.collect()
    released = sequence.lifetime_diagnostic()
    assert released == {
        "issued": 1,
        "alive_tensors": 0,
        "alive_storage_bytes": 0,
    }


def test_tensor_lifetime_diagnostic_does_not_retain_nested_inputs():
    layer_inputs = [[torch.zeros(2, 4)], [torch.ones(3, 4)]]
    diagnostic = TensorLifetimeDiagnostic(layer_inputs)

    live = diagnostic.lifetime_diagnostic()
    assert live["issued"] == 2
    assert live["alive_tensors"] == 2
    assert live["alive_storage_bytes"] == (2 * 4 + 3 * 4) * 4

    del layer_inputs
    gc.collect()
    released = diagnostic.lifetime_diagnostic()
    assert released == {
        "issued": 2,
        "alive_tensors": 0,
        "alive_storage_bytes": 0,
    }


def test_disk_backed_layer_outputs_resume_complete_shards(tmp_path):
    root = tmp_path / "activations"
    provenance = {"plan_sha256": "b" * 64}
    first = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=3,
        provenance=provenance,
        shard_batches=2,
    )
    first.put(0, [torch.zeros(2, 4)])
    first.put(1, [torch.ones(2, 4)])
    first.abort()

    resumed = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=3,
        provenance=provenance,
        shard_batches=2,
    )
    assert resumed.committed_indices == frozenset({0, 1})
    # The complete first shard is content-validated and need not be rewritten.
    resumed.put(0, [torch.full((2, 4), 99.0)])
    resumed.put(1, [torch.full((2, 4), 99.0)])
    resumed.put(2, [torch.full((1, 4), 2.0)])
    sequence = resumed.finalize()
    assert resumed.committed_indices == frozenset({0, 1, 2})
    assert torch.equal(sequence[0][0], torch.zeros(2, 4))
    assert torch.equal(sequence[1][0], torch.ones(2, 4))
    assert torch.equal(sequence[2][0], torch.full((1, 4), 2.0))


def test_disk_backed_layer_outputs_reject_provenance_drift(tmp_path):
    root = tmp_path / "activations"
    writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=1,
        provenance={"plan": "one"},
    )
    writer.put(0, [torch.zeros(1, 2)])
    writer.abort()

    with pytest.raises(ValueError, match="identity differs"):
        DiskBackedLayerOutputWriter(
            root,
            layer_index=0,
            expected_batches=1,
            provenance={"plan": "two"},
        )


def test_disk_backed_layer_outputs_retry_post_commit_finalizer(tmp_path):
    root = tmp_path / "activations"
    calls = []

    def fail_once(sequence):
        calls.append(sequence.manifest["manifest_sha256"])
        if len(calls) == 1:
            raise RuntimeError("injected finalizer interruption")

    writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=1,
        provenance={"plan": "callback"},
        on_finalize=fail_once,
    )
    writer.put(0, [torch.zeros(1, 2)])
    with pytest.raises(RuntimeError, match="injected"):
        writer.finalize()
    assert (root / "layer-000000" / "manifest.json").is_file()

    resumed = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=1,
        provenance={"plan": "callback"},
        on_finalize=fail_once,
    )
    sequence = resumed.finalize()
    assert len(sequence) == 1
    assert calls == [sequence.manifest["manifest_sha256"]] * 2
    assert resumed.finalize() is sequence
    assert len(calls) == 2
