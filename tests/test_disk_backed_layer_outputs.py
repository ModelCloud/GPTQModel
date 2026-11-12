from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

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


def test_out_of_order_batches_spill_payloads_until_each_shard_is_complete(tmp_path):
    writer = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=64,
        provenance={"plan": "out-of-order"}, shard_batches=2,
    )
    for index in range(0, 64, 2):
        writer.put(index, [torch.full((1, 1024), index, dtype=torch.float32)])

    # Every shard is incomplete. Its activation must live on disk, even when
    # many different shards have received their first out-of-order batch.
    assert len(writer._pending) == 32
    assert all(
        not isinstance(batch, torch.Tensor)
        for shard in writer._pending.values() for batch in shard.values()
    )
    assert len(list(writer.temporary.glob("pending-*.safetensors"))) == 32

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(writer.put, index, [torch.full((1, 1024), index, dtype=torch.float32)])
            for index in range(1, 64, 2)
        ]
        for future in futures:
            future.result()
    assert not writer._pending
    assert not list(writer.temporary.glob("pending-*.safetensors"))
    sequence = writer.finalize()
    for index in (0, 1, 31, 32, 63):
        assert torch.equal(sequence[index][0], torch.full((1, 1024), index))


def test_concurrent_append_reserves_distinct_batch_indices(tmp_path):
    writer = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=32,
        provenance={"plan": "concurrent-append"}, shard_batches=1,
    )
    batches = [[torch.full((1, 2), index)] for index in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer.append, batch) for batch in batches]
        for future in futures:
            future.result()
    sequence = writer.finalize()
    assert sorted(int(sequence[index][0][0, 0]) for index in range(32)) == list(range(32))


@pytest.mark.parametrize("abort", [False, True])
def test_uncommitted_batch_spills_are_discarded_on_resume(tmp_path, abort):
    provenance = {"plan": "interrupted-partial-shard"}
    writer = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=2,
        provenance=provenance, shard_batches=2,
    )
    writer.put(0, [torch.full((1, 2), 7.0)])
    assert list(writer.temporary.glob("pending-*.safetensors"))
    if abort:
        writer.abort()
        assert not list(writer.temporary.glob("pending-*.safetensors"))

    # A restarted writer trusts only complete shards in progress.json.
    resumed = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=2,
        provenance=provenance, shard_batches=2,
    )
    assert not list(resumed.temporary.glob("pending-*.safetensors"))
    resumed.put(0, [torch.zeros(1, 2)])
    resumed.put(1, [torch.ones(1, 2)])
    sequence = resumed.finalize()
    assert torch.equal(sequence[0][0], torch.zeros(1, 2))
    assert torch.equal(sequence[1][0], torch.ones(1, 2))


def test_changed_uncommitted_spill_cannot_enter_committed_shard(tmp_path):
    writer = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=2,
        provenance={"plan": "spill-integrity"}, shard_batches=2,
    )
    writer.put(0, [torch.zeros(1, 2)])
    staged = next(writer.temporary.glob("pending-*.safetensors"))
    save_file({"hidden": torch.full((1, 2), 7.0)}, staged)
    with pytest.raises(ValueError, match="pending batch hash differs"):
        writer.put(1, [torch.ones(1, 2)])
    assert not writer.committed_indices


@pytest.mark.parametrize("shard_batches", [1, 2])
def test_open_sequence_rechecks_shard_hash_on_every_read(tmp_path, shard_batches):
    writer = DiskBackedLayerOutputWriter(
        tmp_path, layer_index=0, expected_batches=shard_batches,
        provenance={"plan": "read-after-replace"}, shard_batches=shard_batches,
    )
    for index in range(shard_batches):
        writer.put(index, [torch.full((1, 2), index, dtype=torch.float32)])
    finalized = writer.finalize()  # finalize skips the redundant open-time hash.
    reopened = DiskBackedLayerOutputSequence.open(writer.destination)
    assert torch.equal(finalized[0][0], torch.zeros(1, 2))
    assert torch.equal(reopened[0][0], torch.zeros(1, 2))

    contents = (
        {"hidden": torch.full((1, 2), 7.0)}
        if shard_batches == 1 else {
            f"batch_{index:09d}": torch.full((1, 2), 7.0)
            for index in range(shard_batches)
        }
    )
    save_file(contents, writer.destination / "outputs-000000.safetensors")
    for sequence in (finalized, reopened):
        with pytest.raises(ValueError, match="shard hash differs"):
            _ = sequence[shard_batches - 1]


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


def test_disk_backed_layer_outputs_resume_after_finalize_rename_crash(tmp_path, monkeypatch):
    root = tmp_path / "activations"
    provenance = {"plan": "rename-crash"}
    writer = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=2,
        provenance=provenance,
        shard_batches=2,
    )
    writer.put(0, [torch.zeros(1, 2)])
    writer.put(1, [torch.ones(1, 2)])

    real_replace = os.replace

    def crash(src, dst):
        if Path(dst) == writer.destination:
            raise OSError("injected rename crash")
        return real_replace(src, dst)

    monkeypatch.setattr("gptqmodel.looper.input_cache.os.replace", crash)
    with pytest.raises(OSError, match="injected rename crash"):
        writer.finalize()
    assert not writer.destination.exists()

    monkeypatch.setattr("gptqmodel.looper.input_cache.os.replace", real_replace)
    resumed = DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=2,
        provenance=provenance,
        shard_batches=2,
    )
    # The completed shards survive the failed rename and need no rewrite.
    assert resumed.committed_indices == frozenset({0, 1})
    sequence = resumed.finalize()
    assert torch.equal(sequence[0][0], torch.zeros(1, 2))
    assert torch.equal(sequence[1][0], torch.ones(1, 2))
    assert not (writer.destination / "progress.json").exists()
