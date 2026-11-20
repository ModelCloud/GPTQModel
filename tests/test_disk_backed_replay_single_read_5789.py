# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5789: schedule disk-backed replay without re-reading shards.

Every ``DiskBackedLayerOutputSequence.__getitem__`` maps the safetensors file
and hashes the whole shard. Multi-device placement used to call it once per
pending batch (twice in ``balanced`` mode) before dispatch read the payload
again. The validated manifest already proves presence, and disk-backed tensors
are always issued on CPU, so scheduling must use that metadata and leave one
verified payload read per dispatched batch.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.looper import input_cache
from gptqmodel.looper.forward_executor import ForwardExecutor


BATCH_COUNT = 8
BATCH_SHAPE = (1, 1024, 512)  # 2 MiB per float32 shard, 16 MiB total


class _InlinePool:
    """Run each submission inline so the test needs no worker threads."""

    def submit(self, device, function, *args, **kwargs):
        del device
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future

    submit_serial = submit


class _IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("one", torch.ones(()))

    def forward(self, x, use_cache=False):
        del use_cache
        return x * self.one


def _collect_row_counts(batches):
    declared = getattr(batches, "row_counts", None)
    if declared is not None:
        return list(declared)
    return [int(batch[0].shape[0]) if batch else 0 for batch in batches]


def _looper(policy):
    return SimpleNamespace(
        support_batch_quantize=False,
        _resolve_batch_total=lambda _count, batches: len(batches),
        _collect_row_counts=_collect_row_counts,
        _batch_row_count=lambda batch: int(batch[0].shape[0]) if batch else 0,
        gptq_model=SimpleNamespace(
            quantize_config=SimpleNamespace(
                calibration_data_device=policy,
                compute_device_filter=None,
            ),
            prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"],
            prepare_layer_replay_keep_mask=lambda **kwargs: kwargs["keep_mask"],
        ),
    )


def _write_disk_cache(root: Path):
    writer = input_cache.DiskBackedLayerOutputWriter(
        root,
        layer_index=0,
        expected_batches=BATCH_COUNT,
        provenance={"fixture": "disk-replay-single-read"},
        shard_batches=1,
    )
    expected = []
    for index in range(BATCH_COUNT):
        payload = torch.full(BATCH_SHAPE, index, dtype=torch.float32)
        expected.append(payload)
        writer.put(index, [payload])
    return writer.finalize(), expected


def _run_parallel(executor, layer_inputs, processor=None):
    if processor is None:
        processor = SimpleNamespace(
            num_batches=len(layer_inputs),
            _set_current_batch_index=lambda _index: None,
        )
    return executor.run_parallel(
        module=_IdentityLayer(),
        processor=processor,
        layer_inputs=layer_inputs,
        layer_input_kwargs=[{} for _ in range(len(layer_inputs))],
        position_ids=[],
        attention_masks=[None] * len(layer_inputs),
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=1,
        need_outputs=True,
        reuse_kv=False,
        devices=[torch.device("cpu:0"), torch.device("cpu:1")],
        apply_moe_config=False,
        clone_module_for_devices_fn=lambda module, devices, **_: dict.fromkeys(devices, module),
        device_thread_pool=_InlinePool(),
    )


@pytest.mark.parametrize("policy", [None, "balanced"])
def test_disk_backed_placement_hashes_each_shard_exactly_once(tmp_path, policy):
    sequence, expected = _write_disk_cache(tmp_path)
    shard_paths = [sequence.root / shard["path"] for shard in sequence.manifest["shards"]]
    shard_bytes = sum(path.stat().st_size for path in shard_paths)
    assert len(shard_paths) == BATCH_COUNT

    hash_file = input_cache._xxh3_128_file
    open_file = input_cache.safe_open
    hashed = []
    opened = []

    def record_hash(path):
        hashed.append(Path(path).name)
        return hash_file(path)

    def record_open(*args, **kwargs):
        opened.append(Path(args[0]).name)
        return open_file(*args, **kwargs)

    with patch.object(input_cache, "_xxh3_128_file", record_hash), patch.object(input_cache, "safe_open", record_open):
        outputs = _run_parallel(ForwardExecutor(_looper(policy)), sequence)

    # One integrity hash and one payload open per dispatched batch, not once
    # for placement plus once (or twice) more before the worker runs.
    assert sorted(hashed) == sorted(path.name for path in shard_paths)
    assert sorted(opened) == sorted(path.name for path in shard_paths)
    assert shard_bytes // 2**20 == 16

    # Scheduling is metadata-only; results keep batch order and payload bytes.
    assert len(outputs) == BATCH_COUNT
    for index, batch_output in enumerate(outputs):
        assert len(batch_output) == 1
        assert torch.equal(batch_output[0], expected[index])


def test_disk_backed_dispatch_still_detects_a_corrupt_shard(tmp_path):
    sequence, _ = _write_disk_cache(tmp_path)
    shard = sequence.root / sequence.manifest["shards"][3]["path"]
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 0xFF
    shard.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="shard hash differs"):
        _run_parallel(ForwardExecutor(_looper(None)), sequence)


def test_disk_backed_scheduling_skips_committed_batches(tmp_path):
    sequence, expected = _write_disk_cache(tmp_path)
    shard_paths = [sequence.root / shard["path"] for shard in sequence.manifest["shards"]]
    committed = frozenset({0, 2})
    hash_file = input_cache._xxh3_128_file
    hashed = []

    def record_hash(path):
        hashed.append(Path(path).name)
        return hash_file(path)

    class _ResumedProcessor:
        num_batches = BATCH_COUNT

        def _set_current_batch_index(self, index):
            del index

        def forward_committed_batch_indices(self, *, layer_index):
            del layer_index
            return committed

    with patch.object(input_cache, "_xxh3_128_file", record_hash):
        outputs = _run_parallel(ForwardExecutor(_looper(None)), sequence, _ResumedProcessor())

    # A resumed layer never reads the batches its output spool already holds.
    assert sorted(hashed) == sorted(path.name for index, path in enumerate(shard_paths) if index not in committed)
    assert len(outputs) == BATCH_COUNT
    assert outputs[0] == [] and outputs[2] == []
    for index in range(BATCH_COUNT):
        if index not in committed:
            assert torch.equal(outputs[index][0], expected[index])


@pytest.mark.parametrize("policy", [None, "balanced"])
def test_list_backed_placement_skips_empty_batch_slots(policy):
    layer_inputs = [[] if index == 2 else [torch.full((1, 4), float(index))] for index in range(4)]

    outputs = _run_parallel(ForwardExecutor(_looper(policy)), layer_inputs)

    assert len(outputs) == 4
    assert outputs[2] == []
    for index in (0, 1, 3):
        assert torch.equal(outputs[index][0], layer_inputs[index][0])
