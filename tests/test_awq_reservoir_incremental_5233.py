# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The routed-expert reservoir must merge incrementally (#5233).

The dynamic AWQ capture path used to concatenate, re-sort and rebuild the whole
kept buffer plus a Python provenance list on every capture, all under the
processor-wide lock. That made AWQ quantize 3-6x slower on MoE models. The
buffer is now a fixed set of slots that receives only the surviving new rows,
and provenance is materialized once per layer.
"""

import threading
from types import SimpleNamespace

import torch

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState, _TokenRowPlan
from gptqmodel.quantization.config import CalibrationConfig


def _processor(max_tokens, hidden):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._reservoir_locks = {}
    processor._capture_host_cache = {}
    processor._capture_sampled_cache = {}
    processor._token_row_plans = {}
    processor._capture_host_batch = None
    processor._module_forward_kwargs = {}
    processor._feature_task_names = set()
    processor._feature_stats = {}
    processor._awq_feature_kwargs = {}
    processor._scale_feature_by_module = {}
    processor._nsamples_total = 1
    cache = SimpleNamespace(
        layer_inputs=None,
        attention_masks=[],
        position_ids=[],
        layer_input_kwargs=[],
    )
    processor.inputs_cache = cache
    processor.calibration = CalibrationConfig(exclude_attention_sinks=False)
    gptq_model = SimpleNamespace()
    gptq_model.awq_input_feature_aggregation = lambda name: {
        "mode": "token_rows",
        "max_tokens": max_tokens,
    }
    processor.gptq_model = gptq_model
    return processor


def test_reservoir_reuses_one_buffer_instead_of_rebuilding_it():
    hidden = 4
    processor = _processor(max_tokens=64, hidden=hidden)
    module_name = "mlp.experts.0.gate_proj"

    for batch in range(20):
        processor._set_current_batch_index(batch)
        processor._record_input_feature(module_name, torch.randn(1, 5, hidden))

        entry = processor.tasks[module_name]
        buffer = entry["reservoir_buffer"]
        if batch == 0:
            first_buffer = buffer
        assert buffer is first_buffer
        # ``inputs`` is a view of the live buffer: no copy of the kept rows.
        captured = entry["inputs"][0]
        assert captured.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        # Provenance stays in integer tensors until replay.
        assert "provenance" not in entry

    entry = processor.tasks[module_name]
    assert int(entry["reservoir_used"]) == 64
    assert entry["inputs"][0].shape == (1, 64, hidden)


def test_reservoir_keeps_every_row_aligned_with_its_provenance():
    hidden = 3
    processor = _processor(max_tokens=40, hidden=hidden)
    module_name = "mlp.experts.1.gate_proj"

    lengths = [7, 3, 9, 5, 11, 6]
    for batch, length in enumerate(lengths):
        processor._set_current_batch_index(batch)
        # Every row of a batch carries its batch id so provenance can be
        # verified against the retained values.
        processor._record_input_feature(
            module_name, torch.full((1, length, hidden), float(batch))
        )

    state = _AWQLayerState(modules={module_name: torch.nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    retained = features[module_name]
    entry = processor.tasks[module_name]

    assert retained.shape == (1, 40, hidden)
    provenance = entry["provenance"]
    assert len(provenance) == 40
    for packed_row, (source_index, row_index) in enumerate(provenance):
        assert 0 <= source_index < len(lengths)
        assert 0 <= row_index < lengths[source_index]
        assert torch.all(retained[0, packed_row] == float(source_index))

    # Every batch that contributed rows keeps them, and the cap is exact.
    assert {source for source, _row in provenance} == set(range(len(lengths)))


def test_reservoir_handles_a_batch_larger_than_the_cap():
    hidden = 2
    processor = _processor(max_tokens=8, hidden=hidden)
    module_name = "mlp.experts.2.down_proj"

    processor._set_current_batch_index(0)
    processor._record_input_feature(module_name, torch.full((1, 20, hidden), 0.0))
    processor._set_current_batch_index(1)
    processor._record_input_feature(module_name, torch.full((1, 20, hidden), 1.0))

    state = _AWQLayerState(modules={module_name: torch.nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    entry = processor.tasks[module_name]

    assert features[module_name].shape == (1, 8, hidden)
    provenance = entry["provenance"]
    assert len(provenance) == 8
    for packed_row, (source_index, row_index) in enumerate(provenance):
        assert 0 <= row_index < 20
        assert torch.all(features[module_name][0, packed_row] == float(source_index))


def test_reservoir_keeps_rows_captured_before_it_took_over():
    """A plan capture followed by a dynamic one must not drop the plan rows."""

    hidden = 2
    processor = _processor(max_tokens=16, hidden=hidden)
    module_name = "self_attn.q_proj"
    # A precomputed-plan capture already stored three rows from source 0.
    processor.tasks[module_name] = {
        "inputs": [torch.full((1, 3, hidden), 7.0)],
        "batch_indices": [0],
        "batch_row_counts": [3],
        "provenance": [(0, 0), (0, 1), (0, 2)],
        "raw_tokens": 3,
    }

    processor._set_current_batch_index(1)
    processor._record_input_feature(module_name, torch.full((1, 5, hidden), 9.0))

    entry = processor.tasks[module_name]
    processor._materialize_reservoir_provenance(entry)
    rows = entry["inputs"][0][0]
    assert rows.shape[0] == 8
    assert int((rows == 7.0).sum()) == 3 * hidden
    assert int((rows == 9.0).sum()) == 5 * hidden
    assert {source for source, _row in entry["provenance"]} == {0, 1}


def test_plan_capture_after_reservoir_keeps_its_rows():
    """A plan capture after the reservoir exists must merge, not vanish (#5264).

    Batches are processed longest first, so the short batch that matches the
    precomputed plan often arrives after the routed expert's reservoir already
    exists. The plan path used to append to ``inputs``/``provenance``, which the
    next reservoir merge overwrote, so the whole batch was silently dropped.
    """

    processor = _processor(max_tokens=64, hidden=2)
    lengths = [5, 3, 4, 6]
    plan = _TokenRowPlan(
        lengths=lengths,
        budget=18,
        max_tokens=64,
        quotas=list(lengths),
        local_indices={batch: torch.arange(length) for batch, length in enumerate(lengths)},
    )
    processor._token_row_plan_for_module = lambda *args, **kwargs: plan
    module_name = "mlp.experts.3.gate_proj"

    # Batch 0 keeps 2 of 5 rows through the reservoir, batch 1 matches the plan
    # exactly (3 of 3), then two more reservoir captures follow.
    for batch, length in [(0, 2), (1, 3), (2, 1), (3, 2)]:
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            module_name, torch.full((1, length, 2), float(batch))
        )

    features = processor._layer_input_features(
        _AWQLayerState(modules={module_name: torch.nn.Linear(2, 2)})
    )
    rows = features[module_name][0, :, 0].tolist()
    assert rows == [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0]


def test_plan_capture_after_reservoir_keeps_its_local_row_indices():
    """A partial plan quota merged after the reservoir keeps its own rows (#5264).

    The reservoir stores each surviving row's index inside its calibration
    batch; a plan capture that keeps only ``local_indices`` must pass those
    indices instead of renumbering the sampled rows from zero.
    """

    processor = _processor(max_tokens=64, hidden=2)
    lengths = [5, 3, 4, 6]
    plan = _TokenRowPlan(
        lengths=lengths,
        budget=7,
        max_tokens=64,
        quotas=[2, 2, 1, 2],
        local_indices={
            0: torch.tensor([0, 3]),
            1: torch.tensor([0, 2]),
            2: torch.tensor([1]),
            3: torch.tensor([0, 5]),
        },
    )
    processor._token_row_plan_for_module = lambda *args, **kwargs: plan
    module_name = "mlp.experts.5.down_proj"

    # Batch 0 falls out of the plan (2 of 5 rows) and creates the reservoir;
    # batch 1 matches the plan with a partial quota of 2 of its 3 rows.
    for batch, length in [(0, 2), (1, 3)]:
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            module_name, torch.full((1, length, 2), float(batch))
        )

    entry = processor.tasks[module_name]
    processor._materialize_reservoir_provenance(entry)
    assert entry["provenance"] == [(0, 0), (0, 1), (1, 0), (1, 2)]
    assert entry["inputs"][0][0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
