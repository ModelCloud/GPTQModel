# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""A routed-expert reservoir must keep the final sample's rows (#5384).

The per-expert budget grows with the captures seen so far:
``min(raw_tokens, max(largest_capture, captures))``. Evicting to that budget on
every merge discarded early rows that a later, larger capture would have
admitted: with captures of 64, 64 and 512 rows the final cap is 512, but the
38 lowest-priority rows of the first two captures were already gone. The
reservoir now retains evicted rows while the budget can still grow and keeps
the final budget's lowest priorities at materialization.
"""

import torch

from gptqmodel.looper.awq_processor import _AWQLayerState
from tests.test_awq_reservoir_incremental_5233 import _processor


def _capture_rows(batch, length, hidden=4):
    """Rows encode their ``(batch, row)`` identity in the first two columns."""

    rows = torch.zeros(1, length, hidden)
    rows[0, :, 0] = float(batch)
    rows[0, :, 1] = torch.arange(length)
    return rows


def _priority_oracle(processor, lengths, final_cap):
    """Return the ``final_cap`` lowest-priority ``(batch, row)`` of the stream."""

    priorities = torch.cat(
        [processor._reservoir_priorities(batch, length) for batch, length in enumerate(lengths)]
    )
    keep = torch.topk(priorities, final_cap, largest=False).indices
    offsets = []
    offset = 0
    for length in lengths:
        offsets.append(offset)
        offset += length
    oracle = set()
    for index in keep.tolist():
        for batch in range(len(lengths) - 1, -1, -1):
            if index >= offsets[batch]:
                oracle.add((batch, index - offsets[batch]))
                break
    return oracle


def test_growing_capture_budget_keeps_the_lowest_final_priorities():
    hidden = 4
    lengths = [64, 64, 512]
    processor = _processor(max_tokens=None, hidden=hidden)
    module_name = "mlp.experts.0.up_proj"

    for batch, length in enumerate(lengths):
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            module_name, _capture_rows(batch, length, hidden)
        )

    entry = processor.tasks[module_name]
    final_cap = processor._reservoir_budget_cap(None, None, entry)
    assert final_cap == 512
    # The capture buffer itself stays at the current budget; the extra rows
    # wait in the candidate pool until the budget is final.
    assert int(entry["reservoir_used"]) == final_cap

    oracle = _priority_oracle(processor, lengths, final_cap)
    assert len(oracle) == final_cap

    state = _AWQLayerState(modules={module_name: torch.nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    retained = features[module_name]
    provenance = entry["provenance"]

    assert retained.shape == (1, final_cap, hidden)
    assert set(provenance) == oracle
    # Every packed row still belongs to the provenance entry that names it.
    for packed_row, (source, row) in enumerate(provenance):
        assert torch.equal(
            retained[0, packed_row],
            _capture_rows(source, lengths[source], hidden)[0, row],
        )


def test_losing_capture_can_enter_once_the_budget_grows():
    """A capture that loses today is retained for a larger later budget."""

    hidden = 2
    processor = _processor(max_tokens=None, hidden=hidden)
    entry = {
        "inputs": [],
        "batch_indices": [],
        "reservoir_candidates": [],
        "reservoir_candidate_cap": None,
    }

    def merge(priorities, source, cap):
        processor._merge_reservoir_rows(
            entry,
            torch.full((len(priorities), hidden), float(source)),
            torch.tensor(priorities, dtype=torch.float64),
            source_index=source,
            budget_cap=cap,
        )

    # Two kept rows, then a capture whose priorities all lose to them: the
    # short-circuit keeps the buffer untouched and pools the capture.
    merge([0.10, 0.20], 0, 2)
    merge([0.90, 0.95], 1, 2)
    assert int(entry["reservoir_used"]) == 2
    assert sum(int(part[0].shape[0]) for part in entry["reservoir_candidates"]) == 2

    # The budget grows; two worse rows enter, so the losing capture now has
    # the 4 lowest priorities of the stream and must be materialized.
    merge([0.98, 0.99], 2, 4)

    processor._materialize_reservoir_provenance(entry)
    assert entry["provenance"] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert entry["inputs"][0][0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_fixed_budget_does_not_retain_candidates():
    """A configured budget never grows, so its reservoir needs no pool."""

    hidden = 2
    processor = _processor(max_tokens=8, hidden=hidden)
    module_name = "mlp.experts.2.down_proj"

    for batch in range(4):
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            module_name, _capture_rows(batch, 4, hidden)
        )

    entry = processor.tasks[module_name]
    assert "reservoir_candidates" not in entry
    assert int(entry["reservoir_used"]) == 8


def test_candidate_pool_is_pruned_to_the_plan_budget():
    """The pool never keeps more rows than the final budget can use."""

    hidden = 2
    processor = _processor(max_tokens=None, hidden=hidden)
    entry = {"reservoir_candidates": [], "reservoir_candidate_cap": 4}

    for step in range(20):
        processor._remember_reservoir_candidates(
            entry,
            torch.full((1, hidden), float(step)),
            torch.tensor([float(step)], dtype=torch.float64),
            torch.tensor([step], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    retained = torch.cat([part[1] for part in entry["reservoir_candidates"]])
    total = int(retained.numel())
    assert total <= 2 * 4 + 1
    # Only the lowest four priorities can still reach a budget of four.
    assert torch.equal(
        torch.sort(retained).values[:4], torch.arange(4, dtype=torch.float64)
    )
