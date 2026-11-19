# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert AWQ reservoir priorities must not depend on capture order.

``auto_forward_data_parallel`` replays a layer's batches on replicas, so the
capture hooks fire in whatever order the replicas finish. The reservoir used to
seed each capture's row priorities from its arrival slot, which paired a
batch's rows with another batch's priority stream and made the kept rows (and
the AWQ checkpoint) depend on thread timing (#5432).
"""

import random

import torch

from gptqmodel.looper.awq_processor import _TokenRowPlan
from tests.test_awq_reservoir_incremental_5233 import _processor


def _capture_rows(batch, length, hidden=4):
    """Rows encode their ``(batch, row)`` identity in the first two columns."""

    rows = torch.zeros(1, length, hidden)
    rows[0, :, 0] = float(batch)
    rows[0, :, 1] = torch.arange(length)
    return rows


def _kept_rows(processor, module_name):
    entry = processor.tasks[module_name]
    used = int(entry["reservoir_used"])
    rows = entry["inputs"][0][0, :used]
    return {(int(row[0]), int(row[1])) for row in rows}


def _record_in_order(order, lengths, *, max_tokens=40, hidden=4):
    processor = _processor(max_tokens=max_tokens, hidden=hidden)
    module_name = "mlp.experts.3.up_proj"
    for batch in order:
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            module_name, _capture_rows(batch, lengths[batch], hidden)
        )
    return processor, module_name


def test_kept_rows_do_not_depend_on_capture_arrival_order():
    lengths = [3, 7, 12, 5, 9, 11, 4, 8, 6, 10, 2, 5]
    expected_processor, module_name = _record_in_order(range(len(lengths)), lengths)
    expected = _kept_rows(expected_processor, module_name)

    assert expected, "the reservoir kept no rows"
    assert len(expected) <= 40

    generator = random.Random(0)
    for _ in range(10):
        order = list(range(len(lengths)))
        generator.shuffle(order)
        processor, module_name = _record_in_order(order, lengths)
        assert _kept_rows(processor, module_name) == expected, order


def test_priorities_are_seeded_by_batch_index():
    processor = _processor(max_tokens=40, hidden=2)
    assert processor._reservoir_priority_key(4, 17) == 17
    # A capture without a batch index keeps the previous slot-derived seed.
    assert processor._reservoir_priority_key(4, None) == 4

    by_batch = processor._reservoir_priorities(17, 8)
    # The same batch draws the same priorities from a different slot, and a
    # different batch does not reuse them.
    assert torch.equal(by_batch, processor._reservoir_priorities(17, 8))
    assert not torch.equal(by_batch, processor._reservoir_priorities(4, 8))


def test_dynamic_capture_seeds_from_the_batch_index_not_the_slot():
    processor = _processor(max_tokens=40, hidden=2)
    module_name = "mlp.experts.5.down_proj"
    seen = []
    real = processor._reservoir_priorities

    def record_seed(seed_key, count):
        seen.append((int(seed_key), int(count)))
        return real(seed_key, count)

    processor._reservoir_priorities = record_seed

    # Batch 7 is the first capture of this module, so its slot is 0.
    processor._set_current_batch_index(7)
    processor._record_input_feature(module_name, torch.zeros(1, 3, 2))
    processor._set_current_batch_index(9)
    processor._record_input_feature(module_name, torch.zeros(1, 4, 2))

    assert seen == [(7, 3), (9, 4)]


def test_plan_matched_capture_seeds_from_the_batch_index():
    processor = _processor(max_tokens=8, hidden=2)
    module_name = "model.layers.0.mlp.down_proj"
    processor._token_row_plan_for_module = lambda name, hidden, configured=None: _TokenRowPlan(
        lengths=[0, 0, 3],
        budget=3,
        max_tokens=8,
        quotas=[0, 0, 3],
        local_indices={2: torch.arange(3)},
    )
    seen = []
    real = processor._reservoir_priorities

    def record_seed(seed_key, count):
        seen.append((int(seed_key), int(count)))
        return real(seed_key, count)

    processor._reservoir_priorities = record_seed

    # A one-row capture first creates the reservoir (dynamic path, slot 0).
    processor._set_current_batch_index(0)
    processor._record_input_feature(module_name, torch.zeros(1, 1, 2))
    # Batch 2 then matches the plan and merges into the existing reservoir.
    processor._set_current_batch_index(2)
    processor._record_input_feature(module_name, torch.zeros(1, 3, 2))

    assert seen == [(0, 1), (2, 3)]


def test_seeded_plan_rows_keep_their_batch_index_priorities():
    processor = _processor(max_tokens=40, hidden=2)
    module_name = "model.layers.0.mlp.down_proj"
    processor._token_row_plan_for_module = lambda name, hidden, configured=None: _TokenRowPlan(
        lengths=[2],
        budget=2,
        max_tokens=40,
        quotas=[2],
        local_indices={0: torch.arange(2)},
    )

    # A plan-matched capture stays in ``inputs``; a later non-plan capture
    # creates the reservoir and seeds it from those rows.
    processor._set_current_batch_index(0)
    processor._record_input_feature(module_name, _capture_rows(0, 2, 2))
    processor._token_row_plan_for_module = lambda name, hidden, configured=None: None
    processor._set_current_batch_index(3)
    processor._record_input_feature(module_name, _capture_rows(3, 2, 2))

    entry = processor.tasks[module_name]
    used = int(entry["reservoir_used"])
    assert used == 4
    assert _kept_rows(processor, module_name) == {(0, 0), (0, 1), (3, 0), (3, 1)}

    # The seeded rows carry the draws of their own batch id, not of the slot
    # they were captured in (batch 3 landed in slot 1).
    expected = torch.cat(
        [processor._reservoir_priorities(0, 2), processor._reservoir_priorities(3, 2)]
    )
    assert torch.equal(entry["reservoir_priorities"][:used], expected)
