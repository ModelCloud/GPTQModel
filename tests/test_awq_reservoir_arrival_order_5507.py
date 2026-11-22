# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""The replayed expert sample must not depend on capture arrival order (#5507).

``auto_forward_data_parallel`` replays calibration batches on replicas, so the
capture hooks fire in the order the replicas finish. With the default budget
(no ``max_tokens``) a routed expert's reservoir grows with the captures seen so
far, so before #5553 every merge evicted rows down to the budget of that
moment: rows of early captures were gone before a later, larger capture raised
the cap. The kept ``(batch, row)`` set, and therefore the AWQ checkpoint,
changed run to run.

#5553 keeps the evicted rows as candidates while the budget can grow and, when
the layer materializes, keeps the final budget's lowest priorities over the
whole capture stream. This test pins the effective result: the rows the replay
feeds to AWQ are the same whatever order the captures arrive in.
"""

import random

import pytest
import torch

from gptqmodel.looper.awq_processor import _AWQLayerState
from tests.test_awq_reservoir_batch_index_priority_5432 import _capture_rows
from tests.test_awq_reservoir_incremental_5233 import _processor


MODULE = "mlp.experts.3.up_proj"


def _replay(order, lengths, hidden=4):
    """Capture the batches in ``order`` and return the rows AWQ would replay."""

    processor = _processor(max_tokens=None, hidden=hidden)
    for batch in order:
        processor._set_current_batch_index(batch)
        processor._record_input_feature(
            MODULE, _capture_rows(batch, lengths[batch], hidden)
        )
    state = _AWQLayerState(modules={MODULE: torch.nn.Linear(hidden, hidden)})
    retained = processor._layer_input_features(state)[MODULE]
    return {(int(row[0]), int(row[1])) for row in retained[0]}


@pytest.mark.parametrize("lengths", ([2, 2, 10], [5, 6, 3, 20, 4, 4, 2]))
def test_replayed_rows_do_not_depend_on_capture_arrival_order(lengths):
    batches = list(range(len(lengths)))
    expected = _replay(batches, lengths)

    assert expected, "the reservoir kept no rows"
    # The final budget is min(raw, max(largest_capture, captures)); a later
    # capture raises it, which is what made the pre-#5553 evictions lossy.
    assert len(expected) == min(
        sum(lengths), max(max(lengths), len(lengths))
    )

    generator = random.Random(0)
    orders = [batches[::-1]]
    for _ in range(20):
        order = batches[:]
        generator.shuffle(order)
        orders.append(order)
    for order in orders:
        assert _replay(order, lengths) == expected, order
