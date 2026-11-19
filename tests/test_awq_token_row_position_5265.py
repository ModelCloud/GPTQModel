# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A batch that contributes one AWQ token row must not contribute position 0.

``_compute_token_row_quotas`` pinned row 0 for every single-row quota. Position
0 holds the BOS/attention-sink activation, which is identical for every
document, so a calibration set with many short rows (``batch_size=1``) collapsed
the whole AWQ sample onto one vector: Llama-3.2-1B PPL 21.8 -> 91.7 (#5265).
Single-row quotas now keep the centre of the batch.
"""

from __future__ import annotations

import pytest

from gptqmodel.looper.awq_processor import AWQProcessor


def _rows(lengths, max_tokens):
    quotas, local = AWQProcessor._compute_token_row_quotas(list(lengths), max_tokens)
    return quotas, {batch: rows.tolist() for batch, rows in local.items()}


def test_single_row_quota_keeps_a_bounded_sink_share():
    """The issue's case: 64 short documents, budget 64, one row each.

    #5265 measured all 64 rows at position 0 (Llama PPL 91.7); #5310 measured
    no rows at position 0 (Qwen PPL 28.0, worse than RTN). One in eight keeps
    the sink row, the rest keep the batch centre.
    """

    quotas, rows = _rows([40] * 64, 64)

    assert quotas == [1] * 64
    assert [batch for batch, rows_ in rows.items() if rows_ == [0]] == [
        0, 8, 16, 24, 32, 40, 48, 56
    ]
    assert [batch for batch, rows_ in rows.items() if rows_ == [20]] == [
        batch for batch in range(64) if batch % 8
    ]


def test_budget_below_batch_count_mixes_sink_and_centre():
    quotas, rows = _rows([10, 10, 10], 2)

    assert quotas == [1, 0, 1]
    # Ordinal 1 keeps row 0; ordinal 2 keeps the batch centre.
    assert rows == {0: [0], 2: [5]}


def test_a_single_batch_still_keeps_its_first_row():
    """The pre-#5265 behaviour is preserved when there is nothing to spread."""

    quotas, rows = _rows([40], 1)

    assert quotas == [1]
    assert rows == {0: [0]}


def test_multi_row_quotas_keep_the_even_spacing():
    """Two or more rows per batch keep the existing endpoint spacing."""

    quotas, rows = _rows([0, 7, 3], 4)

    assert quotas == [0, 2, 2]
    assert rows == {1: [0, 6], 2: [0, 2]}


def test_full_length_quota_keeps_every_row():
    quotas, rows = _rows([4], 4)

    assert quotas == [4]
    assert rows == {0: [0, 1, 2, 3]}


@pytest.mark.parametrize(
    "lengths",
    [
        [40] * 64,
        [7, 3, 0],
        [5, 5, 5],
        [100, 3, 3],
        [1, 1, 1],
    ],
)
def test_single_row_quotas_bound_the_sink_share(lengths):
    """Neither extreme: 1%-50% of the single-row picks are position 0."""

    for max_tokens in range(1, sum(lengths) + 1):
        quotas, rows = _rows(lengths, max_tokens)
        single = [
            batch
            for batch, row_indices in rows.items()
            if quotas[batch] == 1 and lengths[batch] > 1
        ]
        if len(single) < 8:
            continue
        sinks = sum(1 for batch in single if rows[batch] == [0])
        share = sinks / len(single)
        assert 0.01 <= share <= 0.5, (lengths, max_tokens, share)
