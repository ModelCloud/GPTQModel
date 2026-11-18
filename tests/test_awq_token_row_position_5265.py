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


def test_single_row_quota_takes_the_batch_centre():
    """The issue's case: 64 short documents, budget 64, one row each."""

    quotas, rows = _rows([40] * 64, 64)

    assert quotas == [1] * 64
    assert {tuple(indices) for indices in rows.values()} == {(20,)}


def test_budget_below_batch_count_takes_centres():
    quotas, rows = _rows([10, 10, 10], 2)

    assert quotas == [1, 0, 1]
    assert rows == {0: [5], 2: [5]}


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
def test_no_single_row_batch_contributes_position_zero(lengths):
    for max_tokens in range(1, sum(lengths) + 1):
        quotas, rows = _rows(lengths, max_tokens)
        for batch, row_indices in rows.items():
            if quotas[batch] != 1 or lengths[batch] <= 1:
                continue
            assert row_indices != [0], (lengths, max_tokens, batch)
