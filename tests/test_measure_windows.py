# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Sliding-window coverage: every transition is scored exactly once.

PPL must weight each next-token transition in a row once. Overlapping windows
exist to give the model context, not to score the shared tokens again, and an
adjacent window must keep one token of context so the transition at the seam is
not lost. A position-local model makes that a sharp regression: its logits do
not depend on the window, so changing window or stride cannot change PPL.
"""

import math

import pytest
import torch

from gptqmodel.measure import MeasureConfig, PPLConfig, TokenWindow, build_token_rows
from gptqmodel.measure.runner import measure_models


ROW = [0, 0, 1, 1, 0, 0, 1, 0]
CASES = [
    (8, 8),
    (4, 2),
    (4, 0),
    (4, 4),
    (4, 1),
    (4, 100),
    (2, 1),
]


class _PositionLocal(torch.nn.Module):
    """Logits depend only on the token at each position, never on the window."""

    def forward(self, input_ids, attention_mask):
        return torch.stack([torch.zeros_like(input_ids), input_ids * -2.0], dim=-1)


def _oracle_ppl(row):
    nll = [
        math.log1p(math.exp(-2 * previous)) + 2 * previous * target
        for previous, target in zip(row, row[1:])
    ]
    return math.exp(sum(nll) / len(nll))


def _windows(row, window, stride):
    return build_token_rows(None, [row], window=window, stride=stride, max_rows=1024)


@pytest.mark.parametrize("window,stride", CASES)
def test_window_and_stride_do_not_change_ppl(window, stride):
    model = _PositionLocal()
    report = measure_models(
        model,
        model,
        _windows(ROW, window, stride),
        MeasureConfig(
            metrics=PPLConfig(window=window, stride=stride),
            dataset=[ROW],
            reference=model,
            max_batches=64,
        ),
    )

    assert report.values["PPL"] == pytest.approx(_oracle_ppl(ROW), rel=1e-5)


def test_plain_token_rows_still_score_every_transition():
    model = _PositionLocal()
    report = measure_models(
        model,
        model,
        [torch.tensor(ROW).unsqueeze(0)],
        MeasureConfig(metrics=PPLConfig(window=4, stride=2), dataset=[ROW], reference=model),
    )

    assert report.values["PPL"] == pytest.approx(_oracle_ppl(ROW), rel=1e-5)


def test_model_measure_matches_the_oracle_for_every_window():
    from gptqmodel.models.base import BaseQModel

    class _Wrapper:
        measure = BaseQModel.measure
        _measurement_reference = BaseQModel._measurement_reference

    model = _Wrapper()
    model.model = _PositionLocal()

    for window, stride in CASES:
        report = model.measure(
            MeasureConfig(
                metrics=PPLConfig(window=window, stride=stride),
                dataset=[ROW],
                reference=model.model,
                max_batches=64,
            )
        )
        assert report.values["PPL"] == pytest.approx(_oracle_ppl(ROW), rel=1e-5), (window, stride)


@pytest.mark.parametrize("length", [2, 3, 5, 8, 17])
@pytest.mark.parametrize("window", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("stride", [0, 1, 2, 3, 4, 8, 16, 100])
def test_masks_partition_the_row_exactly_once(length, window, stride):
    row = list(range(100, 100 + length))
    windows = _windows(row, window, stride)

    covered = []
    for window_row in windows:
        assert isinstance(window_row, TokenWindow)
        tokens = window_row.input_ids.reshape(-1).tolist()
        mask = window_row.score_mask.reshape(-1)
        assert len(tokens) <= window
        assert len(tokens) >= 2
        # The window is a contiguous slice of the row, and only targets are marked.
        assert tokens == row[tokens[0] - 100:tokens[0] - 100 + len(tokens)]
        assert not bool(mask[0])
        # Newly scored targets form a contiguous suffix: context may lead, but a
        # scored target is never followed by an unscored one.
        assert bool((mask[1:].int().diff() >= 0).all())
        for position in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
            covered.append(tokens[position])

    # A target token at index t means the transition (t - 1) -> t.
    assert sorted(covered) == list(range(101, 100 + length))


def test_token_window_rejects_a_mismatched_mask():
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="score_mask"):
        TokenWindow(input_ids=ids, score_mask=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="input_ids"):
        TokenWindow(input_ids=torch.zeros(4), score_mask=torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="input_ids"):
        TokenWindow(input_ids=torch.zeros(2, 4), score_mask=torch.zeros(2, 4, dtype=torch.bool))


def test_build_token_rows_rejects_a_degenerate_window():
    with pytest.raises(ValueError, match="window"):
        build_token_rows(None, [ROW], window=1)
