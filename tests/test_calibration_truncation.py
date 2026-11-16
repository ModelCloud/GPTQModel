# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Truncation-side rules for overlong calibration rows.

``prepare_calibration_dataset`` trims a row to the model's position budget.
It used to cut from the left whenever ``tokenizer.padding_side == "left"``,
ignoring ``tokenizer.truncation_side`` (HF default: right), so every trimmed
sample lost its BOS / attention-sink token and its activations stopped
matching inference. Padding direction and truncation direction are independent
tokenizer settings.
"""

from types import SimpleNamespace

import pytest

from gptqmodel.utils.calibration import prepare_calibration_dataset


BOS = 1000
MAX_POSITIONS = 32


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, padding_side: str, truncation_side: str):
        self.padding_side = padding_side
        self.truncation_side = truncation_side

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(
            "<|begin_of_text|>" if int(token) == BOS else f"t{int(token)}"
            for token in ids
        )


def _prepare(padding_side: str, truncation_side: str):
    tokenizer = _Tokenizer(padding_side, truncation_side)
    qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=MAX_POSITIONS)
        ),
    )
    row = [BOS] + list(range(1, 40))
    dataset = [{"input_ids": row, "attention_mask": [1] * len(row)}]
    prepared = prepare_calibration_dataset(
        qmodel, dataset, batch_size=1, calibration_data_min_length=1
    )
    return row, prepared[0]["input_ids"][0].tolist(), prepared[0]["attention_mask"][0].tolist()


def test_left_padding_keeps_the_bos_token():
    """The issue's case: left padding must not front-truncate the row."""
    row, ids, mask = _prepare(padding_side="left", truncation_side="right")

    assert len(ids) == MAX_POSITIONS
    assert ids == row[:MAX_POSITIONS]
    assert ids[0] == BOS
    assert mask == [1] * MAX_POSITIONS


def test_left_and_right_padding_trim_identically():
    row, left_ids, left_mask = _prepare("left", "right")
    _, right_ids, right_mask = _prepare("right", "right")

    assert left_ids == right_ids == row[:MAX_POSITIONS]
    assert left_mask == right_mask == [1] * MAX_POSITIONS


def test_truncation_side_left_keeps_the_tail():
    """An explicit ``truncation_side="left"`` still trims from the front."""
    row, ids, mask = _prepare(padding_side="left", truncation_side="left")

    assert len(ids) == MAX_POSITIONS
    assert ids == row[-MAX_POSITIONS:]
    assert ids[0] != BOS
    assert mask == [1] * MAX_POSITIONS


def test_short_rows_are_not_trimmed():
    tokenizer = _Tokenizer("left", "right")
    qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=MAX_POSITIONS)
        ),
    )
    row = [BOS] + list(range(1, 10))
    prepared = prepare_calibration_dataset(
        qmodel,
        [{"input_ids": row, "attention_mask": [1] * len(row)}],
        batch_size=1,
        calibration_data_min_length=1,
    )
    assert prepared[0]["input_ids"][0].tolist() == row


def test_invalid_truncation_side_is_rejected():
    with pytest.raises(ValueError, match="truncation_side"):
        _prepare(padding_side="left", truncation_side="middle")


def _prepare_rows(rows, *, padding_side="left", truncation_side="right", max_positions=MAX_POSITIONS, batch=False, min_length=0):
    tokenizer = _Tokenizer(padding_side, truncation_side)
    qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=batch,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=max_positions)
        ),
    )
    dataset = [
        {
            "input_ids": [ids for ids, _ in rows],
            "attention_mask": [mask for _, mask in rows],
        }
    ]
    return prepare_calibration_dataset(
        qmodel, dataset, batch_size=1, calibration_data_min_length=min_length
    )


def test_left_padded_overlong_row_strips_padding_before_the_budget():
    """Issue #4238: the leading pads must not consume the position budget."""
    ids = [0, 0, 11, 12, 13, 14]
    mask = [0, 0, 1, 1, 1, 1]

    prepared = _prepare_rows([(ids, mask)], max_positions=4, batch=True)

    assert prepared[0]["input_ids"].tolist() == [[11, 12, 13, 14]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1, 1]]


def test_padding_at_least_as_long_as_the_budget_keeps_real_tokens():
    """The row used to survive as pure padding with an all-zero mask."""
    ids = [0, 0, 0, 0, 0, 0, 11, 12]
    mask = [0, 0, 0, 0, 0, 0, 1, 1]

    prepared = _prepare_rows([(ids, mask)], max_positions=4, batch=True)

    assert prepared[0]["input_ids"].tolist() == [[11, 12]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1]]


def test_all_padding_row_is_dropped_instead_of_kept_empty():
    ids = [0] * 8
    mask = [0] * 8

    # Same contract as the over-short filter: the row is dropped and counted,
    # so no empty row ever reaches the Hessian.
    prepared = _prepare_rows([(ids, mask)], max_positions=4, batch=True)

    assert prepared == []


def test_right_padding_is_stripped_too():
    ids = [11, 12, 13, 14, 15, 0, 0]
    mask = [1, 1, 1, 1, 1, 0, 0]

    prepared = _prepare_rows([(ids, mask)], padding_side="right", max_positions=3, batch=True)

    assert prepared[0]["input_ids"].tolist() == [[11, 12, 13]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_truncation_side_left_keeps_the_real_tail():
    ids = [0, 0, 11, 12, 13, 14]
    mask = [0, 0, 1, 1, 1, 1]

    prepared = _prepare_rows(
        [(ids, mask)], truncation_side="left", max_positions=2, batch=True
    )

    assert prepared[0]["input_ids"].tolist() == [[13, 14]]


def test_ragged_rows_after_stripping_are_re_padded_for_the_unbatched_path():
    """`support_batch_quantize=False` stacks an item, so rows must line up."""
    rows = [
        ([0, 0, 11, 12, 13, 14], [0, 0, 1, 1, 1, 1]),
        ([7, 8, 9, 10, 11, 12], [1, 1, 1, 1, 1, 1]),
    ]

    prepared = _prepare_rows(rows, max_positions=4, batch=False)

    assert prepared[0]["input_ids"].tolist() == [[11, 12, 13, 14], [7, 8, 9, 10]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]


def test_all_padding_row_below_the_budget_is_dropped_too():
    """The contract is "never emit an all-pad row", not "only when trimming"."""
    ids = [0, 0, 0]
    mask = [0, 0, 0]

    prepared = _prepare_rows([(ids, mask)], max_positions=MAX_POSITIONS, batch=True)

    assert prepared == []


def test_padded_and_real_rows_in_one_item_keep_only_the_real_row():
    rows = [
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
        ([5, 6, 7, 8, 9], [1, 1, 1, 1, 1]),
    ]

    prepared = _prepare_rows(rows, max_positions=3, batch=True)

    assert prepared[0]["input_ids"].tolist() == [[5, 6, 7]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]
