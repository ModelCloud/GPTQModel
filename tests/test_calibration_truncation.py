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
