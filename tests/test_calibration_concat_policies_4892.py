# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``concat_size`` packing must run before the per-row calibration policies (#4892).

Packing exists to combine short samples into full windows, but the per-row
``min_length`` filter ran first and discarded every document of
``<= min_length`` tokens (397 of 400 rows on a corpus of short utterances).
The per-row position-budget trim also ran first, so a document longer than
``max_position_embeddings`` lost everything past the budget even though each
packed window was far below it.

The policies now apply to the packed windows, and ``concat_size`` is validated
against the model's position budget up front.
"""

from types import SimpleNamespace

import pytest

from gptqmodel.utils.calibration import (
    _prepare_tensor_kwargs_dataset,
    prepare_calibration_dataset,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = None
    padding_side = "right"
    truncation_side = "right"

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(token)) for token in ids)


def _qmodel(*, max_positions=128):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=max_positions),
        ),
    )


def _rows(*lengths):
    """One pre-tokenized item per length, with distinct token ids."""

    dataset = []
    next_id = 1
    for length in lengths:
        ids = list(range(next_id, next_id + length))
        next_id += length
        dataset.append({"input_ids": [ids], "attention_mask": [[1] * length]})
    return dataset


def _real_tokens(prepared):
    return sum(int(batch["attention_mask"].sum()) for batch in prepared)


def test_concat_keeps_documents_shorter_than_min_length():
    """Ten 3-token documents: 30 real tokens, not zero.

    ``min_length`` is 10, so every source row is shorter than the filter. The
    filter must not run before packing, or the whole corpus disappears.
    """

    prepared = prepare_calibration_dataset(
        _qmodel(),
        _rows(*([3] * 10)),
        calibration_dataset_concat_size=16,
        calibration_data_min_length=10,
    )

    assert len(prepared) == 2
    assert _real_tokens(prepared) == 30
    assert prepared[0]["attention_mask"].tolist() == [[1] * 16]
    assert int(prepared[1]["attention_mask"].sum()) == 14


def test_concat_does_not_trim_a_document_before_packing():
    """A 200-token document under a 64-token budget keeps all 200 tokens.

    Every packed window is 32 tokens, so nothing can exceed the budget. The
    per-row trim used to cut the document to 64 tokens before windowing.
    """

    prepared = prepare_calibration_dataset(
        _qmodel(max_positions=64),
        _rows(200),
        calibration_dataset_concat_size=32,
        calibration_data_min_length=0,
    )

    assert len(prepared) == 7
    assert _real_tokens(prepared) == 200
    assert all(batch["attention_mask"].shape[1] == 32 for batch in prepared)


def test_concat_drops_a_tail_window_below_min_length():
    """Five 4-token rows: one full 16-token window, then a 4-token tail.

    The tail is at or below ``min_length``, so it carries no usable activation
    statistics and is dropped.
    """

    prepared = prepare_calibration_dataset(
        _qmodel(),
        _rows(*([4] * 5)),
        calibration_dataset_concat_size=16,
        calibration_data_min_length=10,
    )

    assert len(prepared) == 1
    assert _real_tokens(prepared) == 16


def test_concat_keeps_a_tail_window_above_min_length():
    """Same shape, but the tail has 14 real tokens, so it is kept."""

    prepared = prepare_calibration_dataset(
        _qmodel(),
        _rows(*([4] * 5)) + _rows(10),
        calibration_dataset_concat_size=16,
        calibration_data_min_length=10,
    )

    # 20 + 10 = 30 real tokens: one full window and one 14-token tail.
    assert len(prepared) == 2
    assert _real_tokens(prepared) == 30


def test_concat_size_above_the_position_budget_is_refused():
    with pytest.raises(ValueError, match="exceeds max_position_embeddings=64"):
        prepare_calibration_dataset(
            _qmodel(max_positions=64),
            _rows(8),
            calibration_dataset_concat_size=65,
            calibration_data_min_length=0,
        )


def test_concat_size_equal_to_the_position_budget_is_accepted():
    prepared = prepare_calibration_dataset(
        _qmodel(max_positions=64),
        _rows(8),
        calibration_dataset_concat_size=64,
        calibration_data_min_length=0,
    )

    assert len(prepared) == 1
    assert prepared[0]["attention_mask"].tolist() == [[1] * 8 + [0] * 56]


def test_non_concat_path_keeps_the_per_row_policies():
    """Without ``concat_size`` a short row is still dropped and a long one trimmed."""

    prepared = prepare_calibration_dataset(
        _qmodel(max_positions=64),
        _rows(3) + _rows(200),
        calibration_dataset_concat_size=None,
        calibration_data_min_length=10,
    )

    assert len(prepared) == 1
    assert _real_tokens(prepared) == 64


def test_concat_size_errors_name_the_public_config_field():
    """Every user-facing concat error names ``CalibrationConfig.concat_size``.

    The removed ``calibration_dataset_concat_size`` quantize keyword raises
    ``TypeError``, so a message that names it sends the user to a dead end
    (#5121).
    """

    with pytest.raises(ValueError, match=r"CalibrationConfig\.concat_size=65 exceeds"):
        prepare_calibration_dataset(
            _qmodel(max_positions=64),
            _rows(8),
            calibration_dataset_concat_size=65,
            calibration_data_min_length=0,
        )

    with pytest.raises(ValueError, match=r"`CalibrationConfig\.concat_size` is specified"):
        prepare_calibration_dataset(
            SimpleNamespace(
                tokenizer=None,
                support_batch_quantize=True,
                model=SimpleNamespace(
                    config=SimpleNamespace(max_position_embeddings=128),
                ),
            ),
            _rows(8),
            calibration_dataset_concat_size=8,
            calibration_data_min_length=0,
        )

    with pytest.raises(ValueError, match=r"`CalibrationConfig\.concat_size` must be larger"):
        prepare_calibration_dataset(
            _qmodel(),
            _rows(2),
            calibration_dataset_concat_size=8,
            calibration_data_min_length=10,
        )

    with pytest.raises(ValueError, match=r"`CalibrationConfig\.concat_size` is token-based"):
        _prepare_tensor_kwargs_dataset(
            [],
            calibration_dataset_concat_size=8,
            batch_size=1,
            log=SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None),
        )
