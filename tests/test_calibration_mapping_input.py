# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer outputs are mappings, not builtin dicts.

Issue #4052: a list of ``tokenizer(text)`` results (``BatchEncoding``, a
``UserDict``) fell through the ``isinstance(example, dict)`` check and was
rejected as an unsupported calibration example.

Issue #4193: passing one ``BatchEncoding`` (or another columnar mapping) as
the whole dataset calibrated on its key names instead of its rows.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from gptqmodel.utils.calibration import prepare_calibration_dataset


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, **kwargs):
        # One token per whitespace-separated word, enough to tell rows apart.
        length = len(text.split())
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[length]]),
                "attention_mask": torch.tensor([[1]]),
            }
        )


def _qmodel():
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=128),
        ),
    )


def test_batch_encoding_calibration_example_is_accepted():
    encoded = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
    )

    prepared = prepare_calibration_dataset(
        _qmodel(),
        [encoded],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5]]


def test_batch_encoding_without_attention_mask_is_accepted():
    encoded = BatchEncoding({"input_ids": torch.tensor([[6, 7, 8]])})

    prepared = prepare_calibration_dataset(
        _qmodel(),
        [encoded],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert prepared[0]["input_ids"].tolist() == [[6, 7, 8]]
    assert prepared[0]["attention_mask"].tolist() == [[True, True, True]]


def test_top_level_batch_encoding_is_unpacked_into_rows():
    encoded = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]]),
        }
    )

    prepared = prepare_calibration_dataset(
        _qmodel(),
        encoded,
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert [batch["input_ids"].tolist() for batch in prepared] == [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]
    assert [batch["attention_mask"].tolist() for batch in prepared] == [[[1, 1, 1, 1]], [[1, 0, 1, 1]]]


def test_top_level_columnar_dict_is_unpacked_into_rows():
    columns = {
        "input_ids": [[11, 12, 13], [21, 22, 23]],
        "attention_mask": [[1, 1, 1], [1, 0, 1]],
    }

    prepared = prepare_calibration_dataset(
        _qmodel(),
        columns,
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert [batch["input_ids"].tolist() for batch in prepared] == [[[11, 12, 13]], [[21, 22, 23]]]
    assert [batch["attention_mask"].tolist() for batch in prepared] == [[[1, 1, 1]], [[1, 0, 1]]]


def test_top_level_single_example_mapping_is_one_row():
    example = {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]}

    prepared = prepare_calibration_dataset(
        _qmodel(),
        example,
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == [[7, 8, 9]]


def test_top_level_text_column_is_tokenized_per_row():
    prepared = prepare_calibration_dataset(
        _qmodel(),
        {"text": ["one two three", "four five"]},
        batch_size=1,
        calibration_data_min_length=0,
    )

    assert [batch["input_ids"].tolist() for batch in prepared] == [[[3]], [[2]]]


def test_dataset_split_mapping_is_rejected_with_split_hint():
    datasets = pytest.importorskip("datasets")
    split = datasets.Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})
    split_mapping = datasets.DatasetDict({"train": split, "validation": split})

    with pytest.raises(ValueError, match="mapping of dataset splits") as excinfo:
        prepare_calibration_dataset(
            _qmodel(),
            split_mapping,
            calibration_data_min_length=1,
        )

    assert "train" in str(excinfo.value)
    assert "select one split" in str(excinfo.value)


def test_unknown_mapping_keys_are_rejected():
    with pytest.raises(ValueError, match="must provide `input_ids`, `text`, or `messages`"):
        prepare_calibration_dataset(
            _qmodel(),
            {"foo": [1, 2]},
            calibration_data_min_length=1,
        )


def test_columnar_mapping_with_mismatched_column_lengths_is_rejected():
    with pytest.raises(ValueError, match="column `attention_mask` has 1 rows, expected 2"):
        prepare_calibration_dataset(
            _qmodel(),
            {"input_ids": [[1, 2], [3, 4]], "attention_mask": [[1, 1]]},
            calibration_data_min_length=1,
        )


def _columnar_token_rows():
    return (
        [[11, 12, 13, 14], [21, 22, 23, 24, 25]],
        [[1, 1, 1, 1], [1, 1, 1, 1, 1]],
    )


def _assert_token_rows(prepared):
    ids, masks = _columnar_token_rows()
    assert [batch["input_ids"].tolist() for batch in prepared] == [[row] for row in ids]
    assert [batch["attention_mask"].tolist() for batch in prepared] == [[mask] for mask in masks]


@pytest.mark.parametrize("format", ["numpy", "pandas", "torch", None])
def test_dataset_formats_unpack_into_token_rows(format):
    """`Dataset.with_format(...)[:]` must reach the same rows as the dataset itself.

    ``numpy`` hands back a 1-D ``ndarray(dtype=object)`` per column, and
    ``pandas`` hands back a ``DataFrame``; neither was recognised as a columnar
    container, so ``numpy`` crashed in tensor conversion and ``pandas``
    calibrated on the column labels.
    """

    datasets = pytest.importorskip("datasets")
    ids, masks = _columnar_token_rows()
    dataset = datasets.Dataset.from_dict({"input_ids": ids, "attention_mask": masks})
    calibration_input = dataset.with_format(format)[:] if format else dataset

    _assert_token_rows(
        prepare_calibration_dataset(
            _qmodel(), calibration_input, batch_size=1, calibration_data_min_length=0
        )
    )


def test_object_dtype_column_of_arrays_is_one_row_per_element():
    np = pytest.importorskip("numpy")
    ids, masks = _columnar_token_rows()
    object_ids = np.empty(len(ids), dtype=object)
    object_ids[:] = [np.asarray(row, dtype=np.int64) for row in ids]
    object_masks = np.empty(len(masks), dtype=object)
    object_masks[:] = [np.asarray(row, dtype=np.int64) for row in masks]

    _assert_token_rows(
        prepare_calibration_dataset(
            _qmodel(),
            {"input_ids": object_ids, "attention_mask": object_masks},
            batch_size=1,
            calibration_data_min_length=0,
        )
    )


def test_dataframe_columns_are_unpacked_into_token_rows():
    pandas = pytest.importorskip("pandas")
    ids, masks = _columnar_token_rows()

    _assert_token_rows(
        prepare_calibration_dataset(
            _qmodel(),
            pandas.DataFrame({"input_ids": ids, "attention_mask": masks}),
            batch_size=1,
            calibration_data_min_length=0,
        )
    )


def test_dataframe_text_column_is_tokenized_per_row():
    pandas = pytest.importorskip("pandas")
    texts = ["one two three four", "five six seven eight"]

    prepared = prepare_calibration_dataset(
        _qmodel(),
        pandas.DataFrame({"text": texts}),
        batch_size=1,
        calibration_data_min_length=0,
    )

    # A DataFrame is not a Mapping; iterating it yields the column label
    # `"text"` and would calibrate on that single word.
    assert [batch["input_ids"].tolist() for batch in prepared] == [[[4]], [[4]]]


def test_dataframe_without_calibration_columns_is_rejected():
    pandas = pytest.importorskip("pandas")

    with pytest.raises(ValueError, match="must provide `input_ids`, `text`, or `messages`"):
        prepare_calibration_dataset(
            _qmodel(),
            pandas.DataFrame({"foo": [1, 2]}),
            calibration_data_min_length=1,
        )


def test_flat_object_column_is_still_one_token_row():
    np = pytest.importorskip("numpy")

    prepared = prepare_calibration_dataset(
        _qmodel(),
        {"input_ids": np.array([7, 8, 9], dtype=object)},
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert [batch["input_ids"].tolist() for batch in prepared] == [[[7, 8, 9]]]

