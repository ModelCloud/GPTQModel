# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicit calibration token ids win over a co-existing ``text`` column (#4635).

``Dataset.map(tokenizer, truncation=True, max_length=N)`` keeps the source
``text`` column next to the new ``input_ids``/``attention_mask``, and so does
``dataset[:]``. The shared calibration helper checked ``text`` before
``input_ids``, so those rows were silently re-tokenized from the full document:
the caller's truncation, special-token, and chat choices were discarded and the
capture grew to the model's whole context window.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.calibration import prepare_calibration_dataset


class _RecordingTokenizer:
    """Tokenize by word count and remember every call."""

    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, text, **_kwargs):
        self.calls.append(text)
        return {
            "input_ids": torch.tensor([[len(text.split())] * 4]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=4096),
        ),
    )


def test_a_row_with_text_and_input_ids_uses_the_provided_ids():
    tokenizer = _RecordingTokenizer()
    row = {
        "text": "a very long document " * 100,
        "input_ids": [11, 12, 13, 14],
        "attention_mask": [1, 1, 1, 0],
    }

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), [row], batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.calls == []
    # The trailing mask-0 slot is tokenizer padding and is stripped (#4889);
    # the ids still come from the caller, not from re-tokenizing `text`.
    assert prepared[0]["input_ids"].tolist() == [[11, 12, 13]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_a_columnar_mapping_with_text_and_input_ids_uses_the_provided_ids():
    tokenizer = _RecordingTokenizer()
    columns = {
        "text": ["a very long document " * 100, "another long document " * 100],
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
    }

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), columns, batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.calls == []
    assert [row["input_ids"].tolist() for row in prepared] == [
        [[1, 2, 3]],
        [[4, 5]],
    ]


def test_a_mapped_hf_dataset_keeps_the_truncated_ids():
    pytest.importorskip("datasets")
    from datasets import Dataset

    tokenizer = _RecordingTokenizer()
    source = Dataset.from_dict({"text": ["a very long document " * 100] * 3})
    mapped = source.map(
        lambda row: {
            "input_ids": [7, 8, 9],
            "attention_mask": [1, 1, 1],
        },
        remove_columns=[],
    )
    assert "text" in mapped.column_names

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), mapped, batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.calls == []
    assert all(row["input_ids"].tolist() == [[7, 8, 9]] for row in prepared)


def test_a_row_with_only_text_still_tokenizes_the_text():
    tokenizer = _RecordingTokenizer()

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        [{"text": "one two three"}],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert tokenizer.calls == ["one two three"]
    assert prepared[0]["input_ids"].tolist() == [[3, 3, 3, 3]]
