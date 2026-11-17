# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicit ``input_ids`` win over a co-existing ``messages`` column (#4670).

``Dataset.map(lambda r: tok.apply_chat_template(r["messages"], tokenize=True, ...))``
keeps the ``messages`` column next to the new ``input_ids``/``attention_mask``,
and so do ``dataset[:]`` and ``list(dataset)``. The shared calibration helper
checked ``messages`` first, so those rows were re-rendered through the chat
template and fed whole conversations instead of the caller's truncated ids
(#4638 fixed the ``text`` half only).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.calibration import prepare_calibration_dataset


LONG = "The quick brown fox jumps over the lazy dog. " * 300


class _RecordingTokenizer:
    """Tokenize by word count and remember every call."""

    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"

    def __init__(self):
        self.calls: list[str] = []
        self.template_calls: list[object] = []

    def __call__(self, text, **_kwargs):
        self.calls.append(text)
        return {
            "input_ids": torch.tensor([[len(text.split())] * 4]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    def apply_template(self, messages, **_kwargs):
        self.template_calls.append(messages)
        # A long rendered conversation, so ignoring the provided ids is visible.
        return LONG


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=4096),
        ),
    )


def test_a_row_with_messages_and_input_ids_uses_the_provided_ids():
    tokenizer = _RecordingTokenizer()
    row = {
        "messages": [{"role": "user", "content": LONG}],
        "input_ids": [11, 12, 13, 14],
        "attention_mask": [1, 1, 1, 0],
    }

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), [row], batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.template_calls == []
    assert tokenizer.calls == []
    assert prepared[0]["input_ids"].tolist() == [[11, 12, 13, 14]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1, 0]]


def test_columnar_mapping_with_messages_and_input_ids_uses_the_provided_ids():
    tokenizer = _RecordingTokenizer()
    columns = {
        "messages": [
            [{"role": "user", "content": LONG}],
            [{"role": "user", "content": LONG}],
        ],
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
    }

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), columns, batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.template_calls == []
    assert [row["input_ids"].tolist() for row in prepared] == [
        [[1, 2, 3]],
        [[4, 5, 6]],
    ]


def test_a_mapped_hf_dataset_keeps_the_truncated_ids():
    pytest.importorskip("datasets")
    from datasets import Dataset

    tokenizer = _RecordingTokenizer()
    source = Dataset.from_dict(
        {"messages": [[{"role": "user", "content": LONG}]] * 3}
    )
    mapped = source.map(
        lambda row: {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]},
        remove_columns=[],
    )
    assert "messages" in mapped.column_names

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer), mapped, batch_size=1, calibration_data_min_length=1
    )

    assert tokenizer.template_calls == []
    assert all(row["input_ids"].tolist() == [[7, 8, 9]] for row in prepared)


def test_a_row_with_only_messages_still_renders_the_template():
    tokenizer = _RecordingTokenizer()

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        [{"messages": [{"role": "user", "content": "hello world"}]}],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(tokenizer.template_calls) == 1
    assert prepared[0]["input_ids"].tolist() == [[len(LONG.split())] * 4]
