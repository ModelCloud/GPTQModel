# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Sample and block length enforcement in `make_data_block` (issue #3882)."""

from __future__ import annotations

import pytest

from gptqmodel.utils.data import make_data_block


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, truncation=False):
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def _row_lengths(result):
    return [len(row) for block in result["input_ids"] for row in block]


def _samples(prompt="ab", label="abcdefghij"):
    return {"prompt": [prompt], "label": [label]}


def test_label_longer_than_sample_max_len_is_capped():
    result = make_data_block(
        _samples(),
        "prompt",
        "label",
        _Tokenizer(),
        sample_max_len=5,
        block_max_len=20,
        merge_prompt_label=True,
    )

    assert _row_lengths(result) == [5]
    assert all(block for block in result["input_ids"])


def test_oversized_first_sample_does_not_create_an_empty_block():
    result = make_data_block(
        _samples(),
        "prompt",
        "label",
        _Tokenizer(),
        sample_max_len=5,
        block_max_len=5,
        merge_prompt_label=True,
    )

    assert _row_lengths(result) == [5]
    assert all(block for block in result["input_ids"])


def test_sample_that_cannot_fit_a_block_raises_a_clear_error():
    with pytest.raises(ValueError, match="block_max_len"):
        make_data_block(
            _samples(),
            "prompt",
            "label",
            _Tokenizer(),
            sample_max_len=5,
            block_max_len=3,
            merge_prompt_label=True,
        )


def test_label_is_truncated_when_prompt_truncation_is_disabled():
    result = make_data_block(
        _samples(prompt="abcd", label="efghij"),
        "prompt",
        "label",
        _Tokenizer(),
        sample_max_len=5,
        block_max_len=20,
        truncate_prompt=False,
    )

    assert _row_lengths(result) == [4]
    assert len(result["labels"][0][0]) == 1
