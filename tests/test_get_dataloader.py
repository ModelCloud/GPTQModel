# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Input handling for `get_dataloader` (issue #3880)."""

from __future__ import annotations

import pytest
from datasets import Dataset, IterableDatasetDict

from gptqmodel.utils.data import get_dataloader


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, truncation=False, **kwargs):
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def test_get_dataloader_default_preprocessor_builds_a_batch():
    dataloader = get_dataloader(
        "ignored",
        "prompt",
        "label",
        _Tokenizer(),
        load_fn=lambda _: Dataset.from_dict({"prompt": ["a"], "label": ["b"]}),
        num_samples=1,
    )

    batches = list(dataloader)
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 1
    assert batches[0]["attention_mask"].shape == batches[0]["input_ids"].shape


def test_get_dataloader_rejects_streaming_datasets():
    def load_fn(_):
        streaming = Dataset.from_dict(
            {"prompt": ["a"], "label": ["b"]}
        ).to_iterable_dataset()
        return IterableDatasetDict({"train": streaming})

    with pytest.raises(TypeError, match="streaming"):
        get_dataloader(
            "ignored",
            "prompt",
            "label",
            _Tokenizer(),
            load_fn=load_fn,
            num_samples=1,
        )
