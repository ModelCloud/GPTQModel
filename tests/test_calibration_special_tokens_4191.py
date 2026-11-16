# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Chat-template calibration rows must not be tokenized twice with special tokens.

Issue #4191: ``apply_chat_template(..., tokenize=False)`` already renders the
template's own BOS for Llama/Gemma-style models. Re-tokenizing that text with
``add_special_tokens=True`` prepended a second BOS to every ``messages`` row.
The concatenation separator had the same problem: ``"\\n"`` became ``[BOS, NL]``
and inserted a BOS at every document boundary.
"""

from types import SimpleNamespace

import torch

from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.paroquant_benchmark import _tokenize_calibration_sample


class _ChatTokenizer:
    """Small tokenizer that renders a BOS into the chat template itself."""

    bos_token = "<bos>"
    bos_token_id = 1
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"

    def __init__(self, include_bos: bool = True):
        self.include_bos = include_bos
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        body = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        return (self.bos_token if self.include_bos else "") + body

    def __call__(self, text, add_special_tokens=True, return_tensors="pt"):
        assert return_tensors == "pt"
        self.calls.append((text, add_special_tokens))

        token_ids = []
        position = 0
        while position < len(text):
            if text.startswith(self.bos_token, position):
                token_ids.append(self.bos_token_id)
                position += len(self.bos_token)
            else:
                token_ids.append(ord(text[position]))
                position += 1

        if add_special_tokens:
            token_ids = [self.bos_token_id, *token_ids]

        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=8192),
        ),
    )


_MESSAGES = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "world"},
]


def test_messages_row_has_exactly_one_bos_from_the_template():
    tokenizer = _ChatTokenizer()

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        [{"messages": _MESSAGES}],
        calibration_data_min_length=0,
    )

    input_ids = prepared[0]["input_ids"][0].tolist()
    assert input_ids[0] == tokenizer.bos_token_id
    assert input_ids.count(tokenizer.bos_token_id) == 1

    rendered = tokenizer.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=False)
    assert (rendered, False) in tokenizer.calls


def test_messages_row_without_template_bos_does_not_gain_one():
    tokenizer = _ChatTokenizer(include_bos=False)

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        [{"messages": _MESSAGES}],
        calibration_data_min_length=0,
    )

    input_ids = prepared[0]["input_ids"][0].tolist()
    assert tokenizer.bos_token_id not in input_ids


def test_raw_text_calibration_still_adds_special_tokens():
    tokenizer = _ChatTokenizer()

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        ["hello"],
        calibration_data_min_length=0,
    )

    input_ids = prepared[0]["input_ids"][0].tolist()
    assert input_ids.count(tokenizer.bos_token_id) == 1
    assert ("hello", True) in tokenizer.calls


def test_concat_separator_does_not_insert_bos_at_document_boundaries():
    tokenizer = _ChatTokenizer()
    dataset = [
        {"input_ids": [[11, 12, 13, 14]], "attention_mask": [[1, 1, 1, 1]]},
        {"input_ids": [[21, 22, 23, 24]], "attention_mask": [[1, 1, 1, 1]]},
    ]

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        dataset,
        calibration_dataset_concat_size=10,
        calibration_data_min_length=0,
        calibration_concat_separator="\n",
    )

    input_ids = prepared[0]["input_ids"][0].tolist()
    assert input_ids == [11, 12, 13, 14, 10, 21, 22, 23, 24, 0]
    assert input_ids.count(tokenizer.bos_token_id) == 0
    assert ("\n", False) in tokenizer.calls


def test_concat_separator_that_fits_fills_the_block():
    """A one-token separator must not force a short, unpadded block."""

    tokenizer = _ChatTokenizer()
    dataset = [
        {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]], "attention_mask": [[1] * 8]},
        {"input_ids": [[11, 12, 13, 14, 15, 16, 17, 18]], "attention_mask": [[1] * 8]},
    ]

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        dataset,
        calibration_dataset_concat_size=9,
        calibration_data_min_length=0,
        calibration_concat_separator="\n",
    )

    first_ids = prepared[0]["input_ids"][0].tolist()
    first_mask = prepared[0]["attention_mask"][0].tolist()
    assert first_ids == [1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert first_mask == [1] * 9
    assert prepared[1]["input_ids"][0].tolist() == [11, 12, 13, 14, 15, 16, 17, 18, 0]


def test_paroquant_benchmark_messages_row_has_exactly_one_bos():
    tokenizer = _ChatTokenizer()
    model = SimpleNamespace(tokenizer=tokenizer)

    tokenized = _tokenize_calibration_sample(model, {"messages": _MESSAGES})

    input_ids = tokenized["input_ids"][0].tolist()
    assert input_ids[0] == tokenizer.bos_token_id
    assert input_ids.count(tokenizer.bos_token_id) == 1

    rendered = tokenizer.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=False)
    assert (rendered, False) in tokenizer.calls
