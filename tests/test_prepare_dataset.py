# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.data import collate_data


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        if isinstance(text, list):
            raise ValueError("_StubTokenizer only supports string inputs")
        token_ids = [self._encode_char(ch) for ch in str(text)]
        attention = [1] * len(token_ids)
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        mask_tensor = torch.tensor([attention], dtype=torch.long)
        return {"input_ids": input_tensor, "attention_mask": mask_tensor}

    @staticmethod
    def _encode_char(ch: str) -> int:
        value = ord(ch)
        return value if value > 0 else 1


class _CausalMaskTokenizer(_StubTokenizer):
    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        tokenized = super().__call__(text, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
        seq = tokenized["input_ids"].shape[1]
        causal_mask = torch.tril(torch.ones((1, 1, seq, seq), dtype=torch.long))
        tokenized["attention_mask"] = causal_mask
        return tokenized


class _ChatStubTokenizer(_StubTokenizer):
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Mirror the HF chat-template path closely enough to verify precedence.
        assert tokenize is False
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class _MissingChatTemplateTokenizer(_StubTokenizer):
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise AssertionError("apply_chat_template should not be used when no chat template is configured")


class _RaisingGetChatTemplateTokenizer(_StubTokenizer):
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise AssertionError("apply_chat_template should not be used when get_chat_template raises")

    def get_chat_template(self, chat_template=None, tools=None):
        raise ValueError("tokenizer.chat_template is not set")


def _make_qmodel() -> BaseQModel:
    model = BaseQModel.__new__(BaseQModel)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    dummy_config = type("_Cfg", (), {"max_position_embeddings": 128})()
    dummy_model = type("_DummyModel", (), {"config": dummy_config})()
    model.model = dummy_model
    return model


def _make_qmodel_with_tokenizer(tokenizer) -> BaseQModel:
    model = _make_qmodel()
    model.tokenizer = tokenizer
    return model


def _sample_dataset():
    return [
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
        {"input_ids": [[3]], "attention_mask": [[1]]},
    ]


def test_prepare_dataset_concat_without_separator():
    qmodel = _make_qmodel()
    dataset = copy.deepcopy(_sample_dataset())

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_concat_size=5,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
        calibration_concat_separator=None,
    )

    assert len(batches) == 1
    input_ids = batches[0]["input_ids"].tolist()
    attention_mask = batches[0]["attention_mask"].int().tolist()

    assert input_ids == [[1, 2, 3, 0, 0]]
    assert attention_mask == [[1, 1, 1, 0, 0]]


def test_prepare_dataset_concat_with_separator():
    qmodel = _make_qmodel()
    dataset = copy.deepcopy(_sample_dataset())

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_concat_size=5,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
        calibration_concat_separator="##",
    )

    assert len(batches) == 1
    input_ids = batches[0]["input_ids"].tolist()
    attention_mask = batches[0]["attention_mask"].int().tolist()

    sep_tokens = [_StubTokenizer._encode_char("#"), _StubTokenizer._encode_char("#")]
    assert input_ids == [[1, 2, *sep_tokens, 3]]
    assert attention_mask == [[1, 1, 1, 1, 1]]


def test_prepare_dataset_splits_long_row_across_blocks():
    qmodel = _make_qmodel()
    long_row = {"input_ids": [[1, 2, 3, 4, 5, 6]], "attention_mask": [[1, 1, 1, 1, 1, 1]]}
    dataset = [long_row]

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_concat_size=5,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
        calibration_concat_separator=None,
    )

    assert len(batches) == 2
    first_ids = batches[0]["input_ids"].tolist()
    second_ids = batches[1]["input_ids"].tolist()
    first_mask = batches[0]["attention_mask"].int().tolist()
    second_mask = batches[1]["attention_mask"].int().tolist()
    assert first_ids == [[1, 2, 3, 4, 5]]
    assert first_mask == [[1, 1, 1, 1, 1]]
    assert second_ids == [[6, 0, 0, 0, 0]]
    assert second_mask == [[1, 0, 0, 0, 0]]


def test_prepare_dataset_collapses_causal_attention_mask():
    qmodel = _make_qmodel_with_tokenizer(_CausalMaskTokenizer())

    batches = qmodel.prepare_dataset(
        calibration_dataset=["abc"],
        calibration_dataset_concat_size=None,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
        calibration_concat_separator=None,
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[97, 98, 99]]
    assert batches[0]["attention_mask"].int().tolist() == [[1, 1, 1]]


def test_prepare_dataset_normalizes_rank_4_attention_mask():
    qmodel = _make_qmodel()
    keep = torch.tensor([True, True, True, False, False], dtype=torch.bool)
    seq_len = keep.numel()
    causal = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.bool)
    for query_idx in range(seq_len):
        causal[0, 0, query_idx] = keep & (torch.arange(seq_len) <= query_idx)

    dataset = [{"input_ids": [[1, 2, 3, 0, 0]], "attention_mask": causal}]

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[1, 2, 3, 0, 0]]
    assert batches[0]["attention_mask"].int().tolist() == [[1, 1, 1, 0, 0]]


def test_prepare_dataset_prefers_apply_chat_template_for_messages():
    qmodel = _make_qmodel()
    qmodel.tokenizer = _ChatStubTokenizer()
    dataset = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "text": "raw-fallback",
        }
    ]

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
    )

    templated = qmodel.tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False, add_generation_prompt=False)
    expected_ids = qmodel.tokenizer(templated, return_tensors="pt")["input_ids"].tolist()
    raw_text_ids = qmodel.tokenizer(dataset[0]["text"], return_tensors="pt")["input_ids"].tolist()

    assert batches[0]["input_ids"].tolist() == expected_ids
    assert batches[0]["input_ids"].tolist() != raw_text_ids


def test_prepare_dataset_falls_back_to_text_when_chat_template_is_missing():
    qmodel = _make_qmodel()
    qmodel.tokenizer = _MissingChatTemplateTokenizer()
    dataset = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "text": "raw-fallback",
        }
    ]

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
    )

    expected_ids = qmodel.tokenizer(dataset[0]["text"], return_tensors="pt")["input_ids"].tolist()
    assert batches[0]["input_ids"].tolist() == expected_ids


def test_prepare_dataset_falls_back_to_text_when_get_chat_template_raises():
    qmodel = _make_qmodel()
    qmodel.tokenizer = _RaisingGetChatTemplateTokenizer()
    dataset = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            "text": "raw-fallback",
        }
    ]

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
    )

    expected_ids = qmodel.tokenizer(dataset[0]["text"], return_tensors="pt")["input_ids"].tolist()
    assert batches[0]["input_ids"].tolist() == expected_ids


def test_collate_data_uses_right_padding_by_default():
    batch = [
        {
            "input_ids": [[1, 2, 3], [4, 5]],
            "attention_mask": [[1, 1, 1], [1, 1]],
        },
    ]

    result = collate_data(batch, pad_token_id=0)

    assert result["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert result["attention_mask"].int().tolist() == [[1, 1, 1], [1, 1, 0]]


def test_collate_data_left_padding_when_requested():
    batch = [
        {
            "input_ids": [[1, 2, 3], [4, 5]],
            "attention_mask": [[1, 1, 1], [1, 1]],
        },
    ]

    result = collate_data(batch, pad_token_id=0, padding_side="left")

    assert result["input_ids"].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert result["attention_mask"].int().tolist() == [[1, 1, 1], [0, 1, 1]]


def test_collate_data_raises_for_invalid_padding_side():
    batch = [
        {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }
    ]

    with pytest.raises(ValueError, match="Unsupported padding_side"):
        collate_data(batch, pad_token_id=0, padding_side="center")


def test_prepare_dataset_uses_tokenizer_padding_side_left():
    qmodel = _make_qmodel()
    qmodel.tokenizer.padding_side = "left"

    batches = qmodel.prepare_dataset(
        calibration_dataset=[[1, 2, 3, 4], [5, 6]],
        calibration_dataset_sort=None,
        batch_size=2,
        calibration_data_min_length=0,
    )

    assert batches[0]["input_ids"].tolist() == [[1, 2, 3, 4], [0, 0, 5, 6]]
    assert batches[0]["attention_mask"].int().tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]


def test_prepare_dataset_concat_respects_tokenizer_padding_side_left():
    qmodel = _make_qmodel()
    qmodel.tokenizer.padding_side = "left"
    dataset = copy.deepcopy(_sample_dataset())

    batches = qmodel.prepare_dataset(
        calibration_dataset=dataset,
        calibration_dataset_concat_size=5,
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
        calibration_concat_separator=None,
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[0, 0, 1, 2, 3]]
    assert batches[0]["attention_mask"].int().tolist() == [[0, 0, 1, 1, 1]]


def test_prepare_dataset_trims_left_padded_rows_from_the_left_edge():
    qmodel = _make_qmodel()
    qmodel.tokenizer.padding_side = "left"
    qmodel.model.config.max_position_embeddings = 4

    batches = qmodel.prepare_dataset(
        calibration_dataset=[{"input_ids": [[0, 0, 11, 12, 13, 14]], "attention_mask": [[0, 0, 1, 1, 1, 1]]}],
        calibration_dataset_sort=None,
        batch_size=1,
        calibration_data_min_length=0,
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [[11, 12, 13, 14]]
    assert batches[0]["attention_mask"].int().tolist() == [[1, 1, 1, 1]]
