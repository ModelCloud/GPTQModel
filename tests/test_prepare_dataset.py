# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy

import torch

from gptqmodel.models.base import BaseQModel


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


def _make_qmodel() -> BaseQModel:
    model = BaseQModel.__new__(BaseQModel)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    dummy_config = type("_Cfg", (), {"max_position_embeddings": 128})()
    dummy_model = type("_DummyModel", (), {"config": dummy_config})()
    model.model = dummy_model
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
