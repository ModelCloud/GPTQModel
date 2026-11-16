# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""VLM calibration must not silently strip multimodal tensors (#4371).

A row produced by ``processor(text=..., images=...)`` carries ``input_ids`` plus
``pixel_values``/``image_grid_thw``. Since 2b51d4d every mapping with
``input_ids`` was classified as pre-tokenized text and handed to the base text
path, which rebuilds the row from ``input_ids``/``attention_mask`` only, so the
image tensors vanished without a warning. A row is pre-tokenized *text* only
when its extra columns are text-only; rows with multimodal model inputs are
rejected with an actionable error instead.
"""

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from gptqmodel.models.definitions._vlm_calibration import (
    is_pretokenized_sample,
    split_calibration_dataset,
)
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ
from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel


MULTIMODAL_ROW_DEFINITIONS = [
    BaseQwen2VLGPTQ,
    Qwen2VLQModel,
    Gemma3ForConditionalGenerationGPTQ,
]


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        token_ids = [ord(char) for char in str(text)] or [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


def _make_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    return model


def _fail_load_processor(self):
    raise AssertionError("A rejected multimodal row must not load the multimodal processor")


def _multimodal_row():
    """The row from the issue: a single processor output with image tensors."""

    return {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "pixel_values": torch.randn(16, 1176),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


def test_multimodal_row_is_not_pretokenized_text():
    row = _multimodal_row()

    assert is_pretokenized_sample(row) is False


def test_multimodal_row_is_rejected_instead_of_reduced_to_text():
    row = _multimodal_row()

    with pytest.raises(ValueError, match="pixel_values") as excinfo:
        split_calibration_dataset([row], label="Qwen2VL")

    message = str(excinfo.value)
    assert "image_grid_thw" in message
    assert "silently dropped" in message
    assert "chat conversations" in message


def test_text_only_extra_columns_stay_pretokenized():
    row = {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "token_type_ids": [0] * 12,
    }

    assert is_pretokenized_sample(row) is True
    rows, conversations = split_calibration_dataset([row], label="Test")

    assert conversations is None
    assert rows == [row]


def test_scalar_metadata_columns_are_not_model_inputs():
    row = {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "id": 7,
        "source": "wiki",
    }

    rows, conversations = split_calibration_dataset([row], label="Test")

    assert conversations is None
    assert rows == [row]


def test_columnar_processor_mapping_with_images_is_rejected():
    encoding = BatchEncoding(
        {
            "input_ids": torch.ones((2, 6), dtype=torch.long),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
            "pixel_values": torch.randn(32, 1176),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        }
    )

    with pytest.raises(ValueError, match="pixel_values"):
        split_calibration_dataset(encoding, label="Qwen2VL")


def test_columnar_text_only_mapping_still_unpacks():
    rows, conversations = split_calibration_dataset(
        {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
            "token_type_ids": [[0, 0], [0, 0]],
        },
        label="Test",
    )

    assert conversations is None
    assert rows == [
        {"input_ids": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [3, 4], "attention_mask": [1, 1]},
    ]


@pytest.mark.parametrize(
    "cls", MULTIMODAL_ROW_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_prepare_dataset_never_reduces_multimodal_row_to_text(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    with pytest.raises(ValueError, match="pixel_values"):
        model.prepare_dataset([_multimodal_row()], batch_size=1)


def test_prepare_dataset_rejects_multimodal_row_after_text_rows(monkeypatch):
    monkeypatch.setattr(BaseQwen2VLGPTQ, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(BaseQwen2VLGPTQ)

    text_row = {"input_ids": list(range(12)), "attention_mask": [1] * 12}

    with pytest.raises(ValueError, match="calibration row 1"):
        model.prepare_dataset([text_row, _multimodal_row()], batch_size=1)


@pytest.mark.parametrize(
    "cls", MULTIMODAL_ROW_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_prepare_dataset_keeps_text_row_with_token_type_ids(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        [
            {"input_ids": list(range(12)), "attention_mask": [1] * 12, "token_type_ids": [0] * 12}
        ],
        batch_size=1,
    )

    assert len(batches) == 1
    assert set(batches[0]) == {"input_ids", "attention_mask"}
    assert batches[0]["input_ids"].tolist() == [list(range(12))]
