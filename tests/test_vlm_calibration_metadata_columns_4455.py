# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4455: VLM calibration accepts harmless metadata columns.

``datasets.map(tokenize)`` keeps the source columns, so pre-tokenized rows
commonly carry ``id``/``url``/``source`` metadata next to ``input_ids``. Those
values are not model inputs; the text calibration path may ignore them. The
#4371 protection still rejects multimodal inputs (``pixel_values``,
``image_grid_thw``, image paths, ...) and any tensor/array-valued column.
"""

import numpy as np
import pytest
import torch
from torch import nn

from gptqmodel.models.definitions._vlm_calibration import split_calibration_dataset
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ


pytestmark = [pytest.mark.cpu]


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"


def _fail_load_processor(self):
    raise AssertionError("Pre-tokenized rows must not load the multimodal processor")


def _row(**extra):
    return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], **extra}


@pytest.mark.parametrize(
    "metadata",
    [
        {"id": 7, "url": "x"},
        {"id": np.int64(3), "score": 0.5, "source": None, "is_valid": True},
        {"tags": ["web", "en"], "title": "doc"},
    ],
    ids=["issue-repro", "numpy-scalar-float-none-bool", "string-list"],
)
def test_metadata_columns_take_text_path(metadata):
    row = _row(**metadata)

    rows, conversations = split_calibration_dataset([row], label="Qwen2VL")

    assert conversations is None
    assert rows == [row]


def test_metadata_columns_columnar_mapping_take_text_path():
    columnar = {
        "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "id": [0, 1],
        "url": ["a", "b"],
    }

    rows, conversations = split_calibration_dataset(columnar, label="Qwen2VL")

    assert conversations is None
    assert [row["input_ids"] for row in rows] == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_metadata_columns_prepare_dataset_keeps_token_columns(monkeypatch):
    monkeypatch.setattr(BaseQwen2VLGPTQ, "load_processor", _fail_load_processor, raising=False)
    model = object.__new__(BaseQwen2VLGPTQ)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True

    # Longer than the default calibration minimum length.
    row = {"input_ids": list(range(12)), "attention_mask": [1] * 12, "id": 7, "url": "x"}
    batches = model.prepare_dataset([row], batch_size=1)

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [list(range(12))]
    assert set(batches[0]) == {"input_ids", "attention_mask"}


@pytest.mark.parametrize(
    "extra",
    [
        {"pixel_values": torch.randn(4, 8)},
        {"image_grid_thw": [[1, 2, 2]]},
        {"video_grid_thw": torch.tensor([[1, 2, 2]])},
        {"image": "cat.png"},
        {"image_url": "https://example.invalid/cat.png"},
        {"input_features": [[0.1, 0.2]]},
        {"embedding": np.zeros((2, 3), dtype=np.float32)},
        {"extra_ids": [1, 2, 3]},
        {"blob": b"\x89PNG"},
        {"payload": {"bytes": b"", "path": "x.png"}},
        {"vec": torch.tensor(1.0)},
    ],
    ids=lambda extra: next(iter(extra)),
)
def test_multimodal_or_tensor_columns_still_rejected(extra):
    name = next(iter(extra))
    with pytest.raises(ValueError, match="silently dropped") as excinfo:
        split_calibration_dataset([_row(id=7, **extra)], label="Qwen2VL")

    message = str(excinfo.value)
    assert name in message
    assert "'id'" not in message


def test_columnar_tensor_column_still_rejected():
    columnar = {
        "input_ids": [[1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1, 1]],
        "id": [0],
        "extra": [torch.randn(2)],
    }

    with pytest.raises(ValueError, match="extra"):
        split_calibration_dataset(columnar, label="Qwen2VL")
