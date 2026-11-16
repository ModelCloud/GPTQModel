# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""VLM calibration must not silently drop processor-produced multimodal tensors.

``processor(text=..., images=...)`` returns a mapping with ``input_ids``,
``attention_mask``, ``pixel_values`` and ``image_grid_thw``. The shared VLM
splitter used to classify any mapping containing ``input_ids`` as pre-tokenized
text; the base text path then kept only the two token columns and dropped the
image tensors (issue #4371). Rows with non-text columns are now rejected with an
actionable error, and text-only pre-tokenized rows still take the text path.
"""

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions._vlm_calibration import split_calibration_dataset
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"


def _make_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    return model


def _fail_load_processor(self):
    raise AssertionError("Calibration row handling must not load the multimodal processor")


def _processor_row():
    return {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "pixel_values": torch.randn(16, 1176),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


def test_split_calibration_dataset_rejects_multimodal_processor_row():
    with pytest.raises(ValueError) as excinfo:
        split_calibration_dataset([_processor_row()], label="Qwen2VL")

    message = str(excinfo.value)
    assert "pixel_values" in message
    assert "image_grid_thw" in message
    assert "silently dropped" in message


def test_prepare_dataset_rejects_multimodal_processor_row(monkeypatch):
    monkeypatch.setattr(BaseQwen2VLGPTQ, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(BaseQwen2VLGPTQ)

    with pytest.raises(ValueError, match="pixel_values"):
        model.prepare_dataset([_processor_row()], batch_size=1)


def test_split_calibration_dataset_rejects_columnar_processor_output():
    columnar = {
        "input_ids": [[1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1, 1]],
        "pixel_values": [torch.randn(16, 1176)],
        "image_grid_thw": [torch.tensor([[1, 2, 2]])],
    }

    with pytest.raises(ValueError, match="pixel_values"):
        split_calibration_dataset(columnar, label="Qwen2VL")


def test_text_only_pretokenized_row_still_takes_text_path(monkeypatch):
    monkeypatch.setattr(BaseQwen2VLGPTQ, "load_processor", _fail_load_processor, raising=False)
    row = {
        "input_ids": list(range(12)),
        "attention_mask": [1] * 12,
        "token_type_ids": [0] * 12,
        "labels": [-100] + list(range(11)),
    }

    rows, conversations = split_calibration_dataset([row], label="Qwen2VL")

    assert conversations is None
    assert rows == [row]

    model = _make_model(BaseQwen2VLGPTQ)
    batches = model.prepare_dataset([row], batch_size=1)

    assert len(batches) == 1
    assert batches[0]["input_ids"].tolist() == [list(range(12))]
    assert set(batches[0]) == {"input_ids", "attention_mask"}
