# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""VLM calibration must not silently drop processor tensors (issue #4371).

``split_calibration_dataset`` used to classify any mapping containing
``input_ids`` as pre-tokenized text. A row produced by
``processor(text=..., images=...)`` therefore kept its image placeholder tokens
but lost ``pixel_values``/``image_grid_thw`` on the base token-id path. These
tests pin the three supported row kinds: text-only token ids still use the base
path, multimodal processor rows are rejected with an actionable error, and chat
conversations still use the processor.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions._vlm_calibration import (
    is_pretokenized_sample,
    split_calibration_dataset,
)
from gptqmodel.models.definitions.base_qwen2_5_omni import BaseQwen2_5_OmniGPTQ
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ
from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
from gptqmodel.models.definitions.internvl import InternVLQModel
from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.definitions.lfm2_vl import LFM2VLQModel
from gptqmodel.models.definitions.llava import LlavaQModel
from gptqmodel.models.definitions.minicpmv import MiniCPMVQModel
from gptqmodel.models.definitions.ovis2 import Ovis2QModel


SHARED_HELPER_DEFINITIONS = [
    BaseQwen2VLGPTQ,
    BaseQwen2_5_OmniGPTQ,
    Gemma3ForConditionalGenerationGPTQ,
    InternVLQModel,
    InternVLChatQModel,
    LFM2VLQModel,
    LlavaQModel,
    MiniCPMVQModel,
    Ovis2QModel,
]


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = None
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, texts, return_tensors="pt", add_special_tokens=True, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[ord(char) for char in text] or [1] for text in texts]
        if padding:
            width = max(len(row) for row in rows)
            rows = [row + [0] * (width - len(row)) for row in rows]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(token)) for token in ids)


def _make_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    return model


def _fail_load_processor(self):
    raise AssertionError("Pre-tokenized calibration must not load the multimodal processor")


def _qwen2_vl_processor_row():
    """A row shaped like ``processor(text=..., images=...)`` output."""

    return {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "pixel_values": torch.randn(16, 1176),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


def _conversation(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


class _FakeQwen2VLProcessor:
    def __init__(self, length=4):
        self.length = length
        self.seen = []
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.seen.append(conversations)
        return "rendered chat"

    def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt"):
        self.calls.append({"text": text, "images": images, "videos": videos})
        input_ids = torch.ones((1, self.length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }


class _FakeGemmaProcessor:
    def __init__(self, length=20):
        self.length = length
        self.seen = []

    def apply_chat_template(self, conversations, **kwargs):
        self.seen.append(conversations)
        input_ids = torch.ones((1, self.length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }


def test_text_only_extra_columns_stay_on_pretokenized_path():
    row = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "token_type_ids": [0, 0, 0],
        "labels": [1, 2, 3],
    }

    assert is_pretokenized_sample(row)
    rows, conversations = split_calibration_dataset([row], label="Test")

    assert conversations is None
    assert rows == [row]


def test_none_valued_processor_columns_are_harmless():
    row = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "pixel_values": None,
    }

    assert is_pretokenized_sample(row)
    rows, conversations = split_calibration_dataset([row], label="Test")

    assert conversations is None
    assert rows == [row]


def test_text_column_alongside_input_ids_stays_accepted():
    row = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "text": "abc",
    }

    assert is_pretokenized_sample(row)
    rows, conversations = split_calibration_dataset([row], label="Test")

    assert conversations is None
    assert rows == [row]


@pytest.mark.parametrize(
    "cls", SHARED_HELPER_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_text_only_pretokenized_rows_still_prepare(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        [
            {"input_ids": list(range(12)), "attention_mask": [1] * 12, "token_type_ids": [0] * 12},
            {"input_ids": list(range(13)), "attention_mask": [1] * 13, "token_type_ids": [0] * 13},
        ],
        batch_size=1,
    )

    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]


@pytest.mark.parametrize(
    "cls", SHARED_HELPER_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_multimodal_processor_rows_are_rejected_not_dropped(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    row = _qwen2_vl_processor_row()

    assert not is_pretokenized_sample(row)
    with pytest.raises(ValueError, match="pixel_values") as excinfo:
        model.prepare_dataset([row], batch_size=1)

    message = str(excinfo.value)
    assert "image_grid_thw" in message
    assert "silently drop" in message
    assert "messages" in message


def test_multimodal_row_after_text_rows_is_rejected():
    text_row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    with pytest.raises(ValueError, match="pixel_values"):
        split_calibration_dataset([text_row, _qwen2_vl_processor_row()], label="Test")


def test_single_row_mapping_with_multimodal_columns_is_rejected():
    with pytest.raises(ValueError, match="pixel_values"):
        split_calibration_dataset(_qwen2_vl_processor_row(), label="Test")


def test_columnar_mapping_with_multimodal_columns_is_rejected():
    columnar = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
        "pixel_values": [torch.randn(4, 8), torch.randn(4, 8)],
    }

    with pytest.raises(ValueError, match="pixel_values"):
        split_calibration_dataset(columnar, label="Test")


def test_conversation_rows_still_use_the_processor(monkeypatch):
    processor = _FakeQwen2VLProcessor(length=7)
    monkeypatch.setattr(BaseQwen2VLGPTQ, "load_processor", lambda self: processor, raising=False)
    model = _make_model(BaseQwen2VLGPTQ)
    conversation = _conversation("hello")

    rows, conversations = split_calibration_dataset([conversation], label="Test")
    assert rows is None
    assert list(conversations) == [conversation]

    batches = model.prepare_dataset([conversation], batch_size=1)

    assert [int(batch["input_ids"].shape[-1]) for batch in batches] == [7]
    assert processor.seen == [[conversation]]
    assert processor.calls[0]["images"] is None


def test_gemma3_conversation_rows_still_use_the_processor(monkeypatch):
    processor = _FakeGemmaProcessor(length=20)
    monkeypatch.setattr(
        Gemma3ForConditionalGenerationGPTQ,
        "load_processor",
        lambda self: processor,
        raising=False,
    )
    model = _make_model(Gemma3ForConditionalGenerationGPTQ)
    conversation = _conversation("hello")

    batches = model.prepare_dataset([conversation], batch_size=1)

    assert [int(batch["input_ids"].shape[-1]) for batch in batches] == [20]
    assert processor.seen == [[conversation]]
