# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""VLM calibration must keep the base text path for pre-tokenized rows (#4228).

The public ``quantize()`` API accepts token-id rows, but the vision-language
definitions route every row through ``processor.apply_chat_template``, which
crashes on token ids. These tests pin the fallback for every VLM definition and
check that the calibration options are honoured or rejected explicitly.
"""

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.base_qwen2_5_omni import BaseQwen2_5_OmniGPTQ
from gptqmodel.models.definitions.base_qwen2_vl import BaseQwen2VLGPTQ
from gptqmodel.models.definitions.base_qwen3_vl import BaseQwen3VLGPTQ
from gptqmodel.models.definitions.gemma3 import Gemma3ForConditionalGenerationGPTQ
from gptqmodel.models.definitions.internvl import InternVLQModel
from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.definitions.lfm2_vl import LFM2VLQModel
from gptqmodel.models.definitions.llava import LlavaQModel
from gptqmodel.models.definitions.minicpm_o import MiniCPMOQModel
from gptqmodel.models.definitions.minicpmv import MiniCPMVQModel
from gptqmodel.models.definitions.minicpmv_4_6 import MiniCPMV4_6QModel
from gptqmodel.models.definitions.ovis2 import Ovis2QModel
from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
from gptqmodel.models.definitions.smolvlm import SmolVLMQModel


# Definitions that keep the base text path for pre-tokenized rows.
PRETOKENIZED_FALLBACK_DEFINITIONS = [
    BaseQwen2VLGPTQ,
    BaseQwen2_5_OmniGPTQ,
    BaseQwen3VLGPTQ,
    Gemma3ForConditionalGenerationGPTQ,
    InternVLQModel,
    InternVLChatQModel,
    LFM2VLQModel,
    LlavaQModel,
    MiniCPMOQModel,
    MiniCPMVQModel,
    MiniCPMV4_6QModel,
    Ovis2QModel,
    SmolVLMQModel,
    # Concrete definitions that inherit the base VLM implementations.
    Qwen2VLQModel,
    Qwen2_5_VLQModel,
    Qwen2_5_OmniGPTQ,
    Qwen3_VLQModel,
]

# Definitions that use the shared conversation helper (and therefore reject a
# dataset that mixes token ids with conversations).
MIXED_ROW_REJECTING_DEFINITIONS = [
    cls
    for cls in PRETOKENIZED_FALLBACK_DEFINITIONS
    if not issubclass(cls, BaseQwen3VLGPTQ)
    and cls is not Gemma3ForConditionalGenerationGPTQ
]


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        token_ids = [ord(ch) for ch in str(text)] or [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


class _FakeProcessor:
    """Processor that returns token lengths from a fixed queue."""

    def __init__(self, lengths):
        self.lengths = list(lengths)

    def _next_length(self):
        return self.lengths.pop(0) if self.lengths else 4

    def apply_chat_template(self, conversations, **kwargs):
        length = self._next_length()
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def __call__(self, *, text, images=None, audio=None, videos=None, padding=True, return_tensors="pt"):
        length = self._next_length()
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


def _make_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    return model


def _fail_load_processor(self):
    raise AssertionError("Pre-tokenized calibration must not load the multimodal processor")


def _conversation(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


@pytest.mark.parametrize(
    "cls", PRETOKENIZED_FALLBACK_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_vlm_pretokenized_id_lists_take_base_text_path(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset([list(range(12)), list(range(13))], batch_size=1)

    assert len(batches) == 2
    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]
    assert [batch["input_ids"].tolist() for batch in batches] == [
        [list(range(12))],
        [list(range(13))],
    ]


@pytest.mark.parametrize(
    "cls", PRETOKENIZED_FALLBACK_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_vlm_pretokenized_mapping_rows_take_base_text_path(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    rows = [
        {"input_ids": list(range(12)), "attention_mask": [1] * 12},
        # An interior mask zero is a caller-built mask, which the base path
        # keeps as given; edge zeros are padding it strips (#4889).
        {"input_ids": list(range(13)), "attention_mask": [1] * 6 + [0] + [1] * 6},
    ]

    batches = model.prepare_dataset(rows, batch_size=1)

    assert len(batches) == 2
    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]
    assert batches[1]["attention_mask"].tolist() == [[1] * 6 + [0] + [1] * 6]


@pytest.mark.parametrize(
    "cls", PRETOKENIZED_FALLBACK_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_vlm_pretokenized_rows_honour_base_calibration_options(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        [list(range(12)), list(range(12))],
        batch_size=1,
        calibration_dataset_concat_size=32,
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"][0].numel() == 32


@pytest.mark.parametrize(
    "cls", MIXED_ROW_REJECTING_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_vlm_mixed_rows_are_rejected_clearly(cls):
    model = _make_model(cls)

    with pytest.raises(ValueError, match="mixes pre-tokenized rows"):
        model.prepare_dataset(
            [list(range(12)), _conversation("hello")],
            batch_size=1,
        )


@pytest.mark.parametrize(
    "cls", PRETOKENIZED_FALLBACK_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_vlm_conversation_path_rejects_concat_size(cls):
    model = _make_model(cls)

    with pytest.raises(ValueError, match=r"CalibrationConfig\.concat_size"):
        model.prepare_dataset(
            [_conversation("hello")],
            batch_size=1,
            calibration_dataset_concat_size=64,
        )


def test_vlm_conversation_path_honours_sort_and_min_length(monkeypatch):
    processor = _FakeProcessor(lengths=[4, 20, 12])
    monkeypatch.setattr(SmolVLMQModel, "load_processor", lambda self: processor)
    model = _make_model(SmolVLMQModel)

    batches = model.prepare_dataset(
        [_conversation("a"), _conversation("b"), _conversation("c")],
        batch_size=1,
        calibration_dataset_sort="desc",
        calibration_data_min_length=10,
    )

    assert [int(batch["input_ids"].shape[-1]) for batch in batches] == [20, 12]


def test_vlm_conversation_path_keeps_short_rows_without_min_length(monkeypatch):
    processor = _FakeProcessor(lengths=[4, 20])
    monkeypatch.setattr(SmolVLMQModel, "load_processor", lambda self: processor)
    model = _make_model(SmolVLMQModel)

    batches = model.prepare_dataset(
        [_conversation("a"), _conversation("b")],
        batch_size=1,
    )

    assert [int(batch["input_ids"].shape[-1]) for batch in batches] == [4, 20]
