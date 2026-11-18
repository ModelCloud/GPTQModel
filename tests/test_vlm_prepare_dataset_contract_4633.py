# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Every VLM ``prepare_dataset`` override keeps the shared calibration contract.

#4228/#4308 gave the vision-language overrides one contract in
``models/definitions/_vlm_calibration.py``: pre-tokenized rows and columnar
mappings take the base text path, ``CalibrationConfig.concat_size`` is rejected for
conversations, and the processor path only sees chat conversations. Definitions
added afterwards copied the conversation loop and silently bypassed it, so a
documented ``[{"input_ids": ...}]`` dataset was rendered as an empty chat
prompt or calibrated on the literal text ``"input_ids"`` (issue #4633).

The parametrization discovers every exported definition that overrides
``prepare_dataset`` instead of listing models by hand, so a new definition
cannot skip the contract again.
"""

import inspect

import pytest
import torch
from torch import nn

import gptqmodel.models.definitions as definitions
from gptqmodel.models.base import BaseQModel


# Every exported definition with its own ``prepare_dataset``. Inherited
# implementations are already covered by the definitions that own them.
VLM_PREPARE_DATASET_DEFINITIONS = sorted(
    (
        cls
        for _, cls in inspect.getmembers(definitions, inspect.isclass)
        if issubclass(cls, BaseQModel) and "prepare_dataset" in cls.__dict__
    ),
    key=lambda cls: (cls.__module__, cls.__name__),
)


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    chat_template = "stub"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        token_ids = [ord(char) for char in str(text)] or [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


class _PaddingProcessor:
    """Processor that pads like the real chat templates, or raises like them.

    ``apply_chat_template`` without padding cannot stack unequal conversation
    lengths into a tensor. Reproducing that failure keeps the ``batch_size > 1``
    regression observable instead of masking it behind a fixed-width stub.
    """

    def __init__(self):
        self.calls = []

    @staticmethod
    def _length(conversation):
        return max(1, len(str(conversation)))

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(kwargs)
        if isinstance(conversations, dict):
            conversations = [conversations]
        lengths = [self._length(conversation) for conversation in conversations]
        padding = kwargs.get("padding")
        if padding is None:
            padding = kwargs.get("processor_kwargs", {}).get("padding")
        if not padding and len(set(lengths)) > 1:
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation "
                "and/or padding with 'padding=True' and 'truncation=True'."
            )
        width = max(lengths)
        input_ids = torch.ones((len(conversations), width), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def _make_model(cls):
    model = object.__new__(cls)
    nn.Module.__init__(model)
    model.tokenizer = _StubTokenizer()
    model.support_batch_quantize = True
    return model


def _fail_load_processor(self):
    raise AssertionError(
        "Pre-tokenized calibration must not load the multimodal processor"
    )


_CALLER_MASK = [1] * 20 + [0] + [1] * 24


def _conversation(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def test_discovery_covers_the_models_named_in_the_issue():
    names = {cls.__name__ for cls in VLM_PREPARE_DATASET_DEFINITIONS}

    assert {
        "CohereCompassQModel",
        "HunYuanVLQModel",
        "MuseGlimmerQModel",
        "InklingMMQModel",
    } <= names


@pytest.mark.parametrize(
    "cls", VLM_PREPARE_DATASET_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_pretokenized_rows_never_reach_the_processor(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    rows = [
        {"input_ids": list(range(100, 140)), "attention_mask": [1] * 40},
        # An interior mask zero is a caller-built mask, which the base path
        # keeps as given; edge zeros are padding it strips (#4889).
        {"input_ids": list(range(200, 245)), "attention_mask": _CALLER_MASK},
    ]

    batches = model.prepare_dataset(rows, batch_size=1)

    assert [batch["input_ids"].tolist() for batch in batches] == [
        [list(range(100, 140))],
        [list(range(200, 245))],
    ]
    assert batches[1]["attention_mask"].tolist() == [_CALLER_MASK]


@pytest.mark.parametrize(
    "cls", VLM_PREPARE_DATASET_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_columnar_mapping_is_unpacked_not_rendered_as_key_names(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        {
            "input_ids": [list(range(100, 140)), list(range(200, 245))],
            "attention_mask": [[1] * 40, _CALLER_MASK],
        },
        batch_size=1,
    )

    assert [batch["input_ids"].tolist() for batch in batches] == [
        [list(range(100, 140))],
        [list(range(200, 245))],
    ]


@pytest.mark.parametrize(
    "cls", VLM_PREPARE_DATASET_DEFINITIONS, ids=lambda cls: cls.__name__
)
def test_conversation_path_rejects_concat_size(cls):
    model = _make_model(cls)

    with pytest.raises(ValueError, match=r"CalibrationConfig\.concat_size"):
        model.prepare_dataset(
            [_conversation("hello")],
            batch_size=1,
            calibration_dataset_concat_size=64,
        )


@pytest.mark.parametrize(
    "cls",
    [definitions.CohereCompassQModel, definitions.InklingMMQModel],
    ids=lambda cls: cls.__name__,
)
def test_batch_size_two_pads_unequal_conversations(cls, monkeypatch):
    processor = _PaddingProcessor()
    monkeypatch.setattr(cls, "load_processor", lambda self: processor, raising=False)
    model = _make_model(cls)

    conversations = [
        _conversation("short."),
        _conversation("a much longer calibration sample."),
    ]

    batches = model.prepare_dataset(conversations, batch_size=2)

    expected_width = max(len(str(row)) for row in conversations)
    assert len(batches) == 1
    assert batches[0]["input_ids"].shape == (2, expected_width)
    assert processor.calls[0]["padding"] is True


@pytest.mark.parametrize(
    "cls",
    [definitions.UnlimitedOCRQModel, definitions.Ovis2_6_MoeQModel],
    ids=lambda cls: cls.__name__,
)
def test_processor_prepared_rows_are_ready_batches_under_the_contract(cls, monkeypatch):
    """Rows the model's own processor built pass through, but keep the contract.

    Both definitions used to return any ``input_ids`` dict list untouched, so
    ``concat_size``/``sort``/``min_length`` were ignored and text-only token
    rows skipped the base text path entirely.
    """

    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    rows = [
        {
            "input_ids": torch.ones((1, width), dtype=torch.long),
            "attention_mask": torch.ones((1, width), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 4, 4)),
        }
        for width in (3, 9, 6)
    ]

    batches = model.prepare_dataset(
        rows, batch_size=1, calibration_dataset_sort="desc", calibration_data_min_length=4
    )

    assert [batch["input_ids"].shape[-1] for batch in batches] == [9, 6]
    assert all(batch["pixel_values"] is not None for batch in batches)
    with pytest.raises(ValueError, match=r"CalibrationConfig\.concat_size"):
        model.prepare_dataset(rows, batch_size=1, calibration_dataset_concat_size=64)


def test_qwen_drive_rejects_chat_conversations_clearly():
    model = _make_model(definitions.QwenDriveQModel)

    with pytest.raises(ValueError, match="chat conversations are not supported"):
        model.prepare_dataset([_conversation("hello")], batch_size=1)
