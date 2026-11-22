# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""VLM calibration must unpack Mapping columns instead of iterating key names (#4308).

``tokenizer(texts, padding=True)`` returns a ``BatchEncoding`` mapping whose
columns are per-row lists. The shared VLM helper used to iterate the mapping,
wrap ``"input_ids"``/``"attention_mask"`` as one-message chats, and calibrate on
those key names. These tests pin the base-helper behaviour: columnar mappings
are unpacked into rows, ``DatasetDict``-style mappings are rejected, and rows
with extra keys or NumPy/0-D-tensor token ids stay on the pre-tokenized path.
"""

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import BatchEncoding

from gptqmodel.models.definitions._vlm_calibration import split_calibration_dataset
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
from gptqmodel.models.definitions.smolvlm import SmolVLMQModel


# Definitions that build on the shared conversation helper.
SHARED_HELPER_DEFINITIONS = [
    BaseQwen2VLGPTQ,
    BaseQwen2_5_OmniGPTQ,
    InternVLQModel,
    InternVLChatQModel,
    LFM2VLQModel,
    LlavaQModel,
    MiniCPMOQModel,
    MiniCPMVQModel,
    MiniCPMV4_6QModel,
    Ovis2QModel,
    SmolVLMQModel,
]

# Gemma3 keeps its own conversation hook but now shares the row splitter.
MAPPING_DEFINITIONS = SHARED_HELPER_DEFINITIONS + [Gemma3ForConditionalGenerationGPTQ]


class _StubTokenizer:
    pad_token_id = 0
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


class _FakeProcessor:
    """Processor that records conversations and returns fixed token lengths."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.seen = []

    def apply_chat_template(self, conversations, **kwargs):
        self.seen.append(conversations)
        length = self.lengths.pop(0) if self.lengths else 4
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


def test_gemma3_batch_encoding_uses_real_token_rows(monkeypatch):
    monkeypatch.setattr(
        Gemma3ForConditionalGenerationGPTQ, "load_processor", _fail_load_processor, raising=False
    )
    model = _make_model(Gemma3ForConditionalGenerationGPTQ)
    sentences = ["The quick brown fox jumps.", "Another calibration sentence."]
    encoding = BatchEncoding(_StubTokenizer()(sentences, padding=True))
    assert encoding["input_ids"].dim() == 2

    batches = model.prepare_dataset(encoding, batch_size=1, calibration_data_min_length=10)

    decoded = [
        "".join(chr(token) for token in batch["input_ids"][0].tolist() if token)
        for batch in batches
    ]
    assert decoded == sentences


def test_gemma3_conversation_path_still_uses_processor(monkeypatch):
    processor = _FakeProcessor(lengths=[20])
    monkeypatch.setattr(
        Gemma3ForConditionalGenerationGPTQ, "load_processor", lambda self: processor, raising=False
    )
    model = _make_model(Gemma3ForConditionalGenerationGPTQ)

    batches = model.prepare_dataset([_conversation("hello")], batch_size=1)

    assert [int(batch["input_ids"].shape[-1]) for batch in batches] == [20]
    assert processor.seen == [[_conversation("hello")]]


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_columnar_mapping_columns_are_unpacked(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        {
            "input_ids": [list(range(12)), list(range(13))],
            "attention_mask": [[1] * 12, [1] * 13],
        },
        batch_size=1,
    )

    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]
    assert batches[1]["input_ids"].tolist() == [list(range(13))]


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_dataset_dict_is_rejected(cls):
    model = _make_model(cls)
    dataset_dict = DatasetDict({"train": Dataset.from_dict({"text": ["hello", "world"]})})

    with pytest.raises(ValueError, match="mapping of dataset splits"):
        model.prepare_dataset(dataset_dict, batch_size=1)


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_extra_mapping_keys_stay_on_pretokenized_path(cls, monkeypatch):
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


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_numpy_and_scalar_tensor_token_ids_are_pretokenized(cls, monkeypatch):
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)

    batches = model.prepare_dataset(
        [np.arange(12, dtype=np.int64), [torch.tensor(token) for token in range(13)]],
        batch_size=1,
    )

    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]
    assert batches[0]["input_ids"].tolist() == [list(range(12))]


def _column_values(row):
    """Normalize a pre-tokenized row's columns for comparison.

    The splitter preserves the caller's container: a mapping of lists stays
    lists, a columnar dataset column can arrive as a NumPy array, and a binary
    mask can be an integer tensor (#5486). Compare the values instead.
    """

    return {
        name: [int(value) for value in row[name]]
        for name in ("input_ids", "attention_mask")
    }


def test_split_calibration_dataset_unpacks_columnar_mapping():
    rows, conversations = split_calibration_dataset(
        {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
        },
        label="Test",
    )

    assert conversations is None
    # ``#5486`` keeps the binary-mask convention while unpacking a columnar
    # mapping, so the mask arrives as an integer tensor rather than a list.
    # Compare the values (and the row order), not the container.
    assert [_column_values(row) for row in rows] == [
        {"input_ids": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [3, 4], "attention_mask": [1, 1]},
    ]


def test_split_calibration_dataset_rejects_dataset_dict():
    dataset_dict = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})

    with pytest.raises(ValueError, match="mapping of dataset splits"):
        split_calibration_dataset(dataset_dict, label="Test")


def test_split_calibration_dataset_keeps_single_row_mapping():
    row = {"input_ids": list(range(12)), "attention_mask": [1] * 12}

    rows, conversations = split_calibration_dataset(row, label="Test")

    assert conversations is None
    assert rows == [row]


@pytest.mark.parametrize("format", ["pandas", "arrow"])
@pytest.mark.parametrize("sliced", [False, True], ids=["iterated", "sliced"])
def test_split_calibration_dataset_ignores_the_dataset_display_format(format, sliced):
    """A ``Dataset`` left in a display format still yields its rows (#4882).

    Iterating a pandas-formatted dataset yields one-row ``DataFrame`` objects
    and an arrow-formatted one yields one-row ``Table`` objects, so the
    splitter rejected every row before it could tell pre-tokenized rows from
    conversations.
    """

    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4]],
            "attention_mask": [[1, 1], [1, 1]],
        }
    )
    calibration_input = dataset.with_format(format)
    if sliced:
        calibration_input = calibration_input[:]

    rows, conversations = split_calibration_dataset(calibration_input, label="Test")

    # ``pandas`` keeps the per-row token columns as NumPy arrays, which the
    # splitter passes through unchanged; compare the ids, not the container.
    assert conversations is None
    assert [_column_values(row) for row in rows] == [
        {"input_ids": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [3, 4], "attention_mask": [1, 1]},
    ]


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_numpy_formatted_dataset_columns_are_unpacked(cls, monkeypatch):
    """`Dataset.with_format("numpy")[:]` must reach the same rows as the dataset.

    The object-dtype column of arrays was read as one token row, so the text
    path failed on tensor conversion before the VLM split even ran.
    """

    np = pytest.importorskip("numpy")
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    rows = [list(range(12)), list(range(13))]
    columns = {
        "input_ids": np.array([np.arange(len(row)) for row in rows], dtype=object),
        "attention_mask": np.array([np.ones(len(row), dtype=np.int64) for row in rows], dtype=object),
    }

    batches = model.prepare_dataset(columns, batch_size=1)

    assert [batch["input_ids"].shape for batch in batches] == [(1, 12), (1, 13)]


@pytest.mark.parametrize("cls", MAPPING_DEFINITIONS, ids=lambda cls: cls.__name__)
def test_dataframe_calibration_input_is_unpacked_into_rows(cls, monkeypatch):
    pandas = pytest.importorskip("pandas")
    monkeypatch.setattr(cls, "load_processor", _fail_load_processor, raising=False)
    model = _make_model(cls)
    rows = [list(range(12)), list(range(13))]

    batches = model.prepare_dataset(
        pandas.DataFrame({"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}),
        batch_size=1,
    )

    assert [batch["input_ids"].tolist() for batch in batches] == [[rows[0]], [rows[1]]]


def test_qwen3_vl_rejects_dataframe_calibration_input():
    """The one definition outside the shared splitter must fail closed."""

    pandas = pytest.importorskip("pandas")
    model = _make_model(BaseQwen3VLGPTQ)

    with pytest.raises(ValueError, match="not a single string, mapping, or DataFrame"):
        model.prepare_dataset(
            pandas.DataFrame({"input_ids": [[1, 2], [3, 4]]}),
            batch_size=1,
        )
