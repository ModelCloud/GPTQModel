# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A `messages` column survives every HF `datasets` display format (#4936).

#4902 made `text`/`input_ids` columns reach the same token rows in every
format, but a `messages` column sliced from a numpy- or pandas-formatted
dataset was still read as one row: `with_format("numpy")[:]` stores it as an
`ndarray` (2-D object array of turns when every conversation has the same
length, else a 1-D object array of conversations) and pandas as a `Series` of
arrays. `_is_columnar_mapping` required `list`/`tuple`, so the whole column
went to `apply_chat_template` as a single "conversation" and every row
rendered only the template's default system turn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.calibration import prepare_calibration_dataset


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# Ragged conversations: numpy stores the column as a 1-D object array. Every
# message keeps a multi-part `content` list: Arrow (and therefore
# `Dataset.from_dict`) cannot mix list and string values in one field.
CONVERSATIONS = [
    [
        {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Weather in Paris?"},
                {"type": "text", "text": "And in Rome?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Paris is mild."}]},
        {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
    ],
    [
        {"role": "user", "content": [{"type": "text", "text": "Name three colors."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Red, green, blue."}]},
    ],
    [
        {"role": "system", "content": [{"type": "text", "text": "Answer in one word."}]},
        {"role": "user", "content": [{"type": "text", "text": "Capital of France?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Paris."}]},
        {"role": "user", "content": [{"type": "text", "text": "And of Italy?"}]},
    ],
]

# A single conversation that mixes plain string content and part lists.
STRING_CONVERSATION = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
    {"role": "assistant", "content": "Mild."},
]

# Qwen2.5's chat template concatenates `content` as a string, so the real
# tokenizer test uses plain string turns (the part-list shape is covered above).
STRING_CONVERSATIONS = [
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "Paris is mild."},
    ],
    [
        {"role": "user", "content": "Name three colors."},
        {"role": "assistant", "content": "Red, green, blue."},
    ],
    [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Italy?"},
    ],
]

# Equal-length conversations: numpy stores a 2-D object array of turns.
EQUAL_LENGTH_CONVERSATIONS = [
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"Question number {index}?"},
        {"role": "assistant", "content": "Sure."},
    ]
    for index in range(4)
]


class _RecordingChatTokenizer:
    """A deterministic stand-in for a chat template.

    It renders role/content pairs (including multi-part `content` lists) and
    records the prompt string, so tests can compare the exact prompt every
    input shape produces. A value that is not a plain role mapping renders as
    `<unrenderable>`, which is what a real Jinja template does when handed an
    `ndarray` of conversations.
    """

    chat_template = "recording"
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.rendered = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        parts = []
        for message in messages:
            if not isinstance(message, Mapping):
                parts.append("<unrenderable>")
                continue
            content = message["content"]
            if isinstance(content, str):
                text = content
            else:
                text = "".join(
                    part["text"] for part in content if part.get("type") == "text"
                )
            parts.append(f"<{message['role']}>{text}</{message['role']}>")
        for key in ("tools", "documents"):
            if key in kwargs:
                parts.append(
                    f"<{key}>{json.dumps(kwargs[key], sort_keys=True, default=repr)}</{key}>"
                )
        rendered = "".join(parts)
        self.rendered.append(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=True, return_tensors="pt", **kwargs):
        assert isinstance(text, str)
        tokens = max(1, len(text.split()))
        ids = torch.arange(1, tokens + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(token)) for token in ids)


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=4096),
        ),
    )


def _prepare(tokenizer, calibration_input):
    return prepare_calibration_dataset(
        _qmodel(tokenizer),
        calibration_input,
        batch_size=1,
        calibration_data_min_length=0,
    )


def _token_rows(prepared):
    return [
        (
            batch["input_ids"].tolist(),
            batch["attention_mask"].tolist(),
        )
        for batch in prepared
    ]


def _dataset(conversations, with_tools):
    datasets = pytest.importorskip("datasets")
    columns = {"messages": [list(conversation) for conversation in conversations]}
    if with_tools:
        columns["tools"] = [TOOLS for _ in conversations]
    return datasets.Dataset.from_dict(columns)


@pytest.mark.parametrize("format", ["numpy", "pandas", "torch", "arrow"])
def test_messages_column_slices_render_like_plain_rows(format):
    dataset = _dataset(CONVERSATIONS, with_tools=True)
    reference_tokenizer = _RecordingChatTokenizer()
    reference = _prepare(
        reference_tokenizer,
        [
            {"messages": conversation, "tools": TOOLS}
            for conversation in CONVERSATIONS
        ],
    )

    tokenizer = _RecordingChatTokenizer()
    prepared = _prepare(tokenizer, dataset.with_format(format)[:])

    assert tokenizer.rendered == reference_tokenizer.rendered
    assert _token_rows(prepared) == _token_rows(reference)


def test_numpy_two_dimensional_object_column_renders_every_conversation():
    np = pytest.importorskip("numpy")
    dataset = _dataset(EQUAL_LENGTH_CONVERSATIONS, with_tools=False)
    numpy_column = dataset.with_format("numpy")[:]["messages"]
    assert isinstance(numpy_column, np.ndarray)
    assert numpy_column.ndim == 2

    reference_tokenizer = _RecordingChatTokenizer()
    reference = _prepare(
        reference_tokenizer,
        [{"messages": conversation} for conversation in EQUAL_LENGTH_CONVERSATIONS],
    )

    tokenizer = _RecordingChatTokenizer()
    prepared = _prepare(tokenizer, dataset.with_format("numpy")[:])

    assert tokenizer.rendered == reference_tokenizer.rendered
    assert _token_rows(prepared) == _token_rows(reference)


@pytest.mark.parametrize("format", ["numpy", "pandas"])
def test_plain_and_formatted_iteration_render_identically(format):
    dataset = _dataset(CONVERSATIONS, with_tools=True)

    plain_tokenizer = _RecordingChatTokenizer()
    plain = _prepare(plain_tokenizer, list(dataset))

    formatted_tokenizer = _RecordingChatTokenizer()
    formatted = _prepare(formatted_tokenizer, dataset.with_format(format))

    assert formatted_tokenizer.rendered == plain_tokenizer.rendered
    assert _token_rows(formatted) == _token_rows(plain)


def test_slice_is_not_mutated_by_calibration():
    np = pytest.importorskip("numpy")
    dataset = _dataset(CONVERSATIONS, with_tools=True)
    sliced = dataset.with_format("numpy")[:]
    messages_column = sliced["messages"]
    tools_column = sliced["tools"]

    _prepare(_RecordingChatTokenizer(), sliced)

    # The caller's slice keeps the same numpy columns: normalization copies.
    assert sliced["messages"] is messages_column
    assert sliced["tools"] is tools_column
    assert isinstance(sliced["messages"], np.ndarray)
    assert isinstance(sliced["messages"][0], np.ndarray)
    assert isinstance(sliced["tools"], np.ndarray)


def test_documents_column_survives_a_numpy_slice():
    """`documents` is a template input like `tools`, not a calibration column."""

    datasets = pytest.importorskip("datasets")
    documents = [{"title": "weather", "text": "Paris is mild in June."}]
    dataset = datasets.Dataset.from_dict(
        {
            "messages": [list(conversation) for conversation in CONVERSATIONS],
            "documents": [documents for _ in CONVERSATIONS],
        }
    )

    reference_tokenizer = _RecordingChatTokenizer()
    reference = _prepare(
        reference_tokenizer,
        [
            {"messages": conversation, "documents": documents}
            for conversation in CONVERSATIONS
        ],
    )

    tokenizer = _RecordingChatTokenizer()
    prepared = _prepare(tokenizer, dataset.with_format("numpy")[:])

    assert "<documents>" in tokenizer.rendered[0]
    assert tokenizer.rendered == reference_tokenizer.rendered
    assert _token_rows(prepared) == _token_rows(reference)


def test_single_row_messages_mapping_is_still_one_row():
    tokenizer = _RecordingChatTokenizer()

    prepared = _prepare(tokenizer, {"messages": STRING_CONVERSATION})

    assert len(prepared) == 1
    assert tokenizer.rendered == [
        "<system>Be terse.</system><user>Weather?</user><assistant>Mild.</assistant>"
    ]


def test_reference_rendering_keeps_the_multi_part_content():
    tokenizer = _RecordingChatTokenizer()

    prepared = _prepare(tokenizer, [{"messages": CONVERSATIONS[0], "tools": TOOLS}])

    assert len(prepared) == 1
    assert tokenizer.rendered[0].startswith(
        "<system>You are terse.</system><user>Weather in Paris?And in Rome?</user>"
    )
    assert "<tools>" in tokenizer.rendered[0]


def test_qwen2_5_messages_column_slices_match_plain_rows():
    """Real tokenizer: numpy/pandas slices produce the same prompt as plain rows."""

    datasets = pytest.importorskip("datasets")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
    except Exception:
        pytest.skip("Qwen2.5 tokenizer is not available offline")

    dataset = datasets.Dataset.from_dict(
        {"messages": [list(conversation) for conversation in STRING_CONVERSATIONS]}
    )
    qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=32768),
        ),
    )

    def prepare(calibration_input):
        return prepare_calibration_dataset(
            qmodel, calibration_input, batch_size=1, calibration_data_min_length=0
        )

    reference = prepare(
        [{"messages": conversation} for conversation in STRING_CONVERSATIONS]
    )
    expected_ids = [
        tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False
        )
        for conversation in STRING_CONVERSATIONS
    ]
    expected_ids = [
        list(ids["input_ids"]) if isinstance(ids, Mapping) else list(ids)
        for ids in expected_ids
    ]
    assert [batch["input_ids"][0].tolist() for batch in reference] == expected_ids

    for format in ("numpy", "pandas"):
        prepared = prepare(dataset.with_format(format)[:])
        assert [batch["input_ids"][0].tolist() for batch in prepared] == expected_ids
