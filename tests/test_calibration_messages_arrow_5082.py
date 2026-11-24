# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#5082: Arrow-unified `messages` render like the plain Python rows.

#4997 normalized the `tools`/`documents` columns, but the `messages` column of
the same rows kept Arrow's unified struct schema. Arrow fills every field a
turn does not use with ``None``:

* a tool call's ``arguments`` mapping gains the other rows' argument names with
  ``None`` values (``{"city": "Paris", "query": null}``), so the Qwen family
  silently calibrates on tokens inference never emits;
* every turn gains ``tool_calls: None`` (and every other key some turn has), so
  a template that tests ``'tool_calls' in message`` (Llama-3.x) renders a user
  turn as a tool call and raises ``TypeError: object of type 'NoneType' has no
  len()``.

The fix drops those injected ``None`` entries before the template runs, for
every Arrow-backed input shape, and leaves a real ``content: None`` (which
templates such as DeepSeek-R1 test with ``is none``) alone. ``arguments`` that
is already a JSON string stays a string: templates such as DeepSeek-R1 insert
it verbatim into a ``json`` block, so decoding it would corrupt the prompt.

Transformers 5.23 makes the DeepSeek-R1 conversation unreachable through
``apply_chat_template``: before rendering, ``tokenization_utils_base`` rebuilds
every message as ``{k: v for k, v in msg.items() if v is not None}``, so a
tool-call turn's real ``content: None`` is deleted. The published DeepSeek-R1
template then falls from its ``content is none`` branch into its text branch,
where its ``in`` test on the now-undefined ``content`` raises
``UndefinedError: 'dict object' has no attribute 'content'``. The template
itself is fine: rendering the same rows through it directly succeeds and
inserts the JSON string verbatim. Because calibration renders exactly like
inference does (``apply_chat_template``), the JSON-string contract below is
pinned by (a) rendering the published template directly and (b) calibrating a
template that does accept the row.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gptqmodel.utils.calibration import prepare_calibration_dataset


QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA = "unsloth/Llama-3.2-1B-Instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def _tool(name, argument):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {argument: {"type": "string"}},
                    "required": [argument],
                },
            },
        }
    ]


# Two tool-call rows with different argument names (Arrow unifies their struct
# schema) plus one plain row (every turn gains `tool_calls: None`). Deliberately
# different lengths so a wrong row cannot match by accident.
ROWS = [
    {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "18C"},
            {"role": "assistant", "content": "18C in Paris."},
        ],
        "tools": _tool("get_weather", "city"),
    },
    {
        "messages": [
            {"role": "user", "content": "Tallest mountain?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "tallest mountain"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Everest"},
            {"role": "assistant", "content": "Everest."},
        ],
        "tools": _tool("search", "query"),
    },
    {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello."},
        ],
        "tools": _tool("get_weather", "city"),
    },
]


def _tokenizer(repo_id):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"{repo_id} tokenizer is not cached locally: {type(exc).__name__}")


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096)),
    )


def _prepare(tokenizer, calibration_input):
    return prepare_calibration_dataset(
        _qmodel(tokenizer),
        calibration_input,
        batch_size=1,
        calibration_data_min_length=0,
    )


def _prepared_ids(prepared):
    return [row["input_ids"][0].tolist() for row in prepared]


def _reference_texts(tokenizer, rows):
    """What inference sends: the conversation as a plain Python list."""

    return [
        tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for row in rows
    ]


def _reference_ids(tokenizer, rows):
    return [
        tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in _reference_texts(tokenizer, rows)
    ]


def _render_published_template(tokenizer, messages, tools=None):
    """Render with the tokenizer's published template, minus HF's ``None`` strip.

    ``transformers`` 5.23's ``apply_chat_template`` deletes every message key
    whose value is ``None`` before compiling the template
    (``tokenization_utils_base``), which removes a tool-call turn's real
    ``content: None``. ``render_jinja_template`` is the same renderer without
    that pre-normalization, so it shows what the published template produces.
    """

    from transformers.utils.chat_template_utils import render_jinja_template

    chat_template = tokenizer.get_chat_template(None, tools)
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=tools,
        documents=None,
        add_generation_prompt=False,
        chat_template=chat_template,
        **tokenizer.special_tokens_map,
    )
    return rendered[0]


def _dataset(rows):
    datasets = pytest.importorskip(
        "datasets", reason="the HF `datasets` package is not installed"
    )
    return datasets.Dataset.from_list(rows)


def _assert_arrow_injected_nulls(dataset):
    """Guard the fixture: without Arrow's unified schema there is no defect."""

    first_turn = dataset[0]["messages"][0]
    first_arguments = dataset[0]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    if first_turn.get("tool_calls") is None and "query" not in first_arguments:
        pytest.skip(
            "this `datasets`/Arrow build did not unify the per-row `messages` struct "
            f"schemas (first turn: {first_turn!r}); nothing to reproduce"
        )
    assert first_turn["tool_calls"] is None
    assert first_arguments == {"city": "Paris", "query": None}


@pytest.mark.parametrize("repo_id", [QWEN, LLAMA], ids=["qwen2.5", "llama3.2"])
def test_hf_dataset_messages_match_the_plain_rows(repo_id):
    tokenizer = _tokenizer(repo_id)
    dataset = _dataset(ROWS)
    _assert_arrow_injected_nulls(dataset)

    expected = _reference_ids(tokenizer, ROWS)
    assert len({len(ids) for ids in expected}) == len(expected), "pick distinct rows"

    assert _prepared_ids(_prepare(tokenizer, dataset)) == expected
    # The same rows as a Python list keep rendering identically.
    assert _prepared_ids(_prepare(tokenizer, ROWS)) == expected


@pytest.mark.parametrize("shape", ["dataset", "slice", "pandas", "numpy", "iter"])
def test_every_arrow_backed_input_shape_matches_the_plain_rows(shape):
    tokenizer = _tokenizer(QWEN)
    dataset = _dataset(ROWS)
    _assert_arrow_injected_nulls(dataset)

    if shape == "dataset":
        calibration_input = dataset
    elif shape == "slice":
        calibration_input = dataset[:]
    elif shape == "pandas":
        calibration_input = dataset.to_pandas()
    elif shape == "numpy":
        calibration_input = dataset.with_format("numpy")[:]
    else:
        calibration_input = list(dataset)

    assert _prepared_ids(_prepare(tokenizer, calibration_input)) == _reference_ids(
        tokenizer, ROWS
    )


def test_llama3_2_dataset_does_not_raise_on_tool_calls_none():
    """The crash half of #5082: a user turn must not render as a tool call."""

    tokenizer = _tokenizer(LLAMA)
    dataset = _dataset(ROWS)
    _assert_arrow_injected_nulls(dataset)

    texts = _prepare_texts(tokenizer, dataset)

    # Same prompts as the plain rows, and the tool call renders as one call
    # rather than the user turn being taken for one.
    assert texts == _reference_texts(tokenizer, ROWS)
    assert texts[0].count('{"name": "get_weather", "parameters": {"city": "Paris"}}') == 1
    assert texts[1].count('{"name": "search", "parameters": {"query": "tallest mountain"}}') == 1


def _prepare_texts(tokenizer, calibration_input):
    """Render through calibration and return the decoded prompt per row."""

    prepared = _prepare(tokenizer, calibration_input)
    return [
        tokenizer.decode(
            row["input_ids"][0][row["attention_mask"][0].bool()],
            skip_special_tokens=False,
        )
        for row in prepared
    ]


def test_json_string_arguments_stay_verbatim_for_raw_text_templates():
    """DeepSeek-R1 inserts `arguments` into a json block; it is not an object."""

    # Imported here so a pre-fix run reaches the behavioural assertions above
    # instead of failing collection on the missing symbol.
    from gptqmodel.utils.calibration import _normalize_chat_messages

    tokenizer = _tokenizer(DEEPSEEK)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "18C"},
                {"role": "assistant", "content": "18C in Paris."},
            ],
            "tools": _tool("get_weather", "city"),
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello."},
            ],
            "tools": _tool("get_weather", "city"),
        },
    ]
    dataset = _dataset(rows)
    assert isinstance(
        dataset[0]["messages"][1]["tool_calls"][0]["function"]["arguments"], str
    ), "fixture must store `arguments` as JSON text"

    # Arrow injected the padding this test exists for: turns without a tool
    # call carry `tool_calls: None`.
    assert dataset[0]["messages"][0]["tool_calls"] is None
    assert dataset[0]["messages"][2]["tool_calls"] is None

    # Normalization drops only that padding: the JSON string stays a string and
    # the tool-call turn's real `content: None` survives. Decoding `arguments`
    # into an object would change the row.
    normalized = _normalize_chat_messages(dataset[0]["messages"])
    assert normalized == rows[0]["messages"]

    # The published template renders that row verbatim: the JSON text lands in
    # the ```json block, which an object (or any re-serialization) would not.
    rendered = _render_published_template(
        tokenizer, normalized, tools=rows[0]["tools"]
    )
    assert '```json\n{"city": "Paris"}\n```' in rendered

    # It is not renderable through `apply_chat_template` on transformers 5.23:
    # that call deletes every message key whose value is None before rendering,
    # which is exactly the `content: None` that marks a tool-call turn. The
    # template falls into its text branch and raises on `'</think' in content`.
    import jinja2

    with pytest.raises(jinja2.exceptions.UndefinedError, match="content"):
        tokenizer.apply_chat_template(
            rows[0]["messages"],
            tools=rows[0]["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # Calibration renders through that same API, so it raises for this row too
    # rather than inventing tokens inference cannot emit. Rows the template does
    # accept (a tool output) still calibrate to exactly the plain Python rows.
    tool_rows = [
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "tool", "content": "18C"},
                {"role": "assistant", "content": "18C in Paris."},
            ],
            "tools": _tool("get_weather", "city"),
        }
    ]
    tool_dataset = _dataset(tool_rows)
    assert _prepared_ids(_prepare(tokenizer, tool_dataset)) == _reference_ids(
        tokenizer, tool_rows
    )

    # A template that does accept tool calls pins the string end to end: a
    # decoded mapping renders as a JSON object and changes every token.
    llama = _tokenizer(LLAMA)
    string_rows = [
        {
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "18C"},
                {"role": "assistant", "content": "18C in Paris."},
            ],
            "tools": _tool("get_weather", "city"),
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello."},
            ],
            "tools": _tool("get_weather", "city"),
        },
    ]
    string_dataset = _dataset(string_rows)
    assert isinstance(
        string_dataset[0]["messages"][1]["tool_calls"][0]["function"]["arguments"],
        str,
    ), "fixture must store `arguments` as JSON text"
    # Llama-3.2 renders a user turn as a tool call when `tool_calls: None`
    # survives, so this equality also proves the padding was dropped.
    assert string_dataset[0]["messages"][0]["tool_calls"] is None
    assert _prepared_ids(_prepare(llama, string_dataset)) == _reference_ids(
        llama, string_rows
    )
    string_rendered = _prepare_texts(llama, string_dataset)[0]
    assert '"parameters": "{\\"city\\": \\"Paris\\"}"' in string_rendered
    assert '"parameters": {"city": "Paris"}' not in string_rendered


def test_normalize_chat_messages_keeps_content_none_and_does_not_mutate():
    # Imported here so a pre-fix run reaches the behavioural assertions above
    # instead of failing collection on the missing symbol.
    from gptqmodel.utils.calibration import _normalize_chat_messages

    conversation = [
        {"role": "user", "content": "hi", "tool_calls": None},
        {
            "role": "assistant",
            "content": None,
            "name": None,
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": "f",
                        "arguments": {"city": "Paris", "query": None},
                    },
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]

    normalized = _normalize_chat_messages(conversation)

    # `tool_calls: None` and the injected argument names are Arrow padding.
    assert normalized[0] == {"role": "user", "content": "hi"}
    assert normalized[1]["tool_calls"][0] == {
        "type": "function",
        "function": {"name": "f", "arguments": {"city": "Paris"}},
    }
    # `content: None` is a real value for a tool-call turn (DeepSeek-R1 tests
    # `message.content is none`), and a `None` inside a list is a JSON value.
    assert normalized[1]["content"] is None
    assert _normalize_chat_messages(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, None]}]
    ) == [{"role": "user", "content": [{"type": "text", "text": "a"}, None]}]

    # The caller's conversation is untouched.
    assert conversation[0]["tool_calls"] is None
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Paris",
        "query": None,
    }
    # A JSON string is not decoded: raw-text templates insert it verbatim.
    assert _normalize_chat_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            }
        ]
    ) == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
        }
    ]
