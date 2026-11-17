# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4934: real HF tool-column layouts render like inference.

#4927 forwarded a row's ``tools`` to ``apply_chat_template`` exactly as stored,
which misses the two layouts function-calling data on the Hub actually uses:

* ``tools`` as a JSON string (Arrow cannot hold a per-row struct schema), which
  the template rejects, so #4927's fallback rendered the row *without* the
  schema (#4858 again);
* ``tools`` as a list of dicts in an HF ``Dataset``, where Arrow unifies the
  struct schema across rows and injects the other rows' parameter names as
  ``null`` (``"properties": {"city": {...}, "tz": null}``), silently changing
  the rendered tool block.

``prepare_calibration_dataset`` now decodes the string form and drops the
injected ``None`` entries, so every layout tokenizes exactly like
``apply_chat_template(messages, tools=<original list>)``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gptqmodel.utils.calibration import prepare_calibration_dataset


MESSAGES = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "Sunny."},
]

# Deliberately different lengths: a row that renders the *other* row's tool (or
# a schema with an injected null) cannot match its reference by accident.
TOOLS_WEATHER = [
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
TOOLS_TIME = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the time",
            "parameters": {
                "type": "object",
                "properties": {"tz": {"type": "string"}},
                "required": ["tz"],
            },
        },
    }
]


def _qwen2_5_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
        )
    except Exception as exc:
        pytest.skip(
            "Qwen2.5-0.5B-Instruct tokenizer is not cached locally; "
            f"`AutoTokenizer.from_pretrained(..., local_files_only=True)` raised "
            f"{type(exc).__name__}: {exc}"
        )


def _hf_dataset():
    datasets = pytest.importorskip(
        "datasets", reason="the HF `datasets` package is not installed"
    )
    return datasets.Dataset


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096)),
    )


def _prepare(tokenizer, dataset):
    return prepare_calibration_dataset(
        _qmodel(tokenizer),
        dataset,
        batch_size=1,
        calibration_data_min_length=0,
    )


def _reference_ids(tokenizer, tools):
    """What inference sends for ``MESSAGES`` plus ``tools``."""

    text = tokenizer.apply_chat_template(
        MESSAGES, tools=tools, tokenize=False, add_generation_prompt=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _prepared_ids(prepared):
    return [row["input_ids"][0].tolist() for row in prepared]


def test_hf_dataset_with_different_tool_schemas_matches_inference():
    Dataset = _hf_dataset()
    tokenizer = _qwen2_5_tokenizer()

    dataset = Dataset.from_list(
        [
            {"messages": MESSAGES, "tools": TOOLS_WEATHER},
            {"messages": MESSAGES, "tools": TOOLS_TIME},
        ]
    )
    # Guard the fixture: if Arrow did not unify the two struct schemas there is
    # no injected null to strip and the test would not exercise the defect.
    properties = dataset[0]["tools"][0]["function"]["parameters"]["properties"]
    if "tz" not in properties or properties["tz"] is not None:
        pytest.skip(
            "this `datasets`/Arrow build did not unify the per-row tool struct "
            f"schemas (row 0 properties: {properties!r}); nothing to reproduce"
        )

    prepared = _prepare(tokenizer, dataset)
    expected = [
        _reference_ids(tokenizer, TOOLS_WEATHER),
        _reference_ids(tokenizer, TOOLS_TIME),
    ]
    assert len(expected[0]) != len(expected[1]), "pick tools with different token lengths"

    assert _prepared_ids(prepared) == expected


def test_hf_dataset_with_json_string_tools_matches_inference():
    Dataset = _hf_dataset()
    tokenizer = _qwen2_5_tokenizer()

    dataset = Dataset.from_list(
        [
            {"messages": MESSAGES, "tools": json.dumps(TOOLS_WEATHER)},
            {"messages": MESSAGES, "tools": json.dumps(TOOLS_TIME)},
        ]
    )
    assert isinstance(dataset[0]["tools"], str), "fixture must store `tools` as JSON text"

    prepared = _prepare(tokenizer, dataset)
    expected = [
        _reference_ids(tokenizer, TOOLS_WEATHER),
        _reference_ids(tokenizer, TOOLS_TIME),
    ]

    assert _prepared_ids(prepared) == expected


def test_plain_list_rows_with_json_string_tools_match_inference():
    """The issue's list-of-rows repro, without an HF ``Dataset`` in between."""

    tokenizer = _qwen2_5_tokenizer()

    prepared = _prepare(
        tokenizer,
        [
            {"messages": MESSAGES, "tools": json.dumps(TOOLS_WEATHER)},
            {"messages": MESSAGES, "tools": json.dumps(TOOLS_TIME)},
        ],
    )
    expected = [
        _reference_ids(tokenizer, TOOLS_WEATHER),
        _reference_ids(tokenizer, TOOLS_TIME),
    ]

    assert _prepared_ids(prepared) == expected


def test_null_stripping_keeps_a_real_null_schema_and_does_not_mutate():
    from gptqmodel.utils.calibration import _normalize_template_input_value

    # `{"type": "null"}` is the JSON-schema spelling of a nullable value; a
    # list element may legitimately be JSON null. Only dict entries whose value
    # is Python ``None`` are Arrow's injected padding.
    schema = {
        "type": "object",
        "properties": {
            "nothing": {"type": "null"},
            "choice": {"enum": [None, "x"]},
            "injected": None,
        },
        "required": ["nothing"],
    }

    normalized = _normalize_template_input_value(schema)

    assert normalized == {
        "type": "object",
        "properties": {
            "nothing": {"type": "null"},
            "choice": {"enum": [None, "x"]},
        },
        "required": ["nothing"],
    }
    # The caller's row is untouched.
    assert schema["properties"]["injected"] is None

    # A JSON string is decoded before the same walk.
    assert _normalize_template_input_value(
        json.dumps([{"name": "f", "parameters": {"properties": {"a": {}, "b": None}}}])
    ) == [{"name": "f", "parameters": {"properties": {"a": {}}}}]


def test_non_json_text_is_forwarded_unchanged():
    """A document that merely looks like text must not be parsed away."""
    from gptqmodel.utils.calibration import _normalize_template_input_value

    for value in ("the weather in Paris", "42", "true", "", "{not json"):
        assert _normalize_template_input_value(value) == value
