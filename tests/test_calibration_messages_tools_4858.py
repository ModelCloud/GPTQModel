# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4858: a `messages` calibration row keeps its `tools`.

The standard HF tool-calling layout is `{"messages": [...], "tools": [...]}`.
`_tokenize_messages_value` rendered only `messages`, so the tool schema block
the model sees at inference never reached the Hessian: on Qwen2.5-0.5B a
one-tool row produced 101 tokens instead of 236.
"""

from __future__ import annotations

import json
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
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "You answer weather questions precisely."},
    {"role": "user", "content": "What is the weather in Paris today?"},
]


class _ToolAwareTokenizer:
    """Renders `tools`/`documents` into the text, like a real chat template."""

    chat_template = "{{ messages }}"
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, reject_template_inputs: bool = False):
        self.reject_template_inputs = reject_template_inputs
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(dict(kwargs))
        if self.reject_template_inputs and kwargs:
            raise ValueError("this chat template does not use tools")
        text = " ".join(message["content"] for message in messages)
        for key in ("tools", "documents"):
            if key in kwargs:
                text += f" {key.upper()} {json.dumps(kwargs[key])}"
        return text

    def __call__(self, text, add_special_tokens=True, return_tensors="pt", **kwargs):
        tokens = max(1, len(str(text).split()))
        ids = torch.arange(1, tokens + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(token)) for token in ids)


def _qmodel(tokenizer):
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096)),
    )


def _prepare(tokenizer, row, **kwargs):
    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("calibration_data_min_length", 0)
    return prepare_calibration_dataset(_qmodel(tokenizer), [row], **kwargs)


def test_tools_are_forwarded_to_the_chat_template():
    tokenizer = _ToolAwareTokenizer()

    plain = _prepare(tokenizer, {"messages": MESSAGES})
    with_tools = _prepare(tokenizer, {"messages": MESSAGES, "tools": TOOLS})

    assert tokenizer.calls[-1]["tools"] == TOOLS
    assert with_tools[0]["input_ids"].shape[-1] > plain[0]["input_ids"].shape[-1]


def test_documents_are_forwarded_to_the_chat_template():
    tokenizer = _ToolAwareTokenizer()
    documents = [{"title": "doc", "text": "the weather in Paris"}]

    prepared = _prepare(tokenizer, {"messages": MESSAGES, "documents": documents})

    assert tokenizer.calls[-1]["documents"] == documents
    assert prepared[0]["input_ids"].numel() > 0


def test_a_row_without_template_inputs_is_unchanged():
    tokenizer = _ToolAwareTokenizer()

    _prepare(tokenizer, {"messages": MESSAGES})

    assert tokenizer.calls[-1] == {}


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def info(self, message, *args, **kwargs):
        pass

    def warn(self, message, *args, **kwargs):
        self.warnings.append(message % args if args else message)

    warning = warn


def test_a_template_that_rejects_tools_keeps_the_row_and_warns():
    tokenizer = _ToolAwareTokenizer(reject_template_inputs=True)
    logger = _RecordingLogger()

    prepared = _prepare(tokenizer, {"messages": MESSAGES, "tools": TOOLS}, logger=logger)

    assert prepared[0]["input_ids"].numel() > 0
    assert any("rejected the calibration row" in message for message in logger.warnings)


def test_qwen2_5_renders_the_tool_schema():
    """Real tokenizer: the rendered row matches `apply_chat_template(tools=...)`."""

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer is not available offline")

    prepared = _prepare(tokenizer, {"messages": MESSAGES, "tools": TOOLS})
    ids = prepared[0]["input_ids"][0].tolist()

    expected = tokenizer.apply_chat_template(
        MESSAGES, tools=TOOLS, tokenize=True, add_generation_prompt=False
    )
    # ``tokenize=True`` returns a ``BatchEncoding`` (a ``UserDict``, not a dict).
    expected_ids = expected["input_ids"] if "input_ids" in expected else expected
    assert ids == list(expected_ids)
