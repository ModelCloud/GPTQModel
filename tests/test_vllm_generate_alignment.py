# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`vllm_generate` must return completions where a Hugging Face caller looks.

The adapter sends vLLM the *unpadded* token rows (a vLLM token prompt is a bare
ID sequence), but a standard caller slices the returned tensor at the width of
its own ``input_ids``:

```python
output[:, input_ids.shape[1]:]
```

Reconstructing each row from vLLM's unpadded prompt echo moved the completion
left by the amount of padding that was stripped, so the caller lost the start
of the completion (or read prompt IDs) whenever a batch had unequal effective
lengths. These tests pin the returned alignment, not just input normalization.
"""

import pytest
import torch

from gptqmodel.utils import vllm as vllm_module


class _Output:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class _Result:
    def __init__(self, prompt_token_ids, completions):
        self.prompt_token_ids = list(prompt_token_ids)
        self.outputs = [_Output(completion) for completion in completions]


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1


class _FakeEngine:
    """Records the prompts vLLM would receive and echoes canned completions."""

    def __init__(self, completions=(101, 102)):
        self.completions = [list(completions)]
        self.seen_prompt_token_ids = None

    def generate(self, prompt_token_ids=None, prompts=None, sampling_params=None, **kwargs):
        assert prompt_token_ids is not None, "token prompts must reach vLLM as token IDs"
        self.seen_prompt_token_ids = [list(row) for row in prompt_token_ids]
        return [_Result(row, self.completions) for row in prompt_token_ids]

    def get_tokenizer(self):
        return _Tokenizer()


@pytest.fixture
def generate(monkeypatch):
    """Call ``vllm_generate`` without a vLLM install."""

    monkeypatch.setattr(vllm_module, "_require_vllm", lambda: None)
    monkeypatch.setattr(vllm_module, "_build_sampling_params", lambda value, kwargs: object())

    def _generate(engine, **kwargs):
        return vllm_module.vllm_generate(engine, **kwargs)

    return _generate


def test_left_padded_batch_keeps_the_completion_at_the_input_width(generate):
    engine = _FakeEngine()
    input_ids = torch.tensor([[0, 0, 11, 12], [21, 22, 23, 24]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    output = generate(engine, input_ids=input_ids, attention_mask=attention_mask, pad_token_id=0)

    # vLLM still receives the unpadded rows.
    assert engine.seen_prompt_token_ids == [[11, 12], [21, 22, 23, 24]]
    # The returned rows keep the caller's original prefix width.
    assert output.tolist() == [[0, 0, 11, 12, 101, 102], [21, 22, 23, 24, 101, 102]]
    # This is the slicing a Hugging Face caller performs.
    assert output[:, input_ids.shape[1]:].tolist() == [[101, 102], [101, 102]]


def test_right_padded_batch_keeps_the_completion_at_the_input_width(generate):
    engine = _FakeEngine()
    input_ids = torch.tensor([[21, 22, 0, 0], [31, 32, 33, 34]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    output = generate(engine, input_ids=input_ids, attention_mask=attention_mask, pad_token_id=0)

    assert engine.seen_prompt_token_ids == [[21, 22], [31, 32, 33, 34]]
    assert output.tolist() == [[21, 22, 0, 0, 101, 102], [31, 32, 33, 34, 101, 102]]
    assert output[:, input_ids.shape[1]:].tolist() == [[101, 102], [101, 102]]


def test_unpadded_batch_is_unchanged(generate):
    """Without padding there is nothing to restore; the rows are the echo."""

    engine = _FakeEngine()
    input_ids = torch.tensor([[11, 12], [21, 22]])

    output = generate(engine, input_ids=input_ids, pad_token_id=0)

    assert output.tolist() == [[11, 12, 101, 102], [21, 22, 101, 102]]
    assert output[:, input_ids.shape[1]:].tolist() == [[101, 102], [101, 102]]


def test_text_prompts_keep_the_prompt_echo(generate):
    """A text prompt has no caller-side width, so the echo is the prefix."""

    engine = _FakeEngine()
    engine.generate = lambda prompts=None, sampling_params=None, **kwargs: [
        _Result([5, 6], [[101, 102]]) for _ in prompts
    ]

    output = generate(engine, prompts=["first", "second"], pad_token_id=0)

    assert output.tolist() == [[5, 6, 101, 102], [5, 6, 101, 102]]


def test_multiple_return_sequences_share_the_restored_prefix(generate):
    engine = _FakeEngine()
    input_ids = torch.tensor([[0, 11, 12], [21, 22, 23]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

    def generate_two(prompt_token_ids=None, sampling_params=None, **kwargs):
        engine.seen_prompt_token_ids = [list(row) for row in prompt_token_ids]
        return [_Result(row, [[101, 102], [201, 202]]) for row in prompt_token_ids]

    engine.generate = generate_two

    output = generate(engine, input_ids=input_ids, attention_mask=attention_mask, pad_token_id=0)

    assert engine.seen_prompt_token_ids == [[11, 12], [21, 22, 23]]
    # Results are flattened per (prompt, return sequence); every row keeps the
    # original input width, so the completion slice is exact for all of them.
    assert output[:, input_ids.shape[1]:].tolist() == [
        [101, 102],
        [201, 202],
        [101, 102],
        [201, 202],
    ]
    assert output.tolist() == [
        [0, 11, 12, 101, 102],
        [0, 11, 12, 201, 202],
        [21, 22, 23, 101, 102],
        [21, 22, 23, 201, 202],
    ]


def test_single_prompt_token_rows_keep_the_width(generate):
    engine = _FakeEngine()
    input_ids = torch.tensor([[0, 0, 7]])

    output = generate(engine, input_ids=input_ids, attention_mask=torch.tensor([[0, 0, 1]]), pad_token_id=0)

    assert engine.seen_prompt_token_ids == [[7]]
    assert output.tolist() == [[0, 0, 7, 101, 102]]
    assert output[:, input_ids.shape[1]:].tolist() == [[101, 102]]


def test_pad_token_falls_back_to_the_tokenizer(generate):
    """Unequal row widths are padded with the tokenizer's pad ID."""

    engine = _FakeEngine()
    input_ids = [[0, 0, 7], [21, 22, 23, 24]]
    attention_mask = [[0, 0, 1], [1, 1, 1, 1]]

    output = generate(engine, input_ids=input_ids, attention_mask=attention_mask)

    assert engine.seen_prompt_token_ids == [[7], [21, 22, 23, 24]]
    assert output.tolist() == [[0, 0, 7, 101, 102, 0], [21, 22, 23, 24, 101, 102]]
