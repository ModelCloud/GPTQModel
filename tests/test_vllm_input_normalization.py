# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""vLLM token-prompt normalization must not silently rewrite prompts (#3884).

A vLLM token prompt is a bare ID sequence. A Hugging Face attention mask that
removes an interior token cannot be represented there: deleting the token
changes the prompt IDs, length, and positions. Only contiguous left/right
padding removal is a faithful translation, and a mask supplied with a text
prompt has no meaning at all.
"""

import pytest
import torch

from gptqmodel.utils.vllm import _normalize_vllm_inputs


def test_left_padding_is_removed():
    prompts, ids = _normalize_vllm_inputs(None, [0, 0, 10, 20, 30], [0, 0, 1, 1, 1])

    assert prompts is None
    assert ids == [[10, 20, 30]]


def test_right_padding_is_removed():
    _, ids = _normalize_vllm_inputs(None, [10, 20, 30, 0, 0], [1, 1, 1, 0, 0])

    assert ids == [[10, 20, 30]]


def test_batched_padding_is_removed_per_row():
    _, ids = _normalize_vllm_inputs(
        None,
        [[0, 10, 20], [30, 40, 0]],
        torch.tensor([[0, 1, 1], [1, 1, 0]]),
    )

    assert ids == [[10, 20], [30, 40]]


def test_interior_hole_is_rejected():
    with pytest.raises(ValueError, match="inside a prompt"):
        _normalize_vllm_inputs(None, [10, 20, 30], [1, 0, 1])


def test_non_binary_mask_is_rejected():
    with pytest.raises(TypeError, match="boolean or 0/1"):
        _normalize_vllm_inputs(None, [10, 20, 30], [1, 2, 1])


def test_float_mask_is_rejected():
    with pytest.raises(TypeError, match="boolean or 0/1"):
        _normalize_vllm_inputs(None, [10, 20], [1.0, 1.0])


def test_mask_removing_every_token_is_rejected():
    with pytest.raises(ValueError, match="removed every token"):
        _normalize_vllm_inputs(None, [10, 20], [0, 0])


def test_mask_with_text_prompt_is_rejected():
    with pytest.raises(ValueError, match="only supported with `input_ids`"):
        _normalize_vllm_inputs("hello", None, [1, 0])
