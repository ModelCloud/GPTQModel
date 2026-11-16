# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounded MMLU-Pro prompt-length retry (issue #3876)."""

from __future__ import annotations

import pytest
import torch
from transformers import BatchEncoding

from gptqmodel.utils import mmlupro


class _FixedTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, length: int | None = None, return_tensors: bool = False):
        self.length = length
        self.return_tensors = return_tensors

    def __call__(self, prompt, return_tensors=None):
        length = self.length if self.length is not None else len(prompt)
        if return_tensors or self.return_tensors:
            return BatchEncoding(
                {"input_ids": torch.zeros((1, length), dtype=torch.long)}
            )
        # Real Hugging Face tokenizers return a Python list of token IDs by default.
        return BatchEncoding({"input_ids": [0] * length})


def _identity_cuda(self, *args, **kwargs):
    """Keep the length probe on CPU regardless of the host."""
    return self


def test_eval_cot_stops_at_zero_shot_and_raises(monkeypatch, tmp_path):
    seen = []

    def fake_prompt(val_df, curr, k):
        # Fail fast instead of hanging forever on the pre-fix retry loop.
        assert k >= 0, f"negative few-shot count {k}"
        seen.append(k)
        return "over budget"

    monkeypatch.setattr(mmlupro, "generate_cot_prompt", fake_prompt)
    monkeypatch.setattr(torch.Tensor, "cuda", _identity_cuda)

    with pytest.raises(ValueError, match="zero few-shot"):
        mmlupro.eval_cot(
            "math",
            model=None,
            tokenizer=_FixedTokenizer(2050),
            val_df=[],
            test_df=[{}],
            output_path=str(tmp_path / "result.json"),
            ntrain=5,
            batch_size=1,
        )

    # Five shots down to zero, never negative.
    assert seen == [5, 4, 3, 2, 1, 0]


def test_eval_cot_stops_decreasing_once_the_prompt_fits(monkeypatch, tmp_path):
    seen = []

    def fake_prompt(val_df, curr, k):
        seen.append(k)
        # Length is 10*k + 5, so k=2 is the first that fits the 30-token budget.
        return "x" * (10 * k + 5)

    monkeypatch.setattr(mmlupro, "generate_cot_prompt", fake_prompt)
    monkeypatch.setattr(torch.Tensor, "cuda", _identity_cuda)
    monkeypatch.setattr(mmlupro, "max_model_length", 30)
    monkeypatch.setattr(mmlupro, "max_new_tokens", 0)
    monkeypatch.setattr(mmlupro, "batch_inference", lambda *args, **kwargs: (["A"], ["Answer: A"]))
    monkeypatch.setattr(mmlupro, "save_res", lambda res, path: (1.0, 1.0, 0.0))

    accuracy, correct, wrong = mmlupro.eval_cot(
        "math",
        model=None,
        tokenizer=_FixedTokenizer(),
        val_df=[],
        test_df=[{}],
        output_path=str(tmp_path / "result.json"),
        ntrain=5,
        batch_size=1,
    )

    assert seen == [5, 4, 3, 2]
    assert (accuracy, correct, wrong) == (1.0, 1.0, 0.0)


@pytest.mark.parametrize("return_tensors", [False, True])
def test_eval_cot_supports_both_list_and_tensor_tokenizers(monkeypatch, tmp_path, return_tensors):
    monkeypatch.setattr(mmlupro, "generate_cot_prompt", lambda val_df, curr, k: "test prompt")
    monkeypatch.setattr(torch.Tensor, "cuda", _identity_cuda)
    monkeypatch.setattr(mmlupro, "max_model_length", 100)
    monkeypatch.setattr(mmlupro, "max_new_tokens", 10)
    monkeypatch.setattr(mmlupro, "batch_inference", lambda *args, **kwargs: (["A"], ["Answer: A"]))
    monkeypatch.setattr(mmlupro, "save_res", lambda res, path: (1.0, 1.0, 0.0))

    accuracy, correct, wrong = mmlupro.eval_cot(
        "math",
        model=None,
        tokenizer=_FixedTokenizer(length=20, return_tensors=return_tensors),
        val_df=[],
        test_df=[{}],
        output_path=str(tmp_path / "result.json"),
        ntrain=1,
        batch_size=1,
    )
    assert (accuracy, correct, wrong) == (1.0, 1.0, 0.0)
