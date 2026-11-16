# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro scores each prediction set once (issue #3877)."""

from __future__ import annotations

import torch
from transformers import BatchEncoding

from gptqmodel.utils import mmlupro


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors=None):
        return BatchEncoding(
            {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
        )


def _identity_cuda(self, *args, **kwargs):
    return self


def test_eval_cot_scores_the_predictions_once(monkeypatch, tmp_path):
    calls = []

    def fake_save_res(res, path):
        calls.append(len(res))
        # A second pass would report a different accuracy; the fix must return
        # the first (and only) result.
        return (1.0, 1.0, 0.0) if len(calls) == 1 else (0.0, 0.0, 1.0)

    monkeypatch.setattr(mmlupro, "save_res", fake_save_res)
    monkeypatch.setattr(mmlupro, "generate_cot_prompt", lambda *args: "short prompt")
    monkeypatch.setattr(torch.Tensor, "cuda", _identity_cuda)
    monkeypatch.setattr(
        mmlupro,
        "batch_inference",
        lambda *args, **kwargs: (["A"], ["Answer: A"]),
    )

    accuracy, correct, wrong = mmlupro.eval_cot(
        "math",
        model=None,
        tokenizer=_Tokenizer(),
        val_df=[],
        test_df=[{"answer": "A", "answer_index": 0}],
        output_path=str(tmp_path / "result.json"),
        ntrain=1,
        batch_size=1,
    )

    assert calls == [1]
    assert (accuracy, correct, wrong) == (1.0, 1.0, 0.0)
