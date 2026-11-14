# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Input handling for the MMLU-Pro evaluation helpers (issue #3875)."""

from __future__ import annotations

import torch
from transformers import BatchEncoding

from gptqmodel.utils import mmlupro


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def _ids(text: str) -> list[int]:
        # Keep IDs away from the pad ID so the mask is meaningful.
        return [2 + (ord(char) % 50) for char in text]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=None,
        truncation=None,
        max_length=None,
        padding_side=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._ids(text) for text in texts]
        if return_tensors != "pt":
            return {"input_ids": rows}

        width = max(len(row) for row in rows)
        input_ids, attention_mask = [], []
        for row in rows:
            pad = width - len(row)
            if padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + row)
                attention_mask.append([0] * pad + [1] * len(row))
            else:
                input_ids.append(row + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(row) + [0] * pad)
        return BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )

    def batch_decode(self, sequences, skip_special_tokens=True):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return ["Answer: A" for _ in sequences]


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": None if attention_mask is None else attention_mask.clone(),
                **kwargs,
            }
        )
        extra = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=1)


def _example(**overrides):
    example = {
        "category": "math",
        "question": "What is 1 + 1?",
        "options": ["1", "2", "3", "4"],
        "cot_content": "A: Let's think step by step. One plus one is two.",
        "answer": "A",
        "answer_index": 0,
    }
    example.update(overrides)
    return example


def test_batch_inference_passes_the_padding_mask_to_generate():
    model = _Model()
    tokenizer = _Tokenizer()

    preds, responses = mmlupro.batch_inference(
        model, tokenizer, ["a", "abc"], batch_size=2
    )

    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["attention_mask"] is not None
    # "a" is left-padded against the longer "abc" prompt.
    assert call["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]
    assert call["input_ids"].tolist()[0][:2] == [tokenizer.pad_token_id] * 2
    assert preds == ["A", "A"]
    assert len(responses) == 2


def test_eval_cot_counts_prompt_tokens_without_cuda(monkeypatch, tmp_path):
    def fail_cuda(self, *args, **kwargs):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(torch.Tensor, "cuda", fail_cuda)

    model = _Model()
    tokenizer = _Tokenizer()
    val_df = [_example()]
    test_df = [_example(question="What is 2 + 2?")]

    accuracy, correct, wrong = mmlupro.eval_cot(
        "math",
        model,
        tokenizer,
        val_df,
        test_df,
        str(tmp_path / "result.json"),
        ntrain=1,
        batch_size=1,
    )

    assert (accuracy, correct, wrong) == (1.0, 1.0, 0.0)
