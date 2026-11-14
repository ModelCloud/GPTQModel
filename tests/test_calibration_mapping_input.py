# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer outputs are mappings, not builtin dicts.

Issue #4052: a list of ``tokenizer(text)`` results (``BatchEncoding``, a
``UserDict``) fell through the ``isinstance(example, dict)`` check and was
rejected as an unsupported calibration example.
"""

from types import SimpleNamespace

import torch
from transformers import BatchEncoding

from gptqmodel.utils.calibration import prepare_calibration_dataset


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"


def _qmodel():
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=128),
        ),
    )


def test_batch_encoding_calibration_example_is_accepted():
    encoded = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
    )

    prepared = prepare_calibration_dataset(
        _qmodel(),
        [encoded],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5]]


def test_batch_encoding_without_attention_mask_is_accepted():
    encoded = BatchEncoding({"input_ids": torch.tensor([[6, 7, 8]])})

    prepared = prepare_calibration_dataset(
        _qmodel(),
        [encoded],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert prepared[0]["input_ids"].tolist() == [[6, 7, 8]]
    assert prepared[0]["attention_mask"].tolist() == [[True, True, True]]
