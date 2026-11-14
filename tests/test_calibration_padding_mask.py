# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Padding masks must reach `support_batch_quantize=False` calibration batches.

Issue #4038: `prepare_calibration_dataset` emitted only `input_ids` for those
models, so a padded tail block from `calibration_dataset_concat_size` (and any
user row that already carried padding) entered the Hessian as real tokens.
"""

from types import SimpleNamespace

from gptqmodel.utils.calibration import prepare_calibration_dataset


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"


def _qmodel(*, support_batch_quantize: bool = False):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        support_batch_quantize=support_batch_quantize,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=128),
        ),
    )


def test_support_batch_quantize_false_keeps_user_attention_mask():
    input_ids = [[5, 6, 7, 8, 9]]
    attention_mask = [[1, 1, 0, 0, 1]]

    prepared = prepare_calibration_dataset(
        _qmodel(),
        [{"input_ids": input_ids, "attention_mask": attention_mask}],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == input_ids
    assert prepared[0]["attention_mask"].tolist() == attention_mask


def test_support_batch_quantize_false_keeps_concat_tail_padding():
    prepared = prepare_calibration_dataset(
        _qmodel(),
        [{"input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1, 1]]}],
        calibration_dataset_concat_size=8,
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5, 0, 0, 0]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]


def test_batched_path_attention_mask_is_unchanged():
    prepared = prepare_calibration_dataset(
        _qmodel(support_batch_quantize=True),
        [{"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}],
        batch_size=1,
        calibration_data_min_length=1,
    )

    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]
