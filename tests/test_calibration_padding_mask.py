# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Padding masks must reach `support_batch_quantize=False` calibration batches.

Issue #4038: `prepare_calibration_dataset` emitted only `input_ids` for those
models, so a padded tail block from `calibration_dataset_concat_size` (and any
user row that already carried padding) entered the Hessian as real tokens.

Issue #4861: a left-padded capture batch passed no `position_ids`, so models
with learned absolute position embeddings (GPT-2, GPT-Neo) embedded every real
token at `pad_len + i` and captured hidden states the model never produces at
inference. `run_input_capture` now derives them from the attention mask, which
is what `generate()` does.
"""

from types import SimpleNamespace

import torch

from gptqmodel.models.base import BaseQModel
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


class _RecordingModel:
    """Causal-LM stand-in that records the kwargs its forward receives."""

    def __init__(self, *, accept_position_ids: bool):
        self.calls = []
        if accept_position_ids:

            def forward(input_ids, attention_mask=None, position_ids=None, use_cache=None):
                self.calls.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    }
                )
                return input_ids

        else:

            def forward(input_ids, attention_mask=None, use_cache=None):
                self.calls.append(
                    {"input_ids": input_ids, "attention_mask": attention_mask}
                )
                return input_ids

        self.forward = forward

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _capture_qmodel(model):
    qmodel = BaseQModel.__new__(BaseQModel)
    qmodel.model = model
    qmodel.INPUT_EMBEDDING_EXTRA_ARGS = {}
    qmodel.ATTENTION_MASKS_DTYPE = torch.long
    return qmodel


def _capture(model, input_ids, attention_mask):
    qmodel = _capture_qmodel(model)
    qmodel.run_input_capture(
        {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)},
        use_cache=False,
        data_device=torch.device("cpu"),
    )
    return model.calls[-1]


def test_left_padded_capture_derives_position_ids_from_the_mask():
    """The left-padded row must count positions from its first real token."""

    call = _capture(
        _RecordingModel(accept_position_ids=True),
        input_ids=[[0, 0, 5, 6, 7], [8, 9, 10, 0, 0]],
        attention_mask=[[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]],
    )

    # Real tokens run 0, 1, 2 from the first non-pad slot on both rows; pad
    # slots take 1, which attention masks out. Same as `generate()`.
    assert call["position_ids"].tolist() == [[1, 1, 0, 1, 2], [0, 1, 2, 1, 1]]


def test_unpadded_capture_position_ids_are_arange():
    """An unpadded batch derives the same ids the model would default to."""

    call = _capture(
        _RecordingModel(accept_position_ids=True),
        input_ids=[[5, 6, 7], [8, 9, 10]],
        attention_mask=[[1, 1, 1], [1, 1, 1]],
    )

    assert call["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]


def test_capture_keeps_explicit_position_ids():
    model = _RecordingModel(accept_position_ids=True)
    qmodel = _capture_qmodel(model)
    explicit = torch.tensor([[7, 7, 7]])

    qmodel.run_input_capture(
        {
            "input_ids": torch.tensor([[0, 5, 6]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
            "position_ids": explicit,
        },
        use_cache=False,
        data_device=torch.device("cpu"),
    )

    assert model.calls[-1]["position_ids"] is explicit


def test_capture_skips_position_ids_when_the_model_does_not_accept_them():
    """A forward without `position_ids` must not receive the keyword."""

    call = _capture(
        _RecordingModel(accept_position_ids=False),
        input_ids=[[0, 5, 6]],
        attention_mask=[[0, 1, 1]],
    )

    assert "position_ids" not in call


def test_columnar_float_0_1_mask_matches_the_list_form():
    """A fully padded float row must stay padding in the columnar form (#5368)."""

    payload = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
    }

    columnar = prepare_calibration_dataset(_qmodel(), payload, calibration_data_min_length=0)
    listed = prepare_calibration_dataset(_qmodel(), [payload], calibration_data_min_length=0)

    assert [(row["input_ids"].tolist(), row["attention_mask"].tolist()) for row in columnar] == [
        (row["input_ids"].tolist(), row["attention_mask"].tolist()) for row in listed
    ]
    assert len(columnar) == 1
    assert columnar[0]["input_ids"].tolist() == [[1, 2]]


def test_columnar_all_zero_float_mask_stays_an_additive_attend_all_mask():
    """With no 0/1 sentinel the column is additive, so zero means attend (#5134)."""

    payload = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    }

    prepared = prepare_calibration_dataset(_qmodel(), payload, calibration_data_min_length=0)

    assert len(prepared) == 2
    for row in prepared:
        assert row["attention_mask"].tolist() == [[1, 1]]


def test_columnar_additive_mask_keeps_its_negative_sentinel():
    payload = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[0.0, 0.0], [0.0, -10000.0]]),
    }

    prepared = prepare_calibration_dataset(_qmodel(), payload, calibration_data_min_length=0)

    assert len(prepared) == 2
    assert prepared[0]["attention_mask"].tolist() == [[1, 1]]
    assert prepared[1]["input_ids"].tolist() == [[3]]
    assert prepared[1]["attention_mask"].tolist() == [[1]]
