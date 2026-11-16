# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Calibration input normalisation and validation contracts (issue #4194)."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.utils.attn_mask import normalize_seq_mask
from gptqmodel.utils.calibration import prepare_calibration_dataset


MAX_POSITIONS = 128


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"t{int(token)}" for token in ids)


def _qmodel(*, support_batch_quantize: bool = True, max_positions: int = MAX_POSITIONS):
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        support_batch_quantize=support_batch_quantize,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=max_positions)
        ),
    )


def _prepare(
    dataset,
    *,
    support_batch_quantize: bool = True,
    max_positions: int = MAX_POSITIONS,
    **kwargs,
):
    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("calibration_data_min_length", 0)
    return prepare_calibration_dataset(
        _qmodel(
            support_batch_quantize=support_batch_quantize,
            max_positions=max_positions,
        ),
        dataset,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Defect 1: [B, T, T] masks must be treated as per-row causal masks.
# ---------------------------------------------------------------------------


def test_normalize_seq_mask_handles_3d_causal_square():
    causal = torch.tril(torch.ones(2, 5, 5, dtype=torch.bool))

    keep = normalize_seq_mask(causal)

    assert keep.shape == (2, 5)
    assert keep.tolist() == [[True] * 5, [True] * 5]


def test_3d_causal_attention_mask_keeps_every_real_token():
    seq_len = 16
    input_ids = torch.arange(1, seq_len + 1).unsqueeze(0)
    causal = torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.long))

    prepared = _prepare([{"input_ids": input_ids, "attention_mask": causal}])

    # Reference: a full causal lower triangle has a real token at every
    # position, so the collapsed keep-mask is all ones.
    assert prepared[0]["attention_mask"].tolist() == [[1] * seq_len]


def test_3d_causal_attention_mask_preserves_padding_positions():
    seq_len = 5
    keep = torch.tensor([True, True, True, False, False])
    causal = torch.zeros((1, seq_len, seq_len), dtype=torch.bool)
    for query_idx in range(seq_len):
        causal[0, query_idx] = keep & (torch.arange(seq_len) <= query_idx)

    prepared = _prepare(
        [
            {
                "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
                "attention_mask": causal,
            }
        ]
    )

    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]


# ---------------------------------------------------------------------------
# Defect 2: min-length filtering is per row and counts only real tokens.
# ---------------------------------------------------------------------------


def test_min_length_filters_each_row_by_real_token_count():
    long_ids = list(range(1, 21))
    short_ids = [31, 32, 33] + [0] * 17
    dataset = [
        {
            "input_ids": [long_ids, short_ids],
            "attention_mask": [[1] * 20, [1, 1, 1] + [0] * 17],
        }
    ]

    prepared = _prepare(dataset, calibration_data_min_length=10)

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].tolist() == [long_ids]
    assert prepared[0]["attention_mask"].tolist() == [[1] * 20]


def test_min_length_does_not_count_padding_as_real_tokens():
    dataset = [
        {
            "input_ids": [[7, 7, 7] + [0] * 30],
            "attention_mask": [[1, 1, 1] + [0] * 30],
        }
    ]

    assert _prepare(dataset, calibration_data_min_length=10) == []


def test_min_length_drops_an_item_when_no_row_survives():
    dataset = [
        {
            "input_ids": [[1, 2, 3, 0], [4, 5, 6, 7]],
            "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 1]],
        }
    ]

    assert _prepare(dataset, calibration_data_min_length=10) == []


def test_min_length_filter_re_pads_ragged_unbatched_items():
    dataset = [
        {
            "input_ids": [list(range(1, 21)), [31, 32, 33, 34, 35, 36, 37, 38] + [0] * 12],
            "attention_mask": [[1] * 20, [1] * 8 + [0] * 12],
        }
    ]

    prepared = _prepare(
        dataset,
        support_batch_quantize=False,
        calibration_data_min_length=5,
        max_positions=10,
    )

    # Both rows survive the per-row minimum, and stripping the over-long row
    # leaves different widths, so the unbatched path must re-pad the shorter row.
    assert prepared[0]["input_ids"].tolist() == [
        list(range(1, 11)),
        [31, 32, 33, 34, 35, 36, 37, 38, 0, 0],
    ]
    assert prepared[0]["attention_mask"].tolist() == [
        [1] * 10,
        [1] * 8 + [0, 0],
    ]


# ---------------------------------------------------------------------------
# Defect 3: validation gaps.
# ---------------------------------------------------------------------------


def test_unknown_calibration_sort_is_rejected():
    dataset = [{"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}]

    with pytest.raises(ValueError, match="calibration_dataset_sort"):
        _prepare(dataset, calibration_dataset_sort="descending")


def test_numpy_integer_token_list_is_accepted():
    np = pytest.importorskip("numpy")
    sample = list(np.array([1, 2, 3], dtype=np.int64))

    prepared = _prepare([sample])

    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_torch_scalar_token_list_is_accepted():
    sample = [torch.tensor(1), torch.tensor(2), torch.tensor(3)]

    prepared = _prepare([sample])

    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3]]


def test_nested_token_id_list_is_accepted():
    prepared = _prepare([[[1, 2, 3]]])

    assert prepared[0]["input_ids"].tolist() == [[1, 2, 3]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 1]]


@pytest.mark.parametrize(
    "sample",
    [
        torch.empty(0, 8, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    ],
)
def test_empty_token_tensor_raises_a_clear_error(sample):
    with pytest.raises(ValueError, match="empty"):
        _prepare([sample])


def test_float_token_ids_are_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="integers"):
        _prepare([torch.tensor([[1.0, 2.0, 3.0]])])
