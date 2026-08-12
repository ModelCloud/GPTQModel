# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for calibration attention-mask normalization."""

import pytest
import torch

from gptqmodel.utils.attn_mask import (
    apply_keep_mask_bt,
    attention_mask_sequence_lengths,
    input_id_sequence_lengths,
    normalize_seq_mask,
)


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("additive", [False, True])
def test_normalize_causal_mask_retains_keys_seen_by_any_query(rank, additive):
    """Causal query axes must collapse by union without resurrecting padded keys."""

    keep = torch.tensor([[True, True, True, False], [True, True, False, False]])
    causal = keep[:, None, :] & torch.tril(torch.ones((4, 4), dtype=torch.bool))[None, :, :]
    if rank == 4:
        causal = causal.unsqueeze(1)
    mask = torch.where(causal, 0.0, float("-inf")) if additive else causal

    actual = normalize_seq_mask(mask, seq_len=4)

    assert torch.equal(actual, keep)


def test_normalize_all_zero_additive_mask_keeps_every_token():
    """A floating extended mask with no negative bias represents an unmasked sequence."""

    mask = torch.zeros((2, 1, 1, 5), dtype=torch.float32)

    assert torch.equal(normalize_seq_mask(mask, seq_len=5), torch.ones((2, 5), dtype=torch.bool))


def test_normalize_binary_zero_mask_drops_every_token():
    """A two-dimensional binary mask keeps the standard zero-means-padding contract."""

    mask = torch.zeros((2, 5), dtype=torch.float32)

    assert torch.equal(normalize_seq_mask(mask, seq_len=5), torch.zeros((2, 5), dtype=torch.bool))


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[1, 0, 1], [0, 1, 1]]),
        torch.tensor([[[1, 0, 1]], [[0, 1, 1]]]),
    ],
)
def test_normalize_binary_masks_preserves_each_batch_row(mask):
    expected = torch.tensor([[True, False, True], [False, True, True]])
    assert torch.equal(normalize_seq_mask(mask, seq_len=3), expected)


def test_normalize_rejects_wrong_sequence_width_and_nan():
    assert normalize_seq_mask(None) is None
    with pytest.raises(ValueError, match="scalar"):
        normalize_seq_mask(torch.tensor(1))
    with pytest.raises(ValueError, match="sequence width"):
        normalize_seq_mask(torch.ones((2, 4), dtype=torch.bool), seq_len=5)
    with pytest.raises(ValueError, match="NaN"):
        normalize_seq_mask(torch.tensor([[1.0, float("nan")]]), seq_len=2)


def test_apply_keep_mask_preserves_order_for_holes_and_empty_rows():
    values = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    keep = torch.tensor([[True, False, True, False], [False, False, False, False]])

    actual = apply_keep_mask_bt(values, keep)

    assert torch.equal(actual, values[0, [0, 2]])


def test_apply_keep_mask_concatenates_selected_rows_across_batches():
    values = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
    keep = torch.tensor([[True, False, True], [False, True, True]])
    expected = torch.stack([values[0, 0], values[0, 2], values[1, 1], values[1, 2]])

    assert torch.equal(apply_keep_mask_bt(values, keep), expected)


def test_apply_keep_mask_rejects_shape_mismatch():
    with pytest.raises(AssertionError, match="does not match"):
        apply_keep_mask_bt(torch.zeros((2, 4, 3)), torch.ones((1, 4), dtype=torch.bool))


def test_sequence_length_helpers_share_normalized_mask_semantics():
    causal_with_padding = torch.tensor(
        [
            [
                [
                    [0.0, -float("inf"), -float("inf")],
                    [0.0, 0.0, -float("inf")],
                    [0.0, 0.0, -float("inf")],
                ]
            ]
        ]
    )
    assert attention_mask_sequence_lengths(causal_with_padding.tolist(), seq_len=3) == [2]
    assert attention_mask_sequence_lengths([1, 0, 1], seq_len=3) == [2]
    assert input_id_sequence_lengths([1, 2, 3]) == [3]
    assert input_id_sequence_lengths([[1, 2, 3], [4, 5, 6]]) == [3, 3]


def test_sequence_length_helpers_support_ragged_python_batches():
    assert attention_mask_sequence_lengths(
        [[1, 1], [1, 0, 1]],
        sequence_lengths=[2, 3],
    ) == [2, 2]
    assert input_id_sequence_lengths([[1, 2], [3, 4, 5]]) == [2, 3]
    assert input_id_sequence_lengths([torch.tensor([1]), torch.tensor([2, 3])]) == [1, 2]


def test_sequence_length_helpers_reject_malformed_inputs():
    assert attention_mask_sequence_lengths(None) == []
    assert input_id_sequence_lengths(None) == []
    with pytest.raises(ValueError, match="rectangular"):
        attention_mask_sequence_lengths([[1, 1], 1])
    with pytest.raises(ValueError, match="rectangular"):
        input_id_sequence_lengths([[1, 2], 3])
    with pytest.raises(ValueError, match="at least one dimension"):
        input_id_sequence_lengths(torch.tensor(1))
    with pytest.raises(ValueError, match=r"\[S\] or \[B, S\]"):
        input_id_sequence_lengths(torch.ones((1, 2, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="batch size"):
        attention_mask_sequence_lengths([[1, 1]], seq_len=2, batch_size=2)
    with pytest.raises(ValueError, match="sequence width"):
        attention_mask_sequence_lengths([[1, 1], [1, 0]], sequence_lengths=[2, 3])
    with pytest.raises(ValueError, match="batch size"):
        attention_mask_sequence_lengths([[1], [1, 0]], sequence_lengths=[1])
    with pytest.raises(ValueError, match="either seq_len or sequence_lengths"):
        attention_mask_sequence_lengths([1], seq_len=1, sequence_lengths=[1])
