# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask


def test_normalize_seq_mask_binary_mask():
    mask = torch.tensor([[1, 0, 1]])

    keep = normalize_seq_mask(mask)

    assert keep.dtype is torch.bool
    assert keep.tolist() == [[True, False, True]]


def test_normalize_seq_mask_additive_zero_keep():
    mask = torch.tensor([[[[0.0, -10000.0, 0.0]]]])

    keep = normalize_seq_mask(mask)

    assert keep.dtype is torch.bool
    assert keep.tolist() == [[True, False, True]]

    values = torch.arange(6, dtype=torch.float32).view(1, 3, 2)
    filtered = apply_keep_mask_bt(values, keep)

    assert torch.equal(filtered, torch.tensor([[0.0, 1.0], [4.0, 5.0]]))


def test_normalize_seq_mask_accepts_multirow_2d():
    mask = torch.tensor([[1, 0, 1], [0, 0, 1]])

    keep = normalize_seq_mask(mask)

    torch.testing.assert_close(
        keep,
        torch.tensor([[True, False, True], [False, False, True]]),
    )


def test_normalize_seq_mask_squeezes_singleton_dims():
    mask = torch.tensor([[[1, 0, 1]], [[0, 1, 1]]], dtype=torch.int)

    keep = normalize_seq_mask(mask)

    torch.testing.assert_close(
        keep,
        torch.tensor([[True, False, True], [False, True, True]]),
    )


def test_normalize_seq_mask_handles_causal_square():
    batch = 2
    seq_len = 5
    causal = torch.tril(torch.ones((batch, 1, seq_len, seq_len), dtype=torch.bool))

    keep = normalize_seq_mask(causal)

    assert keep.shape == (batch, seq_len)
    torch.testing.assert_close(keep, torch.ones((batch, seq_len), dtype=torch.bool))


def test_apply_keep_mask_bt_retains_order():
    x = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    keep = torch.tensor([[True, False, True], [False, True, True]])

    compact = apply_keep_mask_bt(x, keep)

    expected = torch.stack([x[0, 0], x[0, 2], x[1, 1], x[1, 2]])
    torch.testing.assert_close(compact, expected)
