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
