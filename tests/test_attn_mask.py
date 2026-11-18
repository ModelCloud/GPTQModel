# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
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


def test_normalize_seq_mask_all_zero_extended_mask_keeps_every_token():
    """An all-zero extended/additive mask is unpadded, not all-padding (#5134).

    PanGu-alpha passes ``(1.0 - mask[:, None, None, :]) * -10000.0`` to every
    decoder block. For a row without padding that mask is all zeros, which the
    old "any negative means additive" inference read as a 0/1 mask and dropped
    every token from the layer Hessian.
    """

    mask = torch.zeros(1, 1, 1, 5)

    keep = normalize_seq_mask(mask)

    assert keep.dtype is torch.bool
    assert keep.tolist() == [[True] * 5]

    # The exact PanGu-alpha form, including a padded row as a control.
    pangu = (1.0 - torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])[:, None, None, :]) * -10000.0
    assert normalize_seq_mask(pangu).tolist() == [[True, True, True, False, False]]


def test_normalize_seq_mask_partially_padded_additive_mask_keeps_real_tokens():
    """A padded additive mask still drops only the negative (masked) keys."""

    mask = torch.tensor([[[[0.0, 0.0, -10000.0, -10000.0]]]])

    keep = normalize_seq_mask(mask)

    assert keep.tolist() == [[True, True, False, False]]


def test_normalize_seq_mask_all_negative_mask_drops_every_token():
    """A fully masked additive mask stays all-drop; zero is not "masked"."""

    mask = torch.full((1, 1, 1, 5), -10000.0)

    keep = normalize_seq_mask(mask)

    assert keep.tolist() == [[False] * 5]


def test_normalize_seq_mask_2d_integer_all_zero_row_is_all_padding():
    """The unpadded rule must not change a 2-D integer 0/1 mask's meaning."""

    mask = torch.zeros(1, 5, dtype=torch.long)

    keep = normalize_seq_mask(mask)

    assert keep.tolist() == [[False] * 5]


def test_normalize_seq_mask_all_zero_float_2d_is_fully_attended():
    """A floating 2-D mask is additive by convention, so all-zero means keep.

    This matches the calibration path's pre-existing rule; only integer 2-D
    masks carry the 0/1 all-padding meaning.
    """

    mask = torch.zeros(1, 5)

    keep = normalize_seq_mask(mask)

    assert keep.tolist() == [[True] * 5]


def test_normalize_seq_mask_all_zero_rank3_integer_is_fully_attended():
    """Rank >= 3 is the extended/causal form: all-zero is unpadded, not masked.

    An all-zero integer 0/1 causal grid is ambiguous. We follow the additive
    convention used by the extended HF masks, because a genuinely all-masked
    calibration row carries no signal and would only trigger RTN fallback.
    """

    mask = torch.zeros(1, 5, 5, dtype=torch.long)

    keep = normalize_seq_mask(mask)

    assert keep.tolist() == [[True] * 5]


def test_normalize_seq_mask_bool_mask_is_unchanged():
    mask = torch.tensor([[True, False, True, False, True]])

    keep = normalize_seq_mask(mask)

    assert keep.dtype is torch.bool
    assert keep.tolist() == [[True, False, True, False, True]]


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["binary", "additive"])
def test_normalize_seq_mask_numeric_paths_match_reference(device, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if kind == "binary":
        mask = torch.tensor(
            [[1, 0, 1, 1], [0, 0, 1, 0]],
            device=device,
            dtype=torch.int64,
        )
    else:
        mask = torch.tensor(
            [[0.0, -10000.0, 0.0, -10000.0]],
            device=device,
            dtype=torch.float32,
        )

    expected = mask >= 0 if torch.any(mask < 0) else mask > 0
    actual = normalize_seq_mask(mask)

    assert actual.dtype is torch.bool
    assert torch.equal(actual, expected)


def test_apply_keep_mask_bt_retains_order():
    x = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    keep = torch.tensor([[True, False, True], [False, True, True]])

    compact = apply_keep_mask_bt(x, keep)

    expected = torch.stack([x[0, 0], x[0, 2], x[1, 1], x[1, 2]])
    torch.testing.assert_close(compact, expected)
