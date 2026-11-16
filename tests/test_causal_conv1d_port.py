# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import gptqmodel.hf_kernels.mamba_ssm as local_mamba_ssm
from gptqmodel.hf_kernels import causal_conv1d


def test_causal_conv1d_fn_matches_grouped_conv_without_seq_idx():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.5, -0.25], [0.2, -0.1, 0.3]], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32)

    out = causal_conv1d.causal_conv1d_fn(x, weight, bias=bias, activation=None)
    expected = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[:, :, : x.shape[-1]]

    torch.testing.assert_close(out, expected)


def test_causal_conv1d_fn_respects_seq_idx_boundaries():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 10.0]], dtype=torch.float32)
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)

    out = causal_conv1d.causal_conv1d_fn(x, weight, seq_idx=seq_idx, activation=None)
    expected = torch.tensor([[[10.0, 21.0, 30.0, 43.0]]], dtype=torch.float32)

    torch.testing.assert_close(out, expected)


def test_causal_conv1d_update_supports_state_indices_and_cache_seqlens():
    conv_state = torch.tensor(
        [
            [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]],
            [[40.0, 50.0, 60.0], [4.0, 5.0, 6.0]],
            [[70.0, 80.0, 90.0], [7.0, 8.0, 9.0]],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[[100.0], [10.0]], [[200.0], [20.0]]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    cache_seqlens = torch.tensor([1, 2], dtype=torch.int32)
    conv_state_indices = torch.tensor([2, 0], dtype=torch.int32)

    out = causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )

    expected_out = torch.tensor([[[70.0], [10.0]], [[20.0], [20.0]]], dtype=torch.float32)
    expected_state = torch.tensor(
        [
            [[10.0, 20.0, 200.0], [1.0, 2.0, 20.0]],
            [[40.0, 50.0, 60.0], [4.0, 5.0, 6.0]],
            [[70.0, 100.0, 90.0], [7.0, 10.0, 9.0]],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(conv_state, expected_state)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_causal_conv1d_update_chunk_longer_than_state_keeps_newest_tokens(device):
    """Regression for #3868: duplicate circular indices raced on CUDA."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    state = torch.zeros(1, 1, 2, device=device)
    weight = torch.tensor([[1.0, 0.0, 0.0]], device=device)

    first = causal_conv1d.causal_conv1d_update(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device),
        state,
        weight,
        cache_seqlens=torch.tensor([0], device=device),
    )
    assert first.tolist() == [[[0.0, 0.0, 1.0, 2.0]]]
    # Only the final two tokens fit in the cache.
    assert state.tolist() == [[[3.0, 4.0]]]

    next_out = causal_conv1d.causal_conv1d_update(
        torch.tensor([[5.0]], device=device),
        state,
        weight,
        cache_seqlens=torch.tensor([4], device=device),
    )
    assert next_out.tolist() == [[3.0]]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_causal_conv1d_update_long_chunk_wraps_to_unique_indices(device):
    """A chunk longer than the cache must still land on unique slots."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    state = torch.zeros(1, 1, 4, device=device)
    weight = torch.tensor([[1.0, 0.0, 0.0]], device=device)

    causal_conv1d.causal_conv1d_update(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], device=device),
        state,
        weight,
        cache_seqlens=torch.tensor([3], device=device),
    )
    # Tokens 1..6 start at slot 3: 1->3, 2->0, 3->1, 4->2, 5->3, 6->0.
    # The surviving state is tokens 3..6 in circular order.
    assert state.tolist() == [[[6.0, 3.0, 4.0, 5.0]]]

    next_out = causal_conv1d.causal_conv1d_update(
        torch.tensor([[7.0]], device=device),
        state,
        weight,
        cache_seqlens=torch.tensor([9], device=device),
    )
    assert next_out.tolist() == [[5.0]]


def test_local_mamba_chunk_scan_combined_returns_expected_state():
    x = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32)
    dt = torch.ones(1, 2, 1, dtype=torch.float32)
    A = torch.zeros(1, dtype=torch.float32)
    B = torch.ones(1, 2, 1, 1, dtype=torch.float32)
    C = torch.ones(1, 2, 1, 1, dtype=torch.float32)

    out, final_state = local_mamba_ssm.mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=2,
        return_final_states=True,
    )

    expected_out = torch.tensor([[[[1.0]], [[3.0]]]], dtype=torch.float32)
    expected_state = torch.tensor([[[[3.0]]]], dtype=torch.float32)
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(final_state, expected_state)
