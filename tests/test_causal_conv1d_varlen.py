# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Variable-length state extraction with empty sequences (#3871).

`cu_seqlens` is the cumulative sum of sequence lengths, so equal adjacent
offsets describe an empty sequence. The slice `-(end - start):` becomes `-0:`
for such a sequence, which selected the whole state buffer and raised instead
of leaving the pre-zeroed row alone.
"""

import pytest
import torch

from gptqmodel.hf_kernels.causal_conv1d import causal_conv1d_varlen_states


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")),])
def test_empty_first_sequence_keeps_its_state_zeroed(device):
    x = torch.tensor([[1.0], [2.0]], device=device)
    states = causal_conv1d_varlen_states(x, torch.tensor([0, 0, 2]), state_len=2)

    assert states.shape == (2, 1, 2)
    torch.testing.assert_close(states[0], torch.zeros(1, 2, device=device))
    torch.testing.assert_close(states[1], torch.tensor([[1.0, 2.0]], device=device))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")),])
def test_empty_last_sequence_keeps_its_state_zeroed(device):
    x = torch.tensor([[1.0], [2.0]], device=device)
    states = causal_conv1d_varlen_states(x, torch.tensor([0, 2, 2]), state_len=2)

    torch.testing.assert_close(states[0], torch.tensor([[1.0, 2.0]], device=device))
    torch.testing.assert_close(states[1], torch.zeros(1, 2, device=device))


def test_all_sequences_empty_returns_zero_state():
    x = torch.zeros(0, 3)
    states = causal_conv1d_varlen_states(x, torch.tensor([0, 0, 0]), state_len=2)

    assert states.shape == (2, 3, 2)
    torch.testing.assert_close(states, torch.zeros_like(states))


def test_state_is_right_aligned_and_truncated_to_state_len():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    states = causal_conv1d_varlen_states(x, torch.tensor([0, 5]), state_len=2)

    # Only the final `state_len` tokens are copied, right-aligned.
    torch.testing.assert_close(states[0], torch.tensor([[4.0, 5.0]]))

    padded = causal_conv1d_varlen_states(x, torch.tensor([0, 5]), state_len=4)
    torch.testing.assert_close(padded[0], torch.tensor([[2.0, 3.0, 4.0, 5.0]]))


def test_empty_sequence_between_nonempty_sequences():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    states = causal_conv1d_varlen_states(x, torch.tensor([0, 1, 1, 3]), state_len=2)

    torch.testing.assert_close(states[0], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(states[1], torch.zeros(1, 2))
    torch.testing.assert_close(states[2], torch.tensor([[2.0, 3.0]]))
