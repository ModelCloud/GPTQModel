# Adapted from HF kernel https://huggingface.co/kernels-community/causal-conv1d/tree/main
# Copyright (c) 2024, Tri Dao.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def causal_conv1d_varlen_states(x: Tensor, cu_seqlens: Tensor, state_len: int) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the
            sequence lengths, starting from 0.
        state_len: int. For each sequence, how many elements from x should be copied
            to the state. Tokens from earlier sequences are ignored.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    states = torch.zeros(batch, state_len, dim, dtype=x.dtype, device=x.device).transpose(1, 2)
    cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.long)

    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T

    return states
