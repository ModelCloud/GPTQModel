# Adapted from HF kernel https://huggingface.co/kernels-community/causal-conv1d/tree/main
# Copyright (c) 2024, Tri Dao.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def _validate_activation(activation):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")


def _apply_activation(x, activation):
    return x if activation is None else F.silu(x)


def _seq_idx_pad_value(seq_idx: torch.Tensor):
    if seq_idx.dtype.is_floating_point:
        return float("nan")
    return torch.iinfo(seq_idx.dtype).min


def _causal_conv1d_seq_idx_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    x_windows = F.pad(x, (width - 1, 0)).unfold(dimension=-1, size=width, step=1)
    seq_windows = F.pad(seq_idx, (width - 1, 0), value=_seq_idx_pad_value(seq_idx)).unfold(
        dimension=-1, size=width, step=1
    )
    mask = seq_windows.eq(seq_idx.unsqueeze(-1)).unsqueeze(1).to(dtype=x_windows.dtype)
    out = (x_windows * mask * weight.view(1, dim, 1, width)).sum(dim=-1)
    if bias is not None:
        out = out + bias.view(1, dim, 1)
    return out[:, :, :seqlen]


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    _validate_activation(activation)

    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    _, width = weight.shape

    if seq_idx is not None:
        if initial_states is not None:
            raise AssertionError("initial_states must be None if seq_idx is not None")
        if return_final_states:
            raise AssertionError("If seq_idx is not None, we don't return final_states_out")
        out = _causal_conv1d_seq_idx_ref(x, weight, bias, seq_idx.to(device=x.device))
        return _apply_activation(out, activation).to(dtype=dtype_in)

    if initial_states is None:
        conv_input = x
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=weight.shape[0])
    else:
        conv_input = torch.cat([initial_states.to(dtype=weight.dtype, device=x.device), x], dim=-1)
        out = F.conv1d(conv_input, weight.unsqueeze(1), bias, padding=0, groups=weight.shape[0])

    out = out[..., :seqlen]
    out = _apply_activation(out, activation).to(dtype=dtype_in)

    if not return_final_states:
        return out

    final_states = F.pad(conv_input, (width - 1 - conv_input.shape[-1], 0)).to(dtype=dtype_in)
    if final_states_out is not None:
        final_states_out.copy_(final_states)
    else:
        final_states_out = final_states
    return out, final_states_out


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    conv_state_indices: (batch,), dtype int32
        If not None, select which rows in conv_state are updated.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    _validate_activation(activation)

    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    _, width = weight.shape

    if conv_state_indices is None:
        selected_state = conv_state
        state_index = None
    else:
        state_index = conv_state_indices.to(device=conv_state.device, dtype=torch.long)
        selected_state = conv_state.index_select(0, state_index)

    if selected_state.shape[:2] != (batch, dim):
        raise AssertionError("conv_state batch or dim shape mismatch")
    if width > state_len + 1:
        raise AssertionError("conv_state state_len must be >= width - 1")

    x_weight_dtype = x.to(weight.dtype)

    if cache_seqlens is None:
        x_new = torch.cat([selected_state.to(weight.dtype), x_weight_dtype], dim=-1)
        updated_state = x_new[:, :, -state_len:]
    else:
        cache_seqlens = cache_seqlens.to(device=x.device, dtype=torch.long)
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0)
        width_idx = torch.remainder(width_idx + cache_seqlens.unsqueeze(1), state_len)
        width_idx = width_idx.unsqueeze(1).expand(-1, dim, -1)
        prefix = selected_state.gather(2, width_idx).to(weight.dtype)
        x_new = torch.cat([prefix, x_weight_dtype], dim=-1)

        updated_state = selected_state.clone()
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
        copy_idx = torch.remainder(copy_idx + cache_seqlens.unsqueeze(1), state_len)
        copy_idx = copy_idx.unsqueeze(1).expand(-1, dim, -1)
        updated_state.scatter_(2, copy_idx, x.to(dtype=updated_state.dtype))

    if state_index is None:
        conv_state.copy_(updated_state)
    else:
        conv_state.index_copy_(0, state_index, updated_state)

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    out = _apply_activation(out, activation).to(dtype=dtype_in)
    if unsqueeze:
        out = out.squeeze(-1)
    return out
