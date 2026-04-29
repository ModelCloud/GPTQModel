# Adapted from HF kernel https://huggingface.co/kernels-community/mamba-ssm/tree/main
# Copyright (c) 2024, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
):
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    if state_batch_indices is None:
        selected_state = state
        state_index = None
    else:
        state_index = state_batch_indices.to(device=state.device, dtype=torch.long)
        selected_state = state.index_select(0, state_index)

    batch, nheads, dim, dstate = selected_state.shape
    if dt_bias is not None:
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt

    if x.shape != (batch, nheads, dim):
        raise AssertionError("x shape mismatch for selective_state_update")
    if dt.shape != x.shape:
        raise AssertionError("dt shape mismatch for selective_state_update")
    if A.shape != (nheads, dim, dstate):
        raise AssertionError("A shape mismatch for selective_state_update")

    ngroups = B.shape[1]
    if nheads % ngroups != 0:
        raise AssertionError("nheads must be divisible by ngroups")
    if B.shape != (batch, ngroups, dstate) or C.shape != B.shape:
        raise AssertionError("B/C shape mismatch for selective_state_update")
    if D is not None and D.shape != (nheads, dim):
        raise AssertionError("D shape mismatch for selective_state_update")
    if z is not None and z.shape != x.shape:
        raise AssertionError("z shape mismatch for selective_state_update")

    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")
    updated_state = selected_state * dA + dB * rearrange(x, "b h d -> b h d 1")

    if state_index is None:
        state.copy_(updated_state)
    else:
        state.index_copy_(0, state_index, updated_state)

    out = torch.einsum("bhdn,bhn->bhd", updated_state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    return out if has_heads else out.squeeze(1)
