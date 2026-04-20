# Adapted from HF kernel https://huggingface.co/kernels-community/mamba-ssm/tree/main
# Copyright (c) 2024, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (l two) -> ... l two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (l two) -> ... l two", two=2))
    else:
        B = B.float()
        C = C.float()

    state = A.new_zeros((batch, dim, dstate))
    outputs = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        B = repeat(B, "b g n l -> b (g h) n l", h=dim // B.shape[1])
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "b g n l -> b (g h) n l", h=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        state = deltaA[:, :, i] * state + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", state, C)
        elif C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", state, C[:, :, i])
        else:
            y = torch.einsum("bdn,bdn->bd", state, C[:, :, :, i])

        if i == u.shape[2] - 1:
            last_state = state
        if y.is_complex():
            y = y.real * 2
        outputs.append(y)

    y = torch.stack(outputs, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    return selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


def mamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    **_kwargs,
):
    l = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=l)

    if B is None:
        B = x_dbl[:, delta_rank : delta_rank + d_state]
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=l).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=l, two=2).contiguous()

    if C is None:
        C = x_dbl[:, -d_state:]
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=l).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=l, two=2).contiguous()

    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def mamba_inner_fn(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    **kwargs,
):
    return mamba_inner_ref(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B=B,
        C=C,
        D=D,
        delta_bias=delta_bias,
        B_proj_bias=B_proj_bias,
        C_proj_bias=C_proj_bias,
        delta_softplus=delta_softplus,
        **kwargs,
    )
