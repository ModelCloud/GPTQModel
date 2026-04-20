# Adapted from HF kernel https://huggingface.co/kernels-community/mamba-ssm/tree/main
# Copyright (c) 2024, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    x = x.float()
    z_work = z.float() if z is not None else None

    if z_work is not None and not norm_before_gate:
        x = x * F.silu(z_work)

    if group_size is None:
        rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * weight
        if bias is not None:
            out = out + bias
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = torch.rsqrt(x_group.square().mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias

    if z_work is not None and norm_before_gate:
        out = out * F.silu(z_work)
    return out.to(dtype)


def _apply_dt_limit(dt, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    dt = dt.float()
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt = dt + dt_bias.view(1, 1, -1)
        else:
            dt = dt + dt_bias.view(1, 1, *dt_bias.shape)
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    return dt


def _repeat_groups(x, target_heads):
    repeats = target_heads // x.shape[1]
    return x.repeat_interleave(repeats, dim=1, output_size=target_heads)


def _expand_A(A, nheads, headdim, dstate):
    if A.dim() == 1:
        return A.view(1, nheads, 1, 1).expand(1, nheads, headdim, dstate)
    if A.dim() == 2:
        return A.view(1, nheads, headdim, dstate)
    if A.dim() == 3:
        return A.view(1, nheads, headdim, dstate)
    raise AssertionError("Unsupported A shape for mamba scan")


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False,
):
    del chunk_size, cu_seqlens, return_varlen_states

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    if nheads % ngroups != 0:
        raise AssertionError("nheads must be divisible by ngroups")

    state_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    state = (
        initial_states.to(dtype=state_dtype).clone()
        if initial_states is not None
        else torch.zeros(batch, nheads, headdim, dstate, device=x.device, dtype=state_dtype)
    )
    A_expanded = _expand_A(A.to(dtype=state_dtype), nheads, headdim, dstate)
    dt = _apply_dt_limit(dt, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    D_work = None if D is None else (D.view(nheads, 1) if D.dim() == 1 else D).to(dtype=state_dtype, device=x.device)
    outputs = []

    for idx in range(seqlen):
        if seq_idx is not None and idx > 0:
            reset_mask = seq_idx[:, idx] != seq_idx[:, idx - 1]
            if reset_mask.any():
                state[reset_mask] = 0

        B_t = _repeat_groups(B[:, idx].to(dtype=state_dtype), nheads).unsqueeze(2)
        C_t = _repeat_groups(C[:, idx].to(dtype=state_dtype), nheads)
        x_t = x[:, idx].to(dtype=state_dtype)
        dt_t = dt[:, idx]
        if dt_t.dim() == 2:
            dt_factor = dt_t.unsqueeze(-1).unsqueeze(-1)
        else:
            dt_factor = dt_t.unsqueeze(-1)

        dA = torch.exp(dt_factor * A_expanded)
        state = state * dA + (dt_factor * B_t * x_t.unsqueeze(-1))
        out_t = (state.to(C_t.dtype) * C_t.unsqueeze(2)).sum(dim=-1)
        if D_work is not None:
            out_t = out_t + x_t * D_work.unsqueeze(0)
        if z is not None:
            out_t = out_t * F.silu(z[:, idx].to(dtype=out_t.dtype))
        outputs.append(out_t.to(dtype=x.dtype))

    out = torch.stack(outputs, dim=1)
    return (out, state) if return_final_states else out


def mamba_split_conv1d_scan_combined(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
):
    if D.dim() == 1:
        if headdim is None:
            raise AssertionError("headdim must be provided when D is 1D")
        nheads = D.shape[0]
    else:
        nheads, headdim = D.shape

    if nheads % ngroups != 0:
        raise AssertionError("nheads must be divisible by ngroups")

    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    if dstate < 0:
        raise AssertionError("Invalid zxbcdt shape for mamba_split_conv1d_scan_combined")

    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(
        causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation, seq_idx=seq_idx),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z_heads = rearrange(z, "b l (h p) -> b l h p", h=nheads)

    scan_result = mamba_chunk_scan_combined(
        x,
        dt.to(dtype=x.dtype),
        A,
        B,
        C,
        chunk_size=chunk_size,
        D=D.float(),
        z=z_heads if rmsnorm_weight is None else None,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        dt_softplus=True,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
    )

    if return_final_states:
        out, final_states = scan_result
    else:
        out = scan_result
        final_states = None

    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(
            out,
            rmsnorm_weight,
            None,
            z=rearrange(z_heads, "b l h p -> b l (h p)"),
            eps=rmsnorm_eps,
            norm_before_gate=norm_before_gate,
        )
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return (out, final_states) if return_final_states else out
