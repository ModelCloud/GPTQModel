# Copyright 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
# Adapted from the pinned FLA Torch references listed in NOTICE.md; see LICENSE.fla.
"""Portable reference KDA with checkpoint-exact split convolution owners."""

import torch
import torch.nn.functional as F
from torch import nn


def _log1p_exp_neg(x):
    """`log1p(exp(-x))` evaluated only on (0, 1] so every element is geometry-invariant.

    Torch's vectorized CPU `softplus`/`sigmoid` kernels switch vectorized libm
    paths with the tensor length (on AVX512 the same value at length 16 and
    length 64 differs by 1 ulp), so a full call and a chunked call over the
    same values produced different bits. Building the stable identity from
    elementwise ops whose inputs stay in (0, 1] keeps a token's gate bits
    independent of how many tokens share the call, which is what makes a
    cached split run reproduce the single full-sequence run exactly.
    """
    return torch.log1p(torch.exp(-x))


def _softplus(x):
    # Stable `softplus(x) == max(x, 0) + log1p(exp(-|x|))`. The `0.5 * (x + |x|)`
    # form of `max(x, 0)` keeps the derivative at x == 0 equal to 0.5, matching
    # `F.softplus`, and never evaluates `exp` at a positive argument.
    return 0.5 * (x + x.abs()) + _log1p_exp_neg(x.abs())


def _sigmoid(x):
    # `sigmoid(x) == (1 + tanh(x / 2)) / 2`, which avoids the length-dependent
    # fused `sigmoid`/`softplus` kernels and stays finite for any finite input.
    return 0.5 * (1.0 + torch.tanh(0.5 * x))


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, activation="silu"):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        if cu_seqlens is None:
            sequences = list(x.unbind(0))
        else:
            boundaries = cu_seqlens.tolist()
            sequences = [x[0, a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
        outputs, states = [], []
        for index, sequence in enumerate(sequences):
            sequence = sequence.transpose(0, 1).unsqueeze(0)
            previous = (
                sequence.new_zeros(1, self.in_channels, self.kernel_size[0])
                if cache is None
                else cache[index : index + 1]
            )
            extended = torch.cat((previous, sequence), dim=-1)
            states.append(extended[:, :, -self.kernel_size[0] :])
            if sequence.shape[-1]:
                convolved = F.conv1d(extended, self.weight, groups=self.groups)[:, :, -sequence.shape[-1] :]
                outputs.append(F.silu(convolved).transpose(1, 2))
            else:
                outputs.append(sequence.transpose(1, 2))
        output = torch.cat(outputs, dim=1 if cu_seqlens is not None else 0)
        return output, torch.cat(states) if output_final_state else None


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


def kda_forward(q, k, v, g, beta, A_log, dt_bias, initial_state=None, lower_bound=None, cu_seqlens=None):
    dtype = v.dtype
    q, k, v = q.float(), k.float(), v.float()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g = g.float() + dt_bias.float().view(g.shape[-2:])
    decay = A_log.float().exp().view(-1, 1)
    g = -decay * _softplus(g) if lower_bound is None else lower_bound * _sigmoid(decay * g)
    beta = _sigmoid(beta.float())
    if cu_seqlens is None:
        spans = [(i, 0, q.shape[1]) for i in range(q.shape[0])]
    else:
        boundaries = cu_seqlens.tolist()
        spans = [(0, a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    output = torch.zeros_like(v)
    states = []
    for index, (batch, start, end) in enumerate(spans):
        state = (
            k.new_zeros(k.shape[2], k.shape[-1], v.shape[-1])
            if initial_state is None
            else initial_state[index].float()
        )
        for token in range(start, end):
            key, value = k[batch, token], v[batch, token]
            state = state * g[batch, token].exp().unsqueeze(-1)
            delta = (value - (key.unsqueeze(-1) * state).sum(-2)) * beta[batch, token].unsqueeze(-1)
            state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
            output[batch, token] = (q[batch, token].unsqueeze(-1) * state).sum(-2)
        states.append(state)
    return output.to(dtype), torch.stack(states)
