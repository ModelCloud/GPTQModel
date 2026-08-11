# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from gptqmodel.utils.cpp import load_pack_block_extension


def pack_block_cpu(
    weight: Tensor,
    scales: Tensor,
    zeros: Tensor,
    g_idx: Tensor,
    bits: int,
    word_bits: int,
    block_in: int,
    threads: int,
) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("pack_block_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_block_cpu(
        weight,
        scales,
        zeros,
        g_idx,
        int(bits),
        int(word_bits),
        int(block_in),
        int(threads),
    )


def pack_awq_cpu(
    intweight: Tensor,
    zeros: Tensor,
    bits: int,
    threads: Optional[int] = -1,
) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("pack_awq_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_awq_cpu(
        intweight,
        zeros,
        int(bits),
        int(threads),
    )


def pack_qqq_cpu(
    int4_matrix: Tensor,
    bits: int,
    threads: Optional[int] = -1,
) -> Tensor:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("pack_qqq_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_qqq_cpu(
        int4_matrix,
        int(bits),
        int(threads),
    )


def hessian_xtx_cpu(
    X: Tensor,
    out: Tensor | None = None,
    beta: float = 0.0,
    alpha: float = 1.0,
) -> Tensor:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("hessian_xtx_cpu extension unavailable")
    return torch.ops.gptqmodel.hessian_xtx_cpu(
        X,
        out,
        float(beta),
        float(alpha),
    )


def hessian_inverse_cholesky_cpu(H: Tensor, diag_delta: Tensor) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("hessian_inverse_cholesky_cpu extension unavailable")
    return torch.ops.gptqmodel.hessian_inverse_cholesky_cpu(
        H,
        diag_delta,
    )


def find_params_batched_cpu(
    x: Tensor,
    xmin: Tensor,
    xmax: Tensor,
    importance: Tensor | None,
    grid: int,
    maxshrink: float,
    maxq: int,
    sym: bool,
    groupwise: bool,
    method: str,
    mse: float,
) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("find_params_batched_cpu extension unavailable")

    topk_idx, topk_scale, topk_zero = torch.ops.gptqmodel.find_params_batched_cpu_topk(
        x,
        xmin,
        xmax,
        importance,
        int(grid),
        float(maxshrink),
        int(maxq),
        bool(sym),
        bool(groupwise),
        float(mse),
    )

    # Recompute the exact loss for the short-listed candidates with the same
    # arithmetic used by the eager fallback / Triton recompute, and return the
    # single best (scale, zero) per (row, group).
    x_f = x.float()
    maxq_f = float(maxq)
    # The native shortlist includes the exact zero-point orientation selected
    # for each shrink candidate.  In particular, asymmetric 2-bit search may
    # return the same shrink index twice with the two adjacent integer zeros.
    scale_k = topk_scale
    zero_k = topk_zero

    k = topk_idx.size(-1)
    x_exp = x_f.unsqueeze(2).expand(-1, -1, k, -1)
    scale_k_exp = scale_k.unsqueeze(-1)
    zero_k_exp = zero_k.unsqueeze(-1)
    q = x_exp / scale_k_exp
    q.round_()
    if groupwise:
        q.clamp_(-maxq_f, maxq_f)
    else:
        q.add_(zero_k_exp).clamp_(0.0, maxq_f).sub_(zero_k_exp)
    dequant = q * scale_k_exp
    error = dequant - x_exp
    if method == "activation" and importance is not None:
        error.pow_(2)
        error.mul_(importance[None, :, None, :])
    else:
        error.abs_().pow_(mse)
    losses = error.sum(dim=-1)
    best = losses.argmin(dim=-1)
    scale = scale_k.gather(2, best.unsqueeze(-1)).squeeze(-1)
    zero = zero_k.gather(2, best.unsqueeze(-1)).squeeze(-1)
    return scale, zero


def gptq_block_cpu(
    W1: Tensor,
    Hinv1: Tensor,
    scale: Tensor,
    zero: Tensor,
    maxq: int,
    group_size: int,
    groupwise: bool = False,
) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("gptq_block_cpu extension unavailable")
    return torch.ops.gptqmodel.gptq_block_cpu(
        W1,
        Hinv1,
        scale,
        zero,
        int(maxq),
        int(group_size),
        bool(groupwise),
    )
