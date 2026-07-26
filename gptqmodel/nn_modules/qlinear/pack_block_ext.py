# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Tuple

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


def pack_awq_cpu(intweight: Tensor, zeros: Tensor, bits: int) -> Tuple[Tensor, Tensor]:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("pack_awq_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_awq_cpu(
        intweight,
        zeros,
        int(bits),
    )


def pack_qqq_cpu(int4_matrix: Tensor, bits: int) -> Tensor:
    ext = load_pack_block_extension()
    if ext is None:
        raise RuntimeError("pack_qqq_cpu extension unavailable")
    return torch.ops.gptqmodel.pack_qqq_cpu(
        int4_matrix,
        int(bits),
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
