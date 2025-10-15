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
