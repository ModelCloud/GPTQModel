# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Q4NX codebook and checkpoint format by AlexBrandes (Alex Brandes).
# Reference: https://github.com/AlexBrandes/exllamav3/blob/41ef7819945fe421626da3c40c678053d240d051/doc/q4nx.md

import math

import torch


Q4NX_MULTIPLIER = 0x83DCD12D
_Q4NX_ALPHA = 1 / math.sqrt(42.5)


def unpack_q4nx_states(packed: torch.Tensor) -> torch.Tensor:
    """Read 64 zero-prefixed trellis states from each packed tile."""
    kv = packed.shape[-1] // 4
    halves = packed.to(torch.int64) & 0xFFFF
    words = halves[..., 0::2] | (halves[..., 1::2] << 16)
    end = kv * torch.arange(1, 65, device=packed.device)
    index = (end - 1) // 32
    shift = 32 * (index + 1) - end
    previous = torch.where(index > 0, words[..., (index - 1).clamp_min(0)], 0)
    window = (previous << 32) | words[..., index]
    return (window >> shift) & ((1 << (32 - kv)) - 1)


def decode_q4nx_states(states: torch.Tensor) -> torch.Tensor:
    """Decode each state to four normalized weights in EXL3 fragment order."""
    product = states * Q4NX_MULTIPLIER
    shifts = torch.arange(0, 32, 8, device=states.device)
    octets = product.unsqueeze(-1) >> shifts
    values = ((octets >> 4) & 15) + (octets & 15) - 15
    return values.flatten(-2).float() * _Q4NX_ALPHA
