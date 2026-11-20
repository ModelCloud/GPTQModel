# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# R4NX codebook and checkpoint format by ModelCloud.ai.
#
# R4NX is the third EXL3 codebook family. It reuses the Q4NX trellis packing
# byte for byte (same 256-weight tile, same kv-bit states, same 64
# zero-prefixed states) but changes the state -> weight map to a single 32-bit
# multiply followed by a signed-byte extraction:
#
#     product = state * R4NX_MULTIPLIER
#     w[c]    = int8((product >> (8 * c)) & 0xFF) * R4NX_ALPHA
#
# so a decode step is one `v_mul_lo_u32` plus a byte permute on RDNA4, and the
# same table is a plain multiply on CUDA. Q4NX needs a multiply *and* a
# per-nibble `((x >> 4) & 15) + (x & 15)` add, which is what R4NX removes.
#
# `R4NX_MULTIPLIER` is odd, so `state -> product` is a bijection on the state
# space and the four-byte extraction is a bijection onto the codebook the
# quantizer searches. The constant was selected by sweeping odd 32-bit
# multipliers and keeping the one with the lowest reconstruction error on a
# fixed Gaussian weight set at 4.0 bpw; see `tests/kernels/test_exl3_r4nx_codebook.py`
# and the R4NX PR for the measured numbers.

import math

import torch

from .q4nx import unpack_q4nx_states


# Chosen by the multiplier sweep described above.
R4NX_MULTIPLIER = 0x8095576D

# The four extracted values are the signed bytes of `product`, i.e. uniform on
# [-128, 127]: E[v^2] = 1398144 / 256 = 5461.5 and Var[v] = 5461.25. Scaling by
# 1 / sqrt(Var) gives the codebook unit variance, exactly like Q4NX's
# 1 / sqrt(42.5). The optimal loading for Gaussian weights is then ~1.6x the
# tile RMS, which the EXL3 global scale search finds.
R4NX_ALPHA = 1 / math.sqrt(5461.25)

# 64 zero-prefixed states per 256-weight tile, as for Q4NX.
R4NX_STEPS = 64
R4NX_TILE_WEIGHTS = 256
R4NX_STEP_BITS = (10, 12, 14, 16)
R4NX_RATES = (2.5, 3.0, 3.5, 4.0)
R4NX_DEFAULT_BEAM = 128


def r4nx_supports_step_bits(kv: int) -> bool:
    return kv in R4NX_STEP_BITS


def unpack_r4nx_states(packed: torch.Tensor) -> torch.Tensor:
    """Read the 64 zero-prefixed trellis states from each packed tile.

    R4NX keeps the Q4NX tile layout, so the windowing is identical.
    """
    return unpack_q4nx_states(packed)


def _r4nx_values(states: torch.Tensor) -> torch.Tensor:
    """Signed bytes of `state * R4NX_MULTIPLIER`, last axis = fragment order."""
    product = states * R4NX_MULTIPLIER
    shifts = torch.arange(0, 32, 8, device=states.device)
    octets = (product.unsqueeze(-1) >> shifts) & 0xFF
    return torch.where(octets >= 128, octets - 256, octets)


def decode_r4nx_states(states: torch.Tensor) -> torch.Tensor:
    """Decode each state to four normalized weights in EXL3 fragment order."""
    return _r4nx_values(states).flatten(-2).float() * R4NX_ALPHA


def r4nx_states_from_suffixes(suffixes: torch.Tensor, kv: int) -> torch.Tensor:
    """Rebuild the 64 per-tile states from the packed kv-bit suffixes.

    Mirrors `unpack_q4nx_states`: state j is the (32 - kv)-bit window that ends
    at bit kv * (j + 1) of the suffix bitstream, zero padded at the front.
    """
    if not r4nx_supports_step_bits(kv):
        raise ValueError(f"R4NX supports 2.5, 3.0, 3.5, or 4.0 bpw, got kv={kv}")
    mask = (1 << (32 - kv)) - 1
    states = []
    state = torch.zeros(suffixes.shape[:-1], dtype=torch.int64, device=suffixes.device)
    for step in range(suffixes.shape[-1]):
        state = ((state << kv) | (suffixes[..., step].to(torch.int64) & ((1 << kv) - 1))) & mask
        states.append(state)
    return torch.stack(states, dim=-1)


def _viterbi_tile(target: torch.Tensor, kv: int, beam: int | None) -> torch.Tensor:
    """Exact or beam-limited Viterbi for one tile; returns the 64 suffixes."""
    steps = R4NX_STEPS
    class_count = 1 << (32 - 2 * kv)
    suffix_count = 1 << kv
    class_mask = class_count - 1
    live = class_count if beam is None else min(beam, class_count)

    suffixes = torch.arange(suffix_count, device=target.device, dtype=torch.int64)
    history_prev = torch.zeros((steps, class_count), dtype=torch.int64, device=target.device)
    history_suffix = torch.zeros((steps, class_count), dtype=torch.int64, device=target.device)

    classes = torch.zeros(1, dtype=torch.int64, device=target.device)
    costs = torch.zeros(1, dtype=target.dtype, device=target.device)
    for step in range(steps):
        states = (classes[:, None] << kv) | suffixes[None, :]
        values = _r4nx_values(states).to(target.dtype)
        delta = values - target[4 * step:4 * step + 4]
        cost = costs[:, None] + (delta * delta).sum(-1)

        new_class = (states & class_mask).reshape(-1)
        flat = cost.reshape(-1)
        # Deterministic argmin: pack the FP32 cost bits above the predecessor
        # slot so ties resolve to the lowest slot, matching the kernel's key.
        order = torch.argsort(flat, stable=True)
        ranked = new_class[order]
        first = torch.ones_like(ranked, dtype=torch.bool)
        first[1:] = ranked[1:] != ranked[:-1]
        keep = order[first]
        kept_class = new_class[keep]
        kept_cost = flat[keep]

        if kept_cost.numel() > live:
            top = torch.argsort(kept_cost, stable=True)[:live]
            keep = keep[top]
            kept_class = kept_class[top]
            kept_cost = kept_cost[top]

        prev = keep // suffix_count
        suffix = keep % suffix_count
        history_prev[step, kept_class] = classes[prev]
        history_suffix[step, kept_class] = suffix
        classes = kept_class
        costs = kept_cost

    state = classes[torch.argmin(costs)]
    out = torch.zeros(steps, dtype=torch.int64, device=target.device)
    for step in range(steps - 1, -1, -1):
        out[step] = history_suffix[step, state]
        state = history_prev[step, state]
    return out


def encode_r4nx_tiles(
    tiles: torch.Tensor,
    kv: int,
    beam: int | None = R4NX_DEFAULT_BEAM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[rows, 256]` FP32 tiles with the R4NX codebook.

    Returns `(reconstructed, suffixes)`: `reconstructed` holds the decoded
    weights and `suffixes` holds the 64 per-step states in its first 64
    columns, zero padded to 256 entries like the Q4NX encoder output.
    """
    if not r4nx_supports_step_bits(kv):
        raise ValueError(f"R4NX supports 2.5, 3.0, 3.5, or 4.0 bpw, got kv={kv}")
    if tiles.dim() != 2 or tiles.shape[1] != R4NX_TILE_WEIGHTS:
        raise ValueError(f"R4NX quantization expects [rows, {R4NX_TILE_WEIGHTS}] tiles, got {tuple(tiles.shape)}")

    rows = tiles.shape[0]
    out_suffixes = torch.zeros((rows, R4NX_TILE_WEIGHTS), dtype=torch.int16, device=tiles.device)
    if rows == 0:
        return torch.empty_like(tiles), out_suffixes

    targets = tiles.detach().to(torch.float32)
    # Divide by a tensor: torch's scalar division lowers to a reciprocal
    # multiply on CUDA, which differs from the kernel's single IEEE division
    # by one ulp for some inputs, and that is enough to flip a tie.
    targets = targets / targets.new_tensor(R4NX_ALPHA)
    for row in range(rows):
        out_suffixes[row, :R4NX_STEPS] = _viterbi_tile(targets[row], kv, beam).to(torch.int16)

    states = r4nx_states_from_suffixes(out_suffixes[:, :R4NX_STEPS], kv)
    return decode_r4nx_states(states).reshape(rows, R4NX_TILE_WEIGHTS), out_suffixes


def pack_r4nx_trellis(suffixes: torch.Tensor, kv: int) -> torch.Tensor:
    """Pack per-tile suffixes into the Q4NX-compatible int16 tile layout."""
    if not r4nx_supports_step_bits(kv):
        raise ValueError(f"R4NX supports 2.5, 3.0, 3.5, or 4.0 bpw, got kv={kv}")
    steps = suffixes.shape[-1]
    codes = suffixes[..., :steps].to(torch.int64) & ((1 << kv) - 1)
    bits = (codes.unsqueeze(-1) >> torch.arange(kv - 1, -1, -1, device=suffixes.device)) & 1
    weights = 1 << torch.arange(15, -1, -1, device=suffixes.device)
    words = (bits.reshape(*bits.shape[:-2], steps * kv).reshape(*bits.shape[:-2], -1, 16) * weights).sum(-1)
    return words.reshape(*words.shape[:-1], -1, 2).flip(-1).flatten(-2).to(torch.int16)


__all__ = [
    "R4NX_ALPHA",
    "R4NX_DEFAULT_BEAM",
    "R4NX_MULTIPLIER",
    "R4NX_RATES",
    "R4NX_STEPS",
    "R4NX_STEP_BITS",
    "R4NX_TILE_WEIGHTS",
    "decode_r4nx_states",
    "encode_r4nx_tiles",
    "pack_r4nx_trellis",
    "r4nx_states_from_suffixes",
    "r4nx_supports_step_bits",
    "unpack_r4nx_states",
]
