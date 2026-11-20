# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""R4NX: the RDNA4-native NX codebook for EXL3.

R4NX is a third EXL3 codebook family next to ``mcg``/``mul1`` (the 3-inst
codebooks) and ``q4nx``.  It reuses Q4NX's packing exactly -- the same 16x16
tile layout, the same 64 zero-prefixed ``kv``-bit states per 256-weight tile
(``kv = 4 * rate``, so 2.5/3.0/3.5/4.0 bpw) and therefore the same bitstream,
which is why :func:`unpack_r4nx_states` is byte-for-byte the Q4NX unpacking --
and changes only the state-to-weight map.

Decode
------

For a state ``s`` (the ``32 - kv`` bit zero-prefixed sliding window ending at
the state's last packed bit)::

    product  = (s * R4NX_MULTIPLIER) mod 2**32
    level[j] = int8((product >> (8 * j)) & 0xFF) * R4NX_ALPHA     j = 0..3

The four levels are the four bytes of the product read as signed 8-bit values,
in EXL3 fragment order.  On gfx12xx that is one ``v_mul_lo_u32`` plus one
``v_perm_b32`` (or three ``v_lshrrev_b32``) per four weights: no add, no nibble
split and no lookup table.  Q4NX needs the same multiply *and* a per-nibble
``((x >> 4) & 15) + (x & 15)`` add before its ``-15`` bias.

Because the multiplier is odd, multiplication is a bijection on ``Z/2**32``:
the four bytes carry the whole product, so state -> four bytes is injective on
every state the packer can produce and the codebook has no aliased states.

Choosing ``R4NX_MULTIPLIER``
----------------------------

The multiplier only decides *which* 4-D codebook the states sample, and it was
chosen by measurement.  Fixed target set: ``torch.randn`` 4-tuples ``N(0, 1)``
with seeds 3000+``kv`` (1024 tuples, screening), 4000+``kv`` (8192 tuples,
refinement) and 5000+``kv`` / 7001-7002 (32768-65536 tuples, validation).  Each
candidate is scored at all four rates by the nearest-codepoint MSE of the
unit-RMS-normalized codebook under one global scale, which is what EXL3's
``g_scale`` search optimizes (the encoder has one global scale per matrix, so a
per-codepoint scale would overstate every codebook).

512 seeded odd 32-bit candidates were screened on 1024 tuples/rate, the best 6
were re-scored on 8192 tuples/rate and the best 3 on 32768 tuples/rate.  The
winner is ``0xBC707477``.  Measured MSE per weight at 4.0 bpw (``kv = 16``, so
the state is the whole field and the trellis has no memory) on 65536 held-out
tuples::

    R4NX 0xBC707477   0.008673
    Q4NX at 4.0 bpw   0.008240    (the nibble-sum map with kv = 16)
    mcg 3-inst 4.0    0.0119      (3-inst codebook, not an NX family)

i.e. R4NX costs about 5% more reconstruction error than Q4NX at the same bpw.
Across all four rates the winner stays within 0-3.5% of the Q4NX codebook at
that rate (2.5: +0.0%, 3.0: +2.0%, 3.5: +3.5%, 4.0: +5.3%), while the median of
the 512 screened multipliers is 16% worse than Q4NX and the worst is 11x worse.
The spread across *good* multipliers is under 1% (the three finalists are within
0.4% of each other at 4.0 bpw), so the search is really a guard against the
bad tail rather than a way to beat Q4NX: R4NX trades those few percent of
reconstruction error for a cheaper decode, which is the point on RDNA4.

``R4NX_ALPHA``
--------------

``R4NX_ALPHA = 7 / 512`` normalizes the codebook to (almost) unit RMS, the same
convention as Q4NX's ``1 / sqrt(42.5)``.  The 65536 codepoints of the 4.0 bpw
codebook have mean square 5461.4523 per weight, so ``R4NX_ALPHA`` gives RMS
1.0104 -- within 1% of unit, which keeps ``g_scale`` in the same 0.1-1.9 range
Q4NX uses.

The value is exactly representable in binary16, which is what makes the HIP
decode bit-exact against this reference: every ``level * R4NX_ALPHA`` product is
an integer (at most 10 significant bits) times a power of two, so it is exactly
representable in both binary32 and binary16 and the ``half2`` path cannot round
differently from the FP32 oracle.  ``R4NX_ALPHA`` is also a power of two times
7, so ``target / R4NX_ALPHA`` in the encoder is exact for every float32 input
that reaches it.

The encoder
-----------

:func:`encode_r4nx_states` is the reference encoder the quantizer uses: a
Viterbi search over the packed bitstream.  Each step appends ``kv`` new bits
(the field) to a ``32 - 2 * kv`` bit overlap prefix, and the four weights of the
step are decoded from ``prefix | field << (32 - 2 * kv)``.  The state space is
small enough to search exhaustively for 3.5 and 4.0 bpw; 2.5 and 3.0 bpw use the
same bounded-beam search as the Q4NX kernel (``beam`` defaults to 128).

This is a portable torch implementation (CPU, CUDA or ROCm), so R4NX can be
encoded on the ROCm hosts the codebook targets, where the CUDA-only Q4NX
quantizer cannot run.  It is exact but not fast: a production encoder wants the
same Viterbi in a kernel, which is follow-up work.
"""

from __future__ import annotations

import torch


# Odd multiplier selected by the search documented in the module docstring.
R4NX_MULTIPLIER = 0xBC707477

# Mean square (per weight) of the 65536 codepoints of the 4.0 bpw codebook.
R4NX_MEAN_SQUARE = 5461.4523

# 7 / 512: exactly representable in binary16, so level * R4NX_ALPHA is exact in
# both binary32 and binary16 (see the module docstring).
R4NX_ALPHA = 7.0 / 512.0

# Rates that share the Q4NX packing: kv in {10, 12, 14, 16}.
R4NX_RATES = (2.5, 3.0, 3.5, 4.0)

# One tile is 16x16 weights: 64 steps of four weights.
R4NX_STEPS = 64
R4NX_TILE_WEIGHTS = 256

_UINT32_MASK = 0xFFFFFFFF


def r4nx_supports_rate(rate: int | float) -> bool:
    """Whether *rate* is an R4NX trellis rate (2.5, 3.0, 3.5 or 4.0 bpw)."""

    return rate in R4NX_RATES


def r4nx_step_bits(rate: int | float) -> int:
    """Packed bits per trellis step (``kv``) for an R4NX rate."""

    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate not in R4NX_RATES:
        raise ValueError(f"R4NX supports 2.5, 3.0, 3.5, or 4.0 bpw, got {rate!r}")
    return int(4 * rate)


def unpack_r4nx_states(packed: torch.Tensor) -> torch.Tensor:
    """Read the 64 zero-prefixed trellis states from each packed R4NX tile.

    Identical to :func:`gptqmodel.exllamav3.q4nx.unpack_q4nx_states`: the packed
    tile is ``4 * kv`` int16 words holding 64 ``kv``-bit fields, and a state is
    the ``32 - kv`` bit window ending at its field's last bit, with zeros before
    the first field of the tile.
    """

    if packed.shape[-1] % 4:
        raise ValueError("R4NX packed tiles must hold an integer number of 32-bit words")
    kv = packed.shape[-1] // 4
    if not r4nx_supports_rate(kv / 4):
        raise ValueError(f"R4NX packed tiles must hold 10, 12, 14, or 16 bit states, got {kv}")
    halves = packed.to(torch.int64) & 0xFFFF
    words = halves[..., 0::2] | (halves[..., 1::2] << 16)
    end = kv * torch.arange(1, R4NX_STEPS + 1, device=packed.device)
    index = (end - 1) // 32
    shift = 32 * (index + 1) - end
    previous = torch.where(index > 0, words[..., (index - 1).clamp_min(0)], 0)
    window = (previous << 32) | words[..., index]
    return (window >> shift) & ((1 << (32 - kv)) - 1)


def decode_r4nx_states(states: torch.Tensor) -> torch.Tensor:
    """Decode each state to four normalized weights in EXL3 fragment order.

    One 32-bit multiply, four signed bytes and the ``R4NX_ALPHA`` scale.  The
    multiply is done in int64 and masked to 32 bits so the result matches the
    hardware's ``v_mul_lo_u32`` regardless of the input tensor's integer width.
    """

    raw = states.to(torch.int64)
    # int16 state rows carry the packed uint16 field bit pattern; wider
    # tensors are already non-negative states, but mask for int32 callers.
    raw = raw & (0xFFFF if states.dtype == torch.int16 else _UINT32_MASK)
    product = (raw * R4NX_MULTIPLIER) & _UINT32_MASK
    shifts = torch.arange(0, 32, 8, dtype=torch.int64, device=states.device)
    octets = (product.unsqueeze(-1) >> shifts) & 0xFF
    levels = torch.where(octets < 128, octets, octets - 256)
    return levels.to(torch.float32) * R4NX_ALPHA


def r4nx_codebook(kv: int = 16, *, device=None) -> torch.Tensor:
    """The full R4NX codebook for a step, as ``(2 ** (32 - kv), 4)`` weights."""

    if kv not in (10, 12, 14, 16):
        raise ValueError(f"R4NX codebooks have 10, 12, 14, or 16 bit states, got {kv}")
    states = torch.arange(1 << (32 - kv), dtype=torch.int64, device=device)
    return decode_r4nx_states(states)


def pack_r4nx_states(suffixes: torch.Tensor, kv: int) -> torch.Tensor:
    """Pack ``(tiles, 64)`` state fields into the Q4NX bitstream layout.

    Fields are concatenated MSB first into 32-bit words; the returned int16
    tensor is the little-endian half-word view of those words, which is what
    ``unpack_q4nx_states`` and the native kernels expect.
    """

    if suffixes.ndim != 2 or suffixes.shape[1] != R4NX_STEPS:
        raise ValueError("R4NX packing expects (tiles, 64) state fields")
    if kv not in (10, 12, 14, 16):
        raise ValueError(f"R4NX states are 10, 12, 14, or 16 bits, got {kv}")
    device = suffixes.device
    fields = suffixes.to(torch.int64) & ((1 << kv) - 1)
    bit_shifts = torch.arange(kv - 1, -1, -1, dtype=torch.int64, device=device)
    bits = ((fields.unsqueeze(-1) >> bit_shifts) & 1).reshape(-1, R4NX_STEPS * kv)
    words = bits.reshape(-1, 2 * kv, 32)
    word_weights = torch.arange(31, -1, -1, dtype=torch.int64, device=device)
    words = (words * (1 << word_weights)).sum(-1).to(torch.int64)
    low = (words & 0xFFFF).to(torch.int16)
    high = (words >> 16).to(torch.int16)
    # The int16 stream is the little-endian half-word view of the 32-bit words,
    # which is what ``unpack_q4nx_states`` and the native kernels read back.
    return torch.stack((low, high), dim=-1).reshape(-1, 4 * kv)


def _decode_cost_states(states: torch.Tensor, target: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """Accumulate the squared error of ``states``' four levels onto *cost*."""

    product = ((states & _UINT32_MASK) * R4NX_MULTIPLIER) & _UINT32_MASK
    total = cost
    for byte in range(4):
        octet = (product >> (8 * byte)) & 0xFF
        level = torch.where(octet < 128, octet, octet - 256).to(torch.float32)
        total = total + (level - target[..., byte]).square()
    return total


@torch.no_grad()
def encode_r4nx_states(
    weights: torch.Tensor,
    kv: int,
    *,
    beam: int | None = None,
    tile_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(tiles, 256)`` weight tiles with the R4NX trellis.

    Returns ``(values, suffixes)``: the reconstructed weights in the input's
    units, and the 64 ``kv``-bit fields per tile.  The search is a Viterbi over
    the packed bitstream: at every step the field is appended to the overlap
    prefix and the four weights are decoded from the resulting state, so the
    chosen field is scored against the decode the loader will perform.

    ``beam`` bounds the number of live overlap prefixes.  It defaults to the
    whole state space where that is small (3.5 and 4.0 bpw) and to 128
    otherwise, matching the Q4NX kernel's beam.
    """

    if not isinstance(weights, torch.Tensor) or weights.ndim != 2 or weights.shape[1] != R4NX_TILE_WEIGHTS:
        raise ValueError("R4NX encoding expects (tiles, 256) weight tiles")
    if kv not in (10, 12, 14, 16):
        raise ValueError(f"R4NX states are 10, 12, 14, or 16 bits, got {kv}")
    if weights.dtype != torch.float32:
        weights = weights.to(torch.float32)

    prefix_bits = 32 - 2 * kv
    suffix_count = 1 << kv
    state_bits = 32 - kv
    prefix_mask = (1 << prefix_bits) - 1 if prefix_bits else 0
    class_count = 1 << prefix_bits
    if beam is None:
        beam = min(class_count, 128)
    else:
        beam = max(1, min(int(beam), class_count))

    device = weights.device
    tiles = weights.shape[0]
    fields = torch.arange(suffix_count, dtype=torch.int64, device=device)
    out_values = torch.empty((tiles, R4NX_TILE_WEIGHTS), dtype=torch.float32, device=device)
    out_suffixes = torch.empty((tiles, R4NX_STEPS), dtype=torch.int64, device=device)
    if tiles == 0:
        return out_values, out_suffixes

    # Work in codebook units, like the native Q4NX kernel (input / ALPHA).
    targets = (weights / R4NX_ALPHA).reshape(tiles, R4NX_STEPS, 4)
    key_limit = (1 << state_bits) - 1
    chunk = max(1, min(int(tile_chunk), tiles))

    for start in range(0, tiles, chunk):
        stop = min(start + chunk, tiles)
        target = targets[start:stop]
        count = stop - start
        prefixes = torch.zeros((count, 1), dtype=torch.int64, device=device)
        costs = torch.zeros((count, 1), dtype=torch.float32, device=device)
        history_keys: list[torch.Tensor] = []
        history_classes: list[torch.Tensor] = []

        for step in range(R4NX_STEPS):
            # The state is the big-endian sliding window ending at this field:
            # the overlap prefix is more significant than the new field.
            states = (prefixes.unsqueeze(-1) << kv) | fields.reshape(1, 1, -1)
            cost = _decode_cost_states(
                states,
                target[:, step].reshape(count, 1, 1, 4),
                costs.unsqueeze(-1),
            )
            # A non-negative float32 bit pattern sorts like the value, so a
            # single int64 key keeps (cost, state) ordered and breaks ties by
            # the smallest state.
            cost_bits = cost.reshape(count, -1).view(torch.int32).to(torch.int64) << state_bits
            keys = cost_bits | states.reshape(count, -1)
            next_prefix = states.reshape(count, -1) & prefix_mask
            best = torch.full(
                (count, class_count),
                (0x7F7FFFFF << state_bits) | key_limit,
                dtype=torch.int64,
                device=device,
            )
            best.scatter_reduce_(1, next_prefix, keys, reduce="amin", include_self=True)
            keep = min(beam, class_count)
            best_keys, best_classes = torch.topk(best, keep, dim=1, largest=False)
            costs = (best_keys >> state_bits).to(torch.int32).view(torch.float32)
            prefixes = best_classes
            history_keys.append(best_keys)
            history_classes.append(best_classes)

        # Backtrack: the final class picks the cheapest path; each earlier
        # class is the overlap prefix of the state that follows it.
        slot = costs.argmin(dim=1)
        rows = torch.arange(count, device=device)
        suffixes = torch.empty((count, R4NX_STEPS), dtype=torch.int64, device=device)
        for step in range(R4NX_STEPS - 1, -1, -1):
            state = history_keys[step][rows, slot] & key_limit
            suffixes[:, step] = state & (suffix_count - 1)
            if step:
                prefix = state >> kv
                slot = (history_classes[step - 1] == prefix.unsqueeze(-1)).to(torch.int8).argmax(dim=1)

        # Rebuild the states along the chosen path and decode them once.
        values = torch.empty((count, R4NX_STEPS, 4), dtype=torch.float32, device=device)
        state = torch.zeros((count,), dtype=torch.int64, device=device)
        state_mask = (1 << state_bits) - 1
        for step in range(R4NX_STEPS):
            state = ((state << kv) | suffixes[:, step]) & state_mask
            values[:, step] = decode_r4nx_states(state)
        out_values[start:stop] = values.reshape(count, R4NX_TILE_WEIGHTS)
        out_suffixes[start:stop] = suffixes

    return out_values, out_suffixes


__all__ = [
    "R4NX_ALPHA",
    "R4NX_MEAN_SQUARE",
    "R4NX_MULTIPLIER",
    "R4NX_RATES",
    "R4NX_STEPS",
    "R4NX_TILE_WEIGHTS",
    "decode_r4nx_states",
    "encode_r4nx_states",
    "pack_r4nx_states",
    "r4nx_codebook",
    "r4nx_step_bits",
    "r4nx_supports_rate",
    "unpack_r4nx_states",
]
