"""CPU exact state-decoder candidates for experiments 21/22/24/25/26.

P32 is a circular stream of 128 transitions, with 16 state *bits* (65536
states) and 2..7 bits per transition at rates 1..3.5 BPW. Output i is the
state AFTER edge i. Bank XOR/PGC16 lookup and all floating point arithmetic
remain in the canonical decoder; these candidates only replace state recovery.

For f(s)=((s << a) XOR b) & 65535, compose(f,g) means g(f(s)).
The shift-register case has disjoint suffix/state bits, so XOR equals OR.
A full circular transfer has a=16 and a unique fixed point b. There is no
16-entry transition table and no need to search 65536 initial states.

These are executable CPU references, not GPU kernels or performance results.
The existing direct circular-window extraction remains a required comparator.
Python integer bit planes model Boolean operations, not generated LOP3 code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

MASK = 0xFFFF
STEPS = 128


@dataclass(frozen=True)
class Affine:
    """Compact map on all 65536 start states; saturated shift plus XOR suffix."""

    shift: int = 0
    suffix: int = 0

    def __post_init__(self):
        if not 0 <= self.shift <= 16 or not 0 <= self.suffix <= MASK:
            raise ValueError("invalid 16-bit affine transfer")

    def __call__(self, state: int) -> int:
        return ((state << self.shift) ^ self.suffix) & MASK


def compose(first: Affine, second: Affine) -> Affine:
    """Chronological composition: first followed by second."""
    return Affine(min(16, first.shift + second.shift), second(first.suffix))


def _edges(edges: Sequence[int], transition_bits: int) -> tuple[int, ...]:
    if isinstance(transition_bits, bool) or transition_bits not in range(2, 8):
        raise ValueError("P32 transition_bits must be an integer in [2, 7]")
    transition_bits = index(transition_bits)
    values = tuple(index(e) for e in edges)
    if len(values) != STEPS or any(e < 0 or e >= 1 << transition_bits for e in values):
        raise ValueError("expected 128 unsigned transition symbols of the requested width")
    return values


def _prefix(items: Sequence[Affine]) -> list[Affine]:
    """Inclusive Hillis-Steele scan; each stage reads the previous stage."""
    result = list(items)
    distance = 1
    while distance < len(result):
        previous = result
        result = previous[:distance] + [
            compose(previous[i - distance], previous[i]) for i in range(distance, len(previous))
        ]
        distance *= 2
    return result


def affine_scan(edges: Sequence[int], transition_bits: int) -> list[int]:
    """21: compact associative scan and circular fixed-point evaluation."""
    edges = _edges(edges, transition_bits)
    prefixes = _prefix([Affine(transition_bits, e) for e in edges])
    initial = prefixes[-1].suffix
    return [f(initial) for f in prefixes]


def hierarchical_all_start(edges: Sequence[int], transition_bits: int, block_steps: int = 16) -> list[int]:
    """22: independently decode blocks symbolically for *every* initial state.

    Each local output is an Affine object encoding its value for all 65536
    starts. Scan block transfers, select the actual boundary state, then
    evaluate local output objects. This avoids materializing 65536 paths.
    Partial final blocks are supported, including block_steps=6.
    """
    edges = _edges(edges, transition_bits)
    if isinstance(block_steps, bool) or not isinstance(block_steps, int) or not 1 <= block_steps <= STEPS:
        raise ValueError("block_steps must be an integer in [1, 128]")
    local = [_prefix([Affine(transition_bits, e) for e in edges[i:i + block_steps]])
             for i in range(0, STEPS, block_steps)]
    boundaries = _prefix([block[-1] for block in local])
    initial = boundaries[-1].suffix
    starts = [initial] + [f(initial) for f in boundaries[:-1]]
    return [f(start) for block, start in zip(local, starts) for f in block]


def super_symbol(edges: Sequence[int], transition_bits: int, steps: int = 4) -> list[int]:
    """24: pack 2/4/6/8 chronological symbols into a factored super-symbol.

    The packed word is the exact suffix of the multi-step transfer. No
    exponential LUT is allocated. Intermediate outputs use packed prefixes;
    only super-symbol boundaries participate in the associative scan.
    """
    edges = _edges(edges, transition_bits)
    if isinstance(steps, bool) or steps not in (2, 4, 6, 8):
        raise ValueError("super-symbol steps must be 2, 4, 6, or 8")
    steps = index(steps)
    groups = []
    for offset in range(0, STEPS, steps):
        group = edges[offset:offset + steps]
        packed = 0
        for edge in group:
            packed = (packed << transition_bits) | edge
        groups.append((packed, len(group)))
    boundaries = _prefix([Affine(min(16, n * transition_bits), word & MASK) for word, n in groups])
    initial = boundaries[-1].suffix
    starts = [initial] + [f(initial) for f in boundaries[:-1]]
    return [((start << (j * transition_bits)) | (word >> ((n - j) * transition_bits))) & MASK
            for (word, n), start in zip(groups, starts) for j in range(1, n + 1)]


def bitsliced(streams: Sequence[Sequence[int]], transition_bits: int) -> list[list[int]]:
    """25: 16 bit planes advance up to 32 independent streams at once.

    A warm-up pass forgets the unknown start state. A second pass emits the
    circular states. Packing/unpacking cost is deliberately visible here.
    Groups of 32 and a masked partial group support arbitrary stream counts.
    """
    streams = [_edges(stream, transition_bits) for stream in streams]
    # Validate even an empty batch.
    _edges([0] * STEPS, transition_bits)
    output = []
    for offset in range(0, len(streams), 32):
        group = streams[offset:offset + 32]
        planes = [[sum(((stream[t] >> bit) & 1) << lane for lane, stream in enumerate(group))
                   for bit in range(transition_bits)] for t in range(STEPS)]
        state = [0] * 16  # least significant plane first
        for edge in planes:
            state = edge + state[:16 - transition_bits]
        decoded = [[] for _ in group]
        for edge in planes:
            state = edge + state[:16 - transition_bits]
            for lane, row in enumerate(decoded):
                row.append(sum(((plane >> lane) & 1) << bit for bit, plane in enumerate(state)))
        output.extend(decoded)
    return output


def _gf2_compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Compose augmented 17x17 GF(2) matrices stored as row bitmasks."""
    return tuple(_xor_rows(first, row) for row in second)


def _xor_rows(rows: tuple[int, ...], selection: int) -> int:
    value = 0
    while selection:
        low = selection & -selection
        value ^= rows[low.bit_length() - 1]
        selection ^= low
    return value


def gf2_scan(edges: Sequence[int], transition_bits: int) -> list[int]:
    """26: explicit augmented GF(2) jump-ahead scan, independent of Affine.

    This intentionally retains the general matrix representation to check the
    compact specialization. The seventeenth coordinate is the constant one.
    PGC16 mixing is nonlinear and is NOT folded into these matrices.
    """
    edges = _edges(edges, transition_bits)
    matrices = [tuple(((1 << (bit - transition_bits)) if bit >= transition_bits else 0)
                      | (((edge >> bit) & 1) << 16) for bit in range(16)) + (1 << 16,)
                for edge in edges]
    distance = 1
    while distance < STEPS:
        previous = matrices
        matrices = previous[:distance] + [_gf2_compose(previous[i - distance], previous[i])
                                         for i in range(distance, STEPS)]
        distance *= 2
    initial = sum(((row >> 16) & 1) << bit for bit, row in enumerate(matrices[-1][:16]))
    augmented = initial | (1 << 16)
    return [sum(((row & augmented).bit_count() & 1) << bit for bit, row in enumerate(matrix[:16]))
            for matrix in matrices]


CANDIDATES = {21: affine_scan, 22: hierarchical_all_start, 24: super_symbol, 26: gf2_scan}


def decode_planar_states(trellis, *, bits: float, experiment: int, **kwargs):
    """CPU tensor adapter: canonical planar payload -> int64 circular states.

    Only uses canonical unpacking for edge fields, never for recovered states.
    Does not move GPU tensors to CPU implicitly or mutate the input.
    """
    import torch

    from gptqmodel.utils.planar_packing import planar_unpack_rows

    if bits not in (1, 1.5, 2, 2.5, 3, 3.5):
        raise ValueError("unsupported P32 rate")
    if trellis.device.type != "cpu":
        raise ValueError("algebraic references are CPU only")
    if trellis.dtype != torch.int32 or trellis.ndim < 1 or trellis.shape[-1] != int(bits * 8):
        raise ValueError("expected planar int32 P32 circular128 payload")
    if experiment not in (*CANDIDATES, 25):
        raise ValueError("expected experiment 21, 22, 24, 25, or 26")
    width = int(bits * 2)
    flat = trellis.reshape(-1, trellis.shape[-1])
    edges = planar_unpack_rows(flat.T.contiguous(), width).T.tolist()
    if experiment == 25:
        if kwargs:
            raise ValueError("bitsliced candidate takes no tuning parameters")
        states = bitsliced(edges, width)
    else:
        states = [CANDIDATES[experiment](row, width, **kwargs) for row in edges]
    return torch.tensor(states, dtype=torch.int64).reshape(*trellis.shape[:-1], STEPS)
