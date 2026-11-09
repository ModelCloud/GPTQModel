# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# NF12 / NF12-AD reference implementation.
#
# Original implementation: https://github.com/dphnAI/nf12
# Copyright (c) 2026 AlpinDale and the nf12 contributors.
# Algorithm: "NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit
# Weight Compression", Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).
# Vendored into GPT-QModel under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------
"""Pure-PyTorch reference implementation of the NF12 format (SPEC.md).

The CUDA tests use this reference on CPU or CUDA tensors without loading
the compiled extension.

A *group* is 8 BF16 values (slots s0..s7) that decode into the four
A-fragment registers of one lane of `mma.m16n8k16`. Groups are enumerated
in *flat group id* order (SPEC 4.3).
"""

from __future__ import annotations

import torch

__all__ = [
    "MAX_EXP_BASE",
    "SUPER_ATOM_BYTES",
    "SUPER_ATOM_BYTES_BF16T",
    "check_shape",
    "group_coords",
    "group_slots",
    "to_groups",
    "from_groups",
    "group_bytes",
    "escape_counts_all_bases",
    "choose_exp_base",
    "encode",
    "decode",
    "pack_bf16t",
    "unpack_bf16t",
    "linear",
    "embedding",
    "encode_moe",
    "moe_align",
    "moe_linear",
    "moe_combine",
    "moe_silu_mul",
    "fused_moe",
]

MAX_EXP_BASE = 120
NUM_EXP_BASES = MAX_EXP_BASE + 1
SUPER_ATOM_BYTES = 1536
SUPER_ATOM_BYTES_BF16T = 2048

# Matrix view [N, K] -> [nb, rh, g, kb, j, ch, t, p] with
#   row = 16*nb + 8*rh + g,  col = 64*kb + 16*j + 8*ch + 2*t + p.
# SPEC 4.1: lane l = 4*g + t and slot s = 4*ch + 2*rh + p. Groups are ordered
# by flat group id (nb, kb, j, l) and slots by s, which is this permutation:
_TO_GROUPS = (0, 3, 4, 2, 6, 5, 1, 7)
_FROM_GROUPS = tuple(_TO_GROUPS.index(d) for d in range(8))


def check_shape(n: int, k: int) -> None:
    if n <= 0 or k <= 0 or n % 16 or k % 64:
        raise ValueError(f"NF12 needs N % 16 == 0 and K % 64 == 0, got N={n}, K={k}")


def to_groups(w: torch.Tensor) -> torch.Tensor:
    """[N, K] tensor -> [N*K/8, 8] slot values in flat-group-id order."""
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(w.shape)}")
    n, k = w.shape
    check_shape(n, k)
    v = w.reshape(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(_TO_GROUPS)
    return v.reshape(-1, 8)


def from_groups(groups: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of `to_groups`: [N*K/8, 8] -> [N, K]."""
    check_shape(n, k)
    v = groups.reshape(n // 16, k // 64, 4, 8, 4, 2, 2, 2).permute(_FROM_GROUPS)
    return v.reshape(n, k)


def group_coords(n: int, k: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) of every slot, each int64 [N*K/8, 8] in flat-group-id order."""
    check_shape(n, k)
    rows = torch.arange(n, device=device).view(n, 1).expand(n, k)
    cols = torch.arange(k, device=device).view(1, k).expand(n, k)
    return to_groups(rows), to_groups(cols)


def group_slots(n: int, k: int, device=None) -> torch.Tensor:
    """Row-major flat element index (row*K + col) of every slot, int64 [N*K/8, 8]."""
    rows, cols = group_coords(n, k, device)
    return rows * k + cols


def _as_int16(w: torch.Tensor) -> torch.Tensor:
    """BF16 tensor -> same bits as a contiguous int16 tensor."""
    if w.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {w.dtype}")
    return w.contiguous().view(torch.int16)


def _bits(w: torch.Tensor) -> torch.Tensor:
    """BF16 tensor -> its bit patterns as int32 in [0, 65535]."""
    return _as_int16(w).to(torch.int32) & 0xFFFF


def _exp_range(gbits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per group: (min exponent, max exponent, all signs positive)."""
    e = (gbits >> 7) & 0xFF
    return e.amin(1), e.amax(1), (gbits < 0x8000).all(1)


def escape_counts_all_bases(w: torch.Tensor, per_row_block: bool = False) -> torch.Tensor:
    """Escaped-group count for every exponent base B in [0, 120].

    A group fits the window [2B, 2B+15] for an interval of bases
    [ceil((emax-15)/2), floor(emin/2)], minus the single base B = emin >> 1
    when the meta word would be the sentinel (all signs positive and every
    exponent in {2B, 2B+1}). Counting is therefore two histograms, not 121
    encodes. Returns int64 [121], or [N/16, 121] if `per_row_block`.
    """
    gbits = to_groups(_bits(w))
    n = w.shape[0]
    emin, emax, pos = _exp_range(gbits)
    lo = ((emax - 14) // 2).clamp(min=0)
    hi = (emin // 2).clamp(max=MAX_EXP_BASE)
    fits = lo <= hi
    sentinel = pos & ((emin >> 1) == (emax >> 1)) & ((emin >> 1) <= MAX_EXP_BASE)
    groups_per_block = gbits.shape[0] * 16 // n
    nblocks = n // 16 if per_row_block else 1
    block = torch.arange(gbits.shape[0], device=gbits.device) // groups_per_block if per_row_block else None
    width = NUM_EXP_BASES + 1

    def hist(idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if block is not None:
            idx = block * width + idx
        return torch.bincount(idx[mask], minlength=nblocks * width).view(nblocks, width)

    ok = (hist(lo, fits) - hist(hi + 1, fits)).cumsum(1)[:, :NUM_EXP_BASES]
    ok -= hist(emin >> 1, sentinel)[:, :NUM_EXP_BASES]
    total = gbits.shape[0] // nblocks
    counts = total - ok
    return counts if per_row_block else counts[0]


def choose_exp_base(w: torch.Tensor) -> int:
    """SPEC 2: the base minimizing escaped groups, ties to the smallest base."""
    # argmin returns the first minimum, which is the smallest base.
    return int(escape_counts_all_bases(w).argmin())


def group_bytes(w: torch.Tensor, exp_base: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every group: (uint8 [G, 12] = M | L0 | L1 little-endian, escapes int16 [E, 8])."""
    if not 0 <= exp_base <= MAX_EXP_BASE:
        raise ValueError(f"exp_base must be in [0, {MAX_EXP_BASE}], got {exp_base}")
    graw = to_groups(_as_int16(w))
    gbits = graw.to(torch.int32) & 0xFFFF
    e = (gbits >> 7) & 0xFF
    off = e - 2 * exp_base
    in_window = ((off >= 0) & (off <= 15)).all(1)
    o1 = (off >> 1) & 7
    sign = gbits >> 15
    # Meta byte j packs the slot pair (2j, 2j+1), SPEC 3.
    meta = (sign[:, 0::2] << 7) | (sign[:, 1::2] << 6) | (o1[:, 1::2] << 3) | o1[:, 0::2]
    escaped = ~in_window | (meta == 0).all(1)
    out = torch.cat([meta, gbits & 0xFF], dim=1)

    esc_ids = escaped.nonzero().squeeze(1)  # ascending flat group id
    idx = torch.arange(esc_ids.numel(), device=w.device, dtype=torch.int64)
    esc_row = torch.zeros(esc_ids.numel(), 12, dtype=out.dtype, device=w.device)
    esc_row[:, 4:8] = ((idx.view(-1, 1) >> torch.tensor([0, 8, 16, 24], device=w.device)) & 0xFF).to(out.dtype)
    out[esc_ids] = esc_row
    return out.to(torch.uint8), graw[esc_ids].contiguous()


def encode(w: torch.Tensor, exp_base: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a BF16 [N, K] matrix to NF12 tiled.

    Returns (packed uint8 [N, K*3/2], escapes int16 [E, 8], exp_base). With
    `exp_base=None` the base is chosen per SPEC 2; forcing a base is useful
    for tests and studies.
    """
    n, k = w.shape
    if exp_base is None:
        exp_base = choose_exp_base(w)
    gb, escapes = group_bytes(w, exp_base)
    sa = gb.view(-1, 4, 32, 12)  # [super-atom, atom j, lane, 12]
    # Plane 0: lane l holds M of atoms 0..3 at 16l + 4j.
    plane0 = sa[..., :4].permute(0, 2, 1, 3).reshape(-1, 1, 512)
    # Planes 1, 2: lane l holds (L0, L1) of atoms 2p..2p+1 at 16l + 8(j - 2p).
    planes12 = sa[..., 4:].reshape(-1, 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(-1, 2, 512)
    packed = torch.cat([plane0, planes12], dim=1).reshape(n, k * 3 // 2)
    return packed, escapes, exp_base


def decode(packed: torch.Tensor, escapes: torch.Tensor, exp_base: int, n: int | None = None,
           k: int | None = None) -> torch.Tensor:
    """Decode NF12 tiled back to the exact BF16 [N, K] matrix."""
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("packed must be a 2-D uint8 tensor")
    n = packed.shape[0] if n is None else n
    k = packed.shape[1] * 2 // 3 if k is None else k
    check_shape(n, k)
    if packed.numel() != n * k * 3 // 2:
        raise ValueError(f"packed has {packed.numel()} bytes, expected {n * k * 3 // 2}")
    if not 0 <= exp_base <= MAX_EXP_BASE:
        raise ValueError(f"exp_base must be in [0, {MAX_EXP_BASE}], got {exp_base}")
    sa = packed.reshape(-1, 3, 512).to(torch.int32)
    meta = sa[:, 0].reshape(-1, 32, 4, 4).permute(0, 2, 1, 3).reshape(-1, 4)
    lo = sa[:, 1:].reshape(-1, 2, 32, 2, 8).permute(0, 1, 3, 2, 4).reshape(-1, 8)

    hi = torch.empty_like(lo)
    hi[:, 0::2] = (meta & 0x87) + exp_base
    hi[:, 1::2] = (((meta << 1) & 0x80) | ((meta >> 3) & 0x07)) + exp_base
    vals = (hi << 8) | lo
    vals = (vals - ((vals >> 15) << 16)).to(torch.int16)  # [0, 65535] -> same bits as int16

    escaped = (meta == 0).all(1)
    if bool(escaped.any()):
        idx = lo[escaped, 0] | (lo[escaped, 1] << 8) | (lo[escaped, 2] << 16) | (lo[escaped, 3].to(torch.int64) << 24)
        if escapes.dim() != 2 or escapes.shape[1] != 8 or escapes.dtype != torch.int16:
            raise TypeError("escapes must be int16 [E, 8]")
        if int(idx.max()) >= escapes.shape[0]:
            raise ValueError("escape index out of range")
        vals[escaped] = escapes.to(vals.device)[idx.to(torch.int64)]
    return from_groups(vals, n, k).contiguous().view(torch.bfloat16)


def pack_bf16t(w: torch.Tensor) -> torch.Tensor:
    """BF16 [N, K] -> bf16t twin layout (SPEC 4.4) as uint8 [N, 2K].

    Lane l of atom j stores R0..R3 (slots s0..s7 as little-endian BF16) at
    512j + 16l, which is exactly flat-group-id order at 16 bytes per group.
    """
    n, k = w.shape
    g = to_groups(_as_int16(w)).contiguous()
    return g.view(torch.uint8).reshape(n, 2 * k)


def unpack_bf16t(packed: torch.Tensor, n: int | None = None, k: int | None = None) -> torch.Tensor:
    """Inverse of `pack_bf16t`."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed must be uint8")
    n = packed.shape[0] if n is None else n
    k = packed.numel() // (2 * n) if k is None else k
    g = packed.contiguous().view(-1, 16).view(torch.int16)
    return from_groups(g, n, k).contiguous().view(torch.bfloat16)


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """FP32-accumulated `x @ w.T (+ bias)`, rounded once to BF16.

    The accumulation order differs from the kernels, so use a relative tolerance
    here. NF12 kernels must match their bf16t twins without tolerance.
    """
    y = x.float() @ w.float().T
    if bias is not None:
        y = y + bias.float()
    return y.to(torch.bfloat16)


def embedding(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """SPEC 10 on a BF16 table w [V, H]: preserve rows w[ids], shaped [*ids.shape, H].
    IDs outside [0, V) give +0.0 rows."""
    ids = ids.to(device=w.device, dtype=torch.int64)
    valid = (ids >= 0) & (ids < w.shape[0])
    rows = _as_int16(w)[torch.where(valid, ids, 0)]
    return torch.where(valid.unsqueeze(-1), rows, 0).view(torch.bfloat16)


# Mixture-of-experts (SPEC 9).


def encode_moe(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode stacked BF16 experts [E, N, K] (SPEC 9.1): every expert exactly as `encode`.

    Returns (packed uint8 [E, N, K*3/2], escapes int16 [S, 8] expert-major,
    escape_offsets int64 [E + 1], exp_base uint8 [E]).
    """
    parts = [encode(we) for we in w]
    counts = torch.tensor([p[1].shape[0] for p in parts], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return (torch.stack([p[0] for p in parts]), torch.cat([p[1] for p in parts]), offsets.to(w.device),
            torch.tensor([p[2] for p in parts], dtype=torch.uint8, device=w.device))


def moe_align(topk_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """SPEC 9.2: (sorted int32 [T*topk], expert_offsets int64 [E + 1]).

    Assignment ids `t*topk + j` stably sorted by expert; ids outside [0, E) are
    dropped. Positions past `expert_offsets[E]` hold the dropped ids here, but
    the spec leaves them unspecified.
    """
    ids = topk_ids.reshape(-1).to(torch.int64)
    valid = (ids >= 0) & (ids < num_experts)
    key = torch.where(valid, ids, num_experts)
    order = torch.sort(key, stable=True).indices
    counts = torch.bincount(key, minlength=num_experts + 1)[:num_experts]
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return order.to(torch.int32), offsets


def moe_linear(x: torch.Tensor, w: torch.Tensor, expert_offsets: torch.Tensor, gather: torch.Tensor | None = None,
               gather_div: int = 1, scatter: torch.Tensor | None = None, row_scale: torch.Tensor | None = None,
               num_rows: int | None = None) -> torch.Tensor:
    """SPEC 9.2 on BF16 experts w [E, N, K]: y[out(p)] = x[in(p)] . W_e^T (* row_scale[out(p)]).

    FP32 accumulation and scaling, one BF16 rounding. Rows no valid position
    writes are zero. `num_rows` defaults to the gather/scatter length, else the
    rows of x.
    """
    if num_rows is None:
        num_rows = gather.shape[0] if gather is not None else scatter.shape[0] if scatter is not None else x.shape[0]
    y = torch.zeros(num_rows, w.shape[1], dtype=torch.bfloat16, device=x.device)
    offsets = expert_offsets.tolist()
    for e in range(w.shape[0]):
        p = torch.arange(offsets[e], offsets[e + 1], device=x.device)
        rows_in = gather[p].long() // gather_div if gather is not None else p
        rows_out = scatter[p].long() if scatter is not None else p
        acc = x[rows_in].float() @ w[e].float().T
        if row_scale is not None:
            acc = acc * row_scale[rows_out].float().unsqueeze(1)
        y[rows_out] = acc.to(torch.bfloat16)
    return y


def moe_combine(y: torch.Tensor, topk: int) -> torch.Tensor:
    """SPEC 9.2: out[t] = sum_j y[t*topk + j], FP32 in increasing j, one BF16 rounding."""
    v = y.view(-1, topk, y.shape[-1]).float()
    acc = v[:, 0]
    for j in range(1, topk):
        acc = acc + v[:, j]
    return acc.to(torch.bfloat16)


def moe_silu_mul(h: torch.Tensor) -> torch.Tensor:
    """SPEC 9.2: out = bf16(s * u), s = g / (1 + expf(-g)), g/u the FP32 gate/up halves of h."""
    inter = h.shape[-1] // 2
    g = h[:, :inter].float()
    u = h[:, inter:].float()
    s = g / (1 + torch.exp(-g))
    return (s * u).to(torch.bfloat16)


def fused_moe(x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
              topk_ids: torch.Tensor, expert_map: torch.Tensor | None = None) -> torch.Tensor:
    """SPEC 9.2 typical layer on BF16 experts (w13 [E, 2I, K] gate rows first, w2 [E, K2, I]), SiLU-gated.

    Rounds to BF16 where `nf12.fused_moe` does: after w13, after the activation
    (`moe_silu_mul`: FP32, one BF16 rounding) and after the scaled w2.
    """
    topk = topk_ids.shape[1]
    if expert_map is not None:
        # Global ids outside [0, global_E) are dropped, like the local -1 entries.
        valid = (topk_ids >= 0) & (topk_ids < expert_map.shape[0])
        topk_ids = torch.where(valid, expert_map[torch.where(valid, topk_ids, 0)], -1)
    sorted_ids, offsets = moe_align(topk_ids, w13.shape[0])
    h = moe_linear(x, w13, offsets, gather=sorted_ids, gather_div=topk)
    h = moe_silu_mul(h)
    y = moe_linear(h, w2, offsets, scatter=sorted_ids, row_scale=topk_weights.reshape(-1).float())
    return moe_combine(y, topk)
