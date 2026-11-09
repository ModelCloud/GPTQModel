# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the nf12 reference implementation
# (https://github.com/dphnAI/nf12) by AlpinDale and the nf12 contributors.
"""Pure-Torch NF12-AD codec and inference reference.

NF12 (NyoomFloat12) is a lossless 12-bit encoding of BF16 weights from
"NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight
Compression" by Liberman, Fang, Zhang, Dao, and Fu (Together AI, 2026).

This module is a Torch port of the reference implementation published at
https://github.com/dphnAI/nf12 by AlpinDale. It implements the AlpinDale
modified ("AD") tiled layout used by the vendored CUDA kernels in
``gptqmodel_ext/nf12``. Encoding is bit-exact: ``decode(encode(w))`` equals
``w`` for every BF16 input, including signed zero, subnormals, Inf, and NaN
payloads.

The layout is described by ``gptqmodel_ext/nf12/README.md``. A group is eight
BF16 slots that decode into the four ``A`` fragments of one lane of
``mma.sync.aligned.m16n8k16``.
"""

from __future__ import annotations

import torch


MAX_EXP_BASE = 120
NUM_EXP_BASES = MAX_EXP_BASE + 1
SUPER_ATOM_BYTES = 1536
SUPER_ATOM_BYTES_BF16T = 2048

# [N, K] -> [nb, rh, g, kb, j, ch, t, p] where
#   row = 16*nb + 8*rh + g, col = 64*kb + 16*j + 8*ch + 2*t + p.
# SPEC 4.1: lane l = 4*g + t and slot s = 4*ch + 2*rh + p.
_TO_GROUPS = (0, 3, 4, 2, 6, 5, 1, 7)
_FROM_GROUPS = tuple(_TO_GROUPS.index(dim) for dim in range(8))


def check_shape(n: int, k: int) -> None:
    """Validate the NF12 matrix constraints."""

    if n <= 0 or k <= 0 or n % 16 or k % 64:
        raise ValueError(f"NF12 needs N % 16 == 0 and K % 64 == 0, got N={n}, K={k}")


def to_groups(w: torch.Tensor) -> torch.Tensor:
    """Return ``[N*K/8, 8]`` slot values in flat-group-id order."""

    if w.dim() != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(w.shape)}")
    n, k = w.shape
    check_shape(n, k)
    view = w.reshape(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(_TO_GROUPS)
    return view.reshape(-1, 8)


def from_groups(groups: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of :func:`to_groups`."""

    check_shape(n, k)
    view = groups.reshape(n // 16, k // 64, 4, 8, 4, 2, 2, 2).permute(_FROM_GROUPS)
    return view.reshape(n, k)


def _as_int16(w: torch.Tensor) -> torch.Tensor:
    if w.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {w.dtype}")
    return w.contiguous().view(torch.int16)


def _bits(w: torch.Tensor) -> torch.Tensor:
    return _as_int16(w).to(torch.int32) & 0xFFFF


def _exp_range(gbits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per group: (min exponent, max exponent, all signs positive)."""

    exponent = (gbits >> 7) & 0xFF
    return exponent.amin(1), exponent.amax(1), (gbits < 0x8000).all(1)


def escape_counts_all_bases(w: torch.Tensor) -> torch.Tensor:
    """Escaped-group count for every exponent base in ``[0, 120]``."""

    gbits = to_groups(_bits(w))
    emin, emax, positive = _exp_range(gbits)
    low = ((emax - 14) // 2).clamp(min=0)
    high = (emin // 2).clamp(max=MAX_EXP_BASE)
    fits = low <= high
    sentinel = positive & ((emin >> 1) == (emax >> 1)) & ((emin >> 1) <= MAX_EXP_BASE)
    width = NUM_EXP_BASES + 1

    def hist(index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.bincount(index[mask], minlength=width)

    ok = (hist(low, fits) - hist(high + 1, fits)).cumsum(0)[:NUM_EXP_BASES]
    ok -= hist(emin >> 1, sentinel)[:NUM_EXP_BASES]
    return gbits.shape[0] - ok


def choose_exp_base(w: torch.Tensor) -> int:
    """Return the base minimizing escaped groups, ties to the smallest base."""

    # argmin returns the first minimum, which is the smallest base.
    return int(escape_counts_all_bases(w).argmin())


def group_bytes(w: torch.Tensor, exp_base: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every group to ``(uint8 [G, 12], int16 [E, 8])``."""

    if not 0 <= exp_base <= MAX_EXP_BASE:
        raise ValueError(f"exp_base must be in [0, {MAX_EXP_BASE}], got {exp_base}")
    graw = to_groups(_as_int16(w))
    gbits = graw.to(torch.int32) & 0xFFFF
    exponent = (gbits >> 7) & 0xFF
    offset = exponent - 2 * exp_base
    in_window = ((offset >= 0) & (offset <= 15)).all(1)
    offset_hi = (offset >> 1) & 7
    sign = gbits >> 15
    meta = (
        (sign[:, 0::2] << 7)
        | (sign[:, 1::2] << 6)
        | (offset_hi[:, 1::2] << 3)
        | offset_hi[:, 0::2]
    )
    escaped = ~in_window | (meta == 0).all(1)
    out = torch.cat([meta, gbits & 0xFF], dim=1)

    escape_ids = escaped.nonzero().squeeze(1)
    index = torch.arange(escape_ids.numel(), device=w.device, dtype=torch.int64)
    escape_row = torch.zeros(escape_ids.numel(), 12, dtype=out.dtype, device=w.device)
    shifts = torch.tensor([0, 8, 16, 24], device=w.device)
    escape_row[:, 4:8] = ((index.view(-1, 1) >> shifts) & 0xFF).to(out.dtype)
    out[escape_ids] = escape_row
    return out.to(torch.uint8), graw[escape_ids].contiguous()


def encode(
    w: torch.Tensor, exp_base: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode a BF16 ``[N, K]`` matrix to ``(packed, escapes, exp_base)``."""

    if w.dim() != 2:
        raise ValueError(f"NF12 expects a 2-D matrix, got shape {tuple(w.shape)}")
    n, k = w.shape
    check_shape(n, k)
    if w.dtype != torch.bfloat16:
        raise TypeError(f"NF12 encodes bfloat16 weights, got {w.dtype}")
    w = w.contiguous()
    if exp_base is None:
        exp_base = choose_exp_base(w)
    groups, escapes = group_bytes(w, exp_base)
    super_atom = groups.view(-1, 4, 32, 12)  # [super-atom, atom j, lane, 12]
    # Plane 0: lane l holds M of atoms 0..3 at 16l + 4j.
    plane0 = super_atom[..., :4].permute(0, 2, 1, 3).reshape(-1, 1, 512)
    # Planes 1, 2: lane l holds (L0, L1) of atoms 2p..2p+1 at 16l + 8(j - 2p).
    planes12 = (
        super_atom[..., 4:]
        .reshape(-1, 2, 2, 32, 8)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, 2, 512)
    )
    packed = torch.cat([plane0, planes12], dim=1).reshape(n, k * 3 // 2)
    return packed, escapes, exp_base


def decode(
    packed: torch.Tensor,
    escapes: torch.Tensor,
    exp_base: int,
    n: int | None = None,
    k: int | None = None,
) -> torch.Tensor:
    """Decode NF12 tiled bytes back to the exact BF16 ``[N, K]`` matrix."""

    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("packed must be a 2-D uint8 tensor")
    n = packed.shape[0] if n is None else n
    k = packed.shape[1] * 2 // 3 if k is None else k
    check_shape(n, k)
    if packed.numel() != n * k * 3 // 2:
        raise ValueError(f"packed has {packed.numel()} bytes, expected {n * k * 3 // 2}")
    if not 0 <= exp_base <= MAX_EXP_BASE:
        raise ValueError(f"exp_base must be in [0, {MAX_EXP_BASE}], got {exp_base}")
    super_atom = packed.reshape(-1, 3, 512).to(torch.int32)
    meta = super_atom[:, 0].reshape(-1, 32, 4, 4).permute(0, 2, 1, 3).reshape(-1, 4)
    low = (
        super_atom[:, 1:]
        .reshape(-1, 2, 32, 2, 8)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, 8)
    )

    high = torch.empty_like(low)
    high[:, 0::2] = (meta & 0x87) + exp_base
    high[:, 1::2] = (((meta << 1) & 0x80) | ((meta >> 3) & 0x07)) + exp_base
    values = (high << 8) | low
    values = (values - ((values >> 15) << 16)).to(torch.int16)

    escaped = (meta == 0).all(1)
    if bool(escaped.any()):
        index = (
            low[escaped, 0]
            | (low[escaped, 1] << 8)
            | (low[escaped, 2] << 16)
            | (low[escaped, 3].to(torch.int64) << 24)
        )
        if escapes.dim() != 2 or escapes.shape[1] != 8 or escapes.dtype != torch.int16:
            raise TypeError("escapes must be int16 [E, 8]")
        if int(index.max()) >= escapes.shape[0]:
            raise ValueError("escape index out of range")
        values[escaped] = escapes.to(values.device)[index.to(torch.int64)]
    return from_groups(values, n, k).contiguous().view(torch.bfloat16)


def pack_bf16t(w: torch.Tensor) -> torch.Tensor:
    """Return the uncompressed BF16-tiled twin layout as uint8 ``[N, 2K]``."""

    n, k = w.shape
    check_shape(n, k)
    groups = to_groups(_as_int16(w)).contiguous()
    return groups.view(torch.uint8).reshape(n, 2 * k)


def unpack_bf16t(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of :func:`pack_bf16t`."""

    if packed.dtype != torch.uint8:
        raise TypeError("packed must be uint8")
    groups = packed.contiguous().view(-1, 16).view(torch.int16)
    return from_groups(groups, n, k).contiguous().view(torch.bfloat16)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP32-accumulated ``x @ weight.T (+ bias)`` rounded once to BF16.

    This is the oracle, not the tiled kernel: accumulation order differs from
    the CUDA kernels, so use a relative tolerance when comparing.
    """

    y = x.float() @ weight.float().T
    if bias is not None:
        y = y + bias.float()
    return y.to(torch.bfloat16)


def encode_weight(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    exp_base: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cast a 2-D weight to BF16 and encode it.

    The BF16 cast is the only lossy step in NF12-AD quantization; the encode
    itself is bit-exact.
    """

    if weight.dim() != 2:
        raise ValueError(f"NF12-AD expects a 2-D weight, got shape {tuple(weight.shape)}")
    matrix = weight.detach().to(device="cpu", dtype=dtype).contiguous()
    return encode(matrix, exp_base=exp_base)


__all__ = [
    "MAX_EXP_BASE",
    "SUPER_ATOM_BYTES",
    "SUPER_ATOM_BYTES_BF16T",
    "check_shape",
    "to_groups",
    "from_groups",
    "escape_counts_all_bases",
    "choose_exp_base",
    "group_bytes",
    "encode",
    "decode",
    "pack_bf16t",
    "unpack_bf16t",
    "linear",
    "encode_weight",
]
