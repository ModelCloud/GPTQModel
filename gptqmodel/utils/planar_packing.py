# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Split-plane (planar) bit packing for the GPTQ `gptq_p` checkpoint format.

A logical `b`-bit code is stored as word-aligned bit planes whose widths each
divide 32, so every packed word contains whole fields and no code straddles a
word boundary (unlike the continuous 3-bit layout):

- 2-bit = single 2-bit plane (bit-identical to the continuous 2-bit layout)
- 3-bit = 2-bit low plane + 1-bit high plane
- 4-bit = single 4-bit plane (bit-identical to the continuous 4-bit layout)
- 5-bit = 4-bit low plane + 1-bit high plane
- 6-bit = 4-bit low plane + 2-bit high plane
- 7-bit = 4-bit low plane + 2-bit mid plane + 1-bit top plane
- 8-bit = single 8-bit plane (bit-identical to the continuous 8-bit layout)

For every 32 consecutive logical codes the planes are stored adjacently as
`bits` int32 words: first the low-plane words, then the higher planes. Within a
plane of width `w`, word `i` holds codes `[i*(32//w), (i+1)*(32//w))` at shifts
`w*j`, matching the row-major `[bits, pack_factor]` convention of the 2/4/8-bit
packers. The layout fills exactly `ceil(n * bits / 32)` words, so qweight and
qzeros keep their standard GPTQ shapes.
"""

from typing import Tuple

import torch


# Bit widths that have no continuous layout and are always stored planar.
PLANAR_BITS: Tuple[int, ...] = (5, 6, 7)
# All bit widths the planar (gptq_p) layout supports.
PLANAR_FORMAT_BITS: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

# bits -> ((plane_width, bit_offset), ...) ordered low to high. Single-plane
# widths (2/4/8) produce words bit-identical to the continuous layout.
_PLANES = {
    2: ((2, 0),),
    3: ((2, 0), (1, 2)),
    4: ((4, 0),),
    5: ((4, 0), (1, 4)),
    6: ((4, 0), (2, 4)),
    7: ((4, 0), (2, 4), (1, 6)),
    8: ((8, 0),),
}

_WORD_BITS = 32
_MASK32 = (1 << 32) - 1


def _require_planar_bits(bits: int) -> None:
    if bits not in _PLANES:
        raise ValueError(f"planar packing supports bits {PLANAR_FORMAT_BITS}, got bits={bits}")


def _require_int32_words(pack_dtype: torch.dtype) -> None:
    if pack_dtype != torch.int32:
        raise NotImplementedError(
            f"planar packing expects 32-bit packed words, got pack_dtype={pack_dtype}."
        )


def planar_pack_rows(values: torch.Tensor, bits: int, *, pack_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Pack `[n, cols]` logical codes along dim 0 into `[n * bits // 32, cols]` int32 words."""
    _require_planar_bits(bits)
    _require_int32_words(pack_dtype)
    n, cols = values.shape
    if n % _WORD_BITS != 0:
        raise ValueError(f"planar packing expects rows divisible by 32, got shape {tuple(values.shape)}")

    blocks = n // _WORD_BITS
    x = values.to(torch.int64).reshape(blocks, _WORD_BITS, cols)
    out = torch.empty((blocks, bits, cols), dtype=torch.int64, device=values.device)
    row = 0
    for width, offset in _PLANES[bits]:
        pack_factor = _WORD_BITS // width
        plane = (x >> offset) & ((1 << width) - 1)
        reshaped = plane.reshape(blocks, width, pack_factor, cols)
        shifts = (
            torch.arange(pack_factor, dtype=torch.int64, device=values.device).view(1, 1, pack_factor, 1) * width
        )
        out[:, row:row + width] = (reshaped << shifts).sum(dim=2, dtype=torch.int64)
        row += width
    return ((out & _MASK32).reshape(blocks * bits, cols)).to(pack_dtype)


def planar_unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack `[n * bits // 32, cols]` int32 words back to `[n, cols]` int32 logical codes."""
    _require_planar_bits(bits)
    _require_int32_words(packed.dtype)
    rows, cols = packed.shape
    if rows % bits != 0:
        raise ValueError(
            f"planar {bits}-bit qweight expects rows divisible by {bits}, got shape {tuple(packed.shape)}"
        )

    blocks = rows // bits
    words = packed.to(torch.int64).reshape(blocks, bits, cols) & _MASK32
    result = torch.zeros((blocks, _WORD_BITS, cols), dtype=torch.int64, device=packed.device)
    row = 0
    for width, offset in _PLANES[bits]:
        pack_factor = _WORD_BITS // width
        shifts = (
            torch.arange(pack_factor, dtype=torch.int64, device=packed.device).view(1, 1, pack_factor, 1) * width
        )
        codes = (words[:, row:row + width].unsqueeze(2) >> shifts) & ((1 << width) - 1)
        result |= codes.reshape(blocks, _WORD_BITS, cols) << offset
        row += width
    return result.reshape(blocks * _WORD_BITS, cols).to(torch.int32)


def planar_pack_cols(values: torch.Tensor, bits: int, *, pack_dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Pack `[rows, n]` logical codes along dim 1 into `[rows, n * bits // 32]` int32 words."""
    _require_planar_bits(bits)
    _require_int32_words(pack_dtype)
    rows, n = values.shape
    if n % _WORD_BITS != 0:
        raise ValueError(f"planar packing expects columns divisible by 32, got shape {tuple(values.shape)}")
    packed = planar_pack_rows(values.transpose(0, 1).contiguous(), bits, pack_dtype=pack_dtype)
    return packed.transpose(0, 1).contiguous()


def planar_unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack `[rows, n * bits // 32]` int32 words back to `[rows, n]` int32 logical codes."""
    _require_planar_bits(bits)
    _require_int32_words(packed.dtype)
    rows, cols = packed.shape
    if cols % bits != 0:
        raise ValueError(
            f"planar {bits}-bit qzeros expects columns divisible by {bits}, got shape {tuple(packed.shape)}"
        )
    unpacked = planar_unpack_rows(packed.transpose(0, 1).contiguous(), bits)
    return unpacked.transpose(0, 1).contiguous()
