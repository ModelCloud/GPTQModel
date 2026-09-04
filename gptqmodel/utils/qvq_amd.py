# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ROCm P32 inference kernels specialized for AMD Instinct MI355X."""

from __future__ import annotations

from operator import index

import torch
import triton
import triton.language as tl

from ..quantization.qvq_rates import (
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)

_P32_RATES = (2.0, 2.5, 3.0, 3.5)
_GFX950_ARCH = "gfx950"


def qvq_p32_amd_supported(device: torch.device | str) -> bool:
    """Return whether ``device`` is the measured gfx950 ROCm target."""

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        properties = torch.cuda.get_device_properties(target)
    except (AssertionError, RuntimeError):
        return False
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] == _GFX950_ARCH


def _bank_mask(transition_bits: int, bank_alt_id: int) -> int:
    masks = {
        4: (0x0000, 0x5A5A, 0x3C3C, 0xC3C3),
        5: (0x0000, 0x9696, 0x3C3C, 0xC3C3),
        6: (0x0000, 0x6969, 0x5A5A, 0x3C3C),
        7: (0x0000, 0xC3C3, 0x9696, 0x5A5A),
    }
    return masks[transition_bits][bank_alt_id]


def _launch_config(m: int) -> tuple[int, int, int]:
    """Choose an MFMA tile without materializing a shape Cartesian product."""

    if m <= 16:
        return 16, 64, 8
    if m <= 64:
        return 32, 64, 8
    return 128, 64, 8


def _integer_argument(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@triton.jit
def _qvq_p32_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    input_ptr,
    trellis_ptr,
    levels_ptr,
    bank_ids_ptr,
    output_ptr,
    size_m,
    size_k: tl.constexpr,
    size_n,
    transition_bits: tl.constexpr,
    words_per_tile: tl.constexpr,
    alternate_mask: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * block_m + tl.arange(0, block_m)
    columns = pid_n * block_n + tl.arange(0, block_n)
    row_mask = rows < size_m
    column_mask = columns < size_n
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    n_tiles = size_n // 16

    for k_tile in range(size_k // 16):
        local_k = tl.arange(0, 16)[:, None]
        local_n = columns[None, :] & 15
        local = local_k * 16 + local_n
        pair = local >> 1
        tile = k_tile * n_tiles + columns[None, :] // 16
        bit_position = (127 - pair) * transition_bits
        first_word = bit_position >> 5
        shift = bit_position & 31
        next_word = tl.where(first_word + 1 == words_per_tile, 0, first_word + 1)
        word_base = tile * words_per_tile
        low = tl.load(trellis_ptr + word_base + first_word, mask=column_mask[None, :], other=0).to(tl.uint32)
        high = tl.load(trellis_ptr + word_base + next_word, mask=column_mask[None, :], other=0).to(tl.uint32)
        state = tl.where(shift == 0, low, (low >> shift) | (high << ((32 - shift) & 31))) & 0xFFFF

        packed_bank = tl.load(bank_ids_ptr + tile, mask=column_mask[None, :], other=0)
        selected = (packed_bank >> (pair >> 4)) & 1
        mixed = state ^ (selected * alternate_mask)
        mixed = mixed ^ (mixed >> 8)
        mixed = (mixed * 40503 + 17011) & 0xFFFF
        mixed = mixed ^ (mixed >> 7)
        level_index = tl.where((local & 1) == 0, mixed >> 8, mixed & 0xFF)
        weight = tl.load(levels_ptr + level_index).to(tl.float16)
        input_offsets = rows[:, None] * size_k + k_tile * 16 + tl.arange(0, 16)[None, :]
        activation = tl.load(input_ptr + input_offsets, mask=row_mask[:, None], other=0.0)
        accumulator = tl.dot(activation, weight, accumulator)

    output_offsets = rows[:, None] * size_n + columns[None, :]
    tl.store(output_ptr + output_offsets, accumulator, mask=row_mask[:, None] & column_mask[None, :])


def qvq_p32_amd(
    x: torch.Tensor,
    window: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int,
    output_fp32: bool = True,
) -> torch.Tensor:
    """Multiply FP16 activations by continuous-window V2B2-P32 tiles on gfx950."""

    bits = normalize_qvq_rate(bits)
    if bits not in _P32_RATES:
        raise ValueError("AMD P32 supports rates W2, W2.5, W3, and W3.5")
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    out_features = _integer_argument("out_features", out_features)
    bank_alt_id = _integer_argument("bank_alt_id", bank_alt_id)
    if not isinstance(output_fp32, bool):
        raise TypeError("output_fp32 must be boolean")
    if not qvq_p32_amd_supported(x.device):
        raise RuntimeError("AMD P32 requires a ROCm gfx950 device")
    if x.ndim != 2 or window.ndim != 2:
        raise ValueError("AMD P32 expects 2D input and window tensors")
    if x.dtype != torch.float16:
        raise TypeError("AMD P32 currently requires float16 input")
    if window.dtype != torch.int32:
        raise TypeError("AMD P32 requires int32 continuous-window words")
    if levels.dtype != torch.float16 or tuple(levels.shape) != (256,):
        raise TypeError("AMD P32 requires the canonical 256-entry float16 PGC16 table")
    if bank_ids.dtype != torch.uint8 or bank_ids.ndim != 1:
        raise TypeError("AMD P32 requires packed uint8 binary bank selectors")
    if any(tensor.device != x.device for tensor in (window, levels, bank_ids)):
        raise ValueError("AMD P32 tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in (x, window, levels, bank_ids)):
        raise ValueError("AMD P32 tensors must be contiguous")
    if not 1 <= bank_alt_id <= 3:
        raise ValueError("AMD P32 alternative bank ID must be in [1, 3]")

    m, k = x.shape
    n = out_features
    if not m or k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"AMD P32 requires positive M and positive K/N divisible by 16, got M={m}, K={k}, N={n}")
    tile_count = (k // 16) * (n // 16)
    expected_window = (tile_count, qvq_words_per_tile(bits, vector_size=2))
    if tuple(window.shape) != expected_window:
        raise ValueError(f"AMD P32 window must have shape {expected_window}")
    if tuple(bank_ids.shape) != (tile_count,):
        raise ValueError(f"AMD P32 selectors must have shape {(tile_count,)}")

    block_m, block_n, num_warps = _launch_config(m)
    output_dtype = torch.float32 if output_fp32 else x.dtype
    output = torch.empty((m, n), device=x.device, dtype=output_dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _qvq_p32_gfx950_kernel[grid](
        x,
        window,
        levels,
        bank_ids,
        output,
        size_m=m,
        size_k=k,
        size_n=n,
        transition_bits=transition_bits,
        words_per_tile=expected_window[1],
        alternate_mask=_bank_mask(transition_bits, bank_alt_id),
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
    )
    return output


__all__ = ["qvq_p32_amd", "qvq_p32_amd_supported"]
