# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Invert the CPU ``aten::_convert_weight_to_int4pack_for_cpu`` permutation.

The CPU int4 kernels hand their checkpoint ``qweight`` to
``aten::_convert_weight_to_int4pack_for_cpu`` and keep the packed result as the
runtime operand. That op is a pure nibble permutation: it reads a
``[rows, columns]`` int32 tile of 4-bit codes and writes ``[rows, columns // 2]``
uint8 bytes, so it can be inverted exactly.

The permutation is selected at *runtime* by PyTorch's CPU capability dispatch
(``vec::Vectorized<float>::size() * 4`` rows per block, plus a lane interleave
for the full AVX2/AVX512 block). A hard-coded layout would corrupt checkpoints
on another host, so :func:`int4pack_block_size` probes the local op once and
:func:`int4pack_to_codes` verifies the round trip before it is used.

Even a probe is only a guess about the shapes a real layer uses, so this module
is never trusted on its own: ``finalize_fused_train_state`` rebuilds a layer's
checkpoint words and compares them bitwise with the clone captured before the
transform. Only an exact match drops that clone; otherwise the clone stays and
is what a save emits. If the probe cannot establish the layout, callers keep
the checkpoint snapshot instead of guessing.
"""

from __future__ import annotations

import functools

import torch


#: Candidate rows-per-block values. ``64`` is AVX512, ``32`` AVX2, ``16`` the
#: common NEON/VSX width, and ``4``/``8`` the scalar and 256-bit-float widths.
_CANDIDATE_BLOCK_SIZES = (64, 32, 16, 8, 4)


def _int4pack_op():
    return getattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu", None)


def _unpack_blocks(packed: torch.Tensor, block_size: int) -> torch.Tensor:
    """Invert the packer for a known rows-per-block value.

    ``packed`` is the op output ``[rows, columns // 2]`` uint8. The return value
    is ``[rows, columns]`` uint8 of 4-bit codes.
    """

    rows, half = packed.shape
    columns = half * 2
    codes = torch.empty((rows, columns), dtype=torch.uint8, device=packed.device)
    for start in range(0, rows, block_size):
        size = min(block_size, rows - start)
        block = packed[start : start + size]
        if block_size == 64 and size == 64:
            # Full AVX512 block: lane 0/2 and lane 1/3 are packed together.
            lanes = block.reshape(columns, 32)
            codes[start : start + size] = torch.cat(
                (lanes & 0x0F, torch.bitwise_right_shift(lanes, 4)), dim=1
            ).t()
        elif block_size == 32 and size == 32:
            # Full AVX2 block: rows d and d+16 share one byte.
            lanes = block.reshape(columns, 16)
            codes[start : start + size] = torch.cat(
                (lanes & 0x0F, torch.bitwise_right_shift(lanes, 4)), dim=1
            ).t()
        else:
            # Partial block or non-vectorized build: adjacent rows share a byte.
            lanes = block.reshape(columns, size // 2)
            codes[start : start + size] = torch.stack(
                (lanes & 0x0F, torch.bitwise_right_shift(lanes, 4)), dim=-1
            ).permute(1, 2, 0).reshape(size, columns)
    return codes


def _round_trips(block_size: int, rows: int, columns: int) -> bool:
    op = _int4pack_op()
    if op is None:
        return False
    generator = torch.Generator().manual_seed(4395 + rows + columns)
    codes = torch.randint(
        0, 16, (rows, columns), dtype=torch.int32, generator=generator
    )
    packed = op(codes, 1)
    if packed.dtype != torch.uint8:
        return False
    recovered = _unpack_blocks(packed, block_size).to(torch.int32)
    return torch.equal(recovered, codes)


@functools.lru_cache(maxsize=None)
def int4pack_block_size() -> int | None:
    """Return the local packer's rows-per-block, or ``None`` if unknown.

    Two shapes are checked: a full 64-row block distinguishes the vectorized
    layouts, and a 48-row shape exercises the partial-block path. The first
    candidate that round-trips both is the local layout.
    """

    if _int4pack_op() is None:
        return None
    for candidate in _CANDIDATE_BLOCK_SIZES:
        try:
            if _round_trips(candidate, 64, 16) and _round_trips(candidate, 48, 32):
                return candidate
        except Exception:
            continue
    return None


def int4pack_to_codes(packed: torch.Tensor) -> torch.Tensor:
    """Return the 4-bit codes the CPU int4pack op was given.

    Raises ``RuntimeError`` when the local layout cannot be verified, so a
    caller never rebuilds a checkpoint from an assumed permutation.
    """

    if not isinstance(packed, torch.Tensor) or packed.dim() != 2:
        raise RuntimeError("CPU int4pack qweight must be a 2D tensor")
    if packed.dtype != torch.uint8:
        raise RuntimeError(
            f"CPU int4pack qweight must be uint8, got {packed.dtype}"
        )
    block_size = int4pack_block_size()
    if block_size is None:
        raise RuntimeError("cannot verify the local CPU int4pack layout")
    return _unpack_blocks(packed, block_size)


def rebuild_gptq_qweight(module) -> torch.Tensor:
    """Rebuild the GPTQ checkpoint ``qweight`` from the live int4pack words.

    The result is ``int32`` in the checkpoint's ``[in_features // pack_factor,
    out_features]`` layout; ``finalize_fused_train_state`` verifies it against
    the pre-transform bytes before the duplicate is dropped.
    """

    codes = int4pack_to_codes(module.qweight)  # [out_features, in_features]
    pack_factor = module.pack_factor
    out_features = module.out_features
    grouped = codes.t().reshape(-1, pack_factor, out_features).to(torch.int32)
    shifts = torch.arange(pack_factor, dtype=torch.int32, device=codes.device) * module.bits
    packed = torch.zeros(
        (grouped.shape[0], out_features), dtype=torch.int32, device=codes.device
    )
    for index in range(pack_factor):
        packed.bitwise_or_(grouped[:, index, :] << shifts[index])
    return packed


def rebuild_awq_qweight(module) -> torch.Tensor:
    """Rebuild the AWQ checkpoint ``qweight`` from the live int4pack words.

    The transform unpacks with :func:`unpack_awq` and reorders with
    :func:`reverse_awq_order`, so the inverse re-applies the AWQ column order
    and packs the words back.
    """

    from ...quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns

    codes = int4pack_to_codes(module.qweight)  # [out_features, in_features]
    return pack_awq_columns(codes.t().to(torch.int32), module.bits, order=AWQ_ORDER)


__all__ = [
    "int4pack_block_size",
    "int4pack_to_codes",
    "rebuild_awq_qweight",
    "rebuild_gptq_qweight",
]
