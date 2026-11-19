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


def int4pack_to_codes(
    packed: torch.Tensor, block_size: int | None = None
) -> torch.Tensor:
    """Return the 4-bit codes the CPU int4pack op was given.

    Raises ``RuntimeError`` when the local layout cannot be verified, so a
    caller never rebuilds a checkpoint from an assumed permutation.

    ``block_size`` may carry the value the transform already probed. A traced
    forward passes it explicitly because dynamo would otherwise re-enter the
    probe (and its ``torch.Generator``) inside the graph.
    """

    if not isinstance(packed, torch.Tensor) or packed.dim() != 2:
        raise RuntimeError("CPU int4pack qweight must be a 2D tensor")
    if packed.dtype != torch.uint8:
        raise RuntimeError(
            f"CPU int4pack qweight must be uint8, got {packed.dtype}"
        )
    if block_size is None:
        block_size = int4pack_block_size()
    elif block_size not in _CANDIDATE_BLOCK_SIZES:
        raise RuntimeError(f"unsupported CPU int4pack block size: {block_size}")
    if block_size is None:
        raise RuntimeError("cannot verify the local CPU int4pack layout")
    return _unpack_blocks(packed, block_size)


def runtime_int4pack_dense_weight(module) -> torch.Tensor:
    """Decode a live CPU int4pack runtime layout to a dense ``[in, out]`` weight.

    The inference transform replaces the checkpoint ``qweight`` with the ATen
    int4pack words and ``qzeros`` with the affine offsets the fused op consumes.
    A grad-requiring call cannot use that op (it runs under ``no_grad``), and a
    traced graph must not restore the checkpoint layout by mutating the module.
    This decodes the runtime operands with the same affine identity instead:

        ``scale * (signed_code - 2**(bits-1)) + zero_offset``

    The result stays in runtime input order; a ``desc_act`` caller must gather
    its activations with ``ret_idx`` before the matmul, exactly as the fused op
    does. Nothing is cached or written back to the module.
    """

    bits = int(getattr(module, "bits", 0))
    if bits != 4:
        raise RuntimeError(
            f"{type(module).__name__} cannot decode a {bits}-bit CPU int4pack layout"
        )
    block_size = getattr(module, "_cpu_int4pack_block_size", None)
    if block_size is None:
        raise RuntimeError(
            f"{type(module).__name__} has no verified CPU int4pack block size; "
            "run the first inference forward before a compiled grad forward"
        )
    qweight = getattr(module, "qweight", None)
    scales = getattr(module, "scales", None)
    zeros = getattr(module, "qzeros", None)
    if not all(isinstance(tensor, torch.Tensor) for tensor in (qweight, scales, zeros)):
        raise RuntimeError(
            f"{type(module).__name__} has no live CPU int4pack weight operands"
        )

    codes = int4pack_to_codes(qweight, block_size=block_size)  # [out, in]
    out_features, in_features = codes.shape
    if scales.dim() != 2 or zeros.dim() != 2:
        raise RuntimeError(
            f"{type(module).__name__} CPU int4pack metadata must be 2D: "
            f"scales={tuple(scales.shape)}, zeros={tuple(zeros.shape)}"
        )
    groups = scales.shape[0]
    if (
        tuple(scales.shape) != (groups, out_features)
        or tuple(zeros.shape) != tuple(scales.shape)
        or groups == 0
        or in_features % groups != 0
    ):
        raise RuntimeError(
            f"{type(module).__name__} CPU int4pack metadata does not match the "
            f"packed weight: codes={tuple(codes.shape)}, "
            f"scales={tuple(scales.shape)}, zeros={tuple(zeros.shape)}"
        )

    group_size = in_features // groups
    # The op dequantizes with one scale/offset per input group; expand them to
    # the runtime input order without touching the module's buffers.
    scale = scales.repeat_interleave(group_size, dim=0)
    zero = zeros.repeat_interleave(group_size, dim=0)
    center = 1 << (bits - 1)
    return scale * (codes.t().to(scale.dtype) - center) + zero


def rebuild_gptq_qweight(module) -> torch.Tensor:
    """Rebuild the GPTQ checkpoint ``qweight`` from the live int4pack words.

    The result is ``int32`` in the checkpoint's ``[in_features // pack_factor,
    out_features]`` layout; ``finalize_fused_train_state`` verifies it against
    the pre-transform bytes before the duplicate is dropped.

    A ``desc_act`` checkpoint reorders the input rows by ``ret_idx`` before
    packing (``weight.index_select(0, ret_idx)``), so the packed codes are in
    *group* order. ``ret_idx`` is a permutation of ``[0, in_features)`` and the
    module keeps it, so scattering the codes back with ``index_copy_`` is the
    exact inverse (#5013).
    """

    codes = int4pack_to_codes(module.qweight)  # [out_features, in_features]
    identity = getattr(module, "_ret_idx_is_identity", None)
    # A ``meta`` move (offload, ``to_empty``) carries no values: the inverse
    # permutation cannot be decided from ``g_idx`` and would not change the
    # shape-only result, so only the checkpoint layout is rebuilt.
    if callable(identity) and not codes.is_meta and not identity():
        ret_idx = module._build_ret_idx().to(
            device=codes.device, dtype=torch.long
        )
        if ret_idx.numel() != codes.shape[1]:
            raise RuntimeError(
                "CPU int4pack ret_idx does not describe the packed input rows"
            )
        restored = torch.empty_like(codes)
        restored.index_copy_(1, ret_idx, codes)
        codes = restored
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


def rebuild_avx512_awq_qweight(module) -> torch.Tensor:
    """Rebuild an AWQ checkpoint ``qweight`` from the AVX512-BF16 runtime words.

    ``TorchCpuAvx512AwqLinear.transform_cpu`` hands the checkpoint words to the
    extension's ``pack_awq_int4``, which is a pure nibble permutation: it reads
    the AWQ-interleaved checkpoint words and writes ``uint8 [out_features,
    in_features // 2]`` where each byte pairs output column ``group * 32 + row``
    (low nibble) with ``group * 32 + row + 16`` (high nibble) for one input
    feature. Undoing it only needs the reverse read plus the same
    ``pack_awq_columns(..., order=AWQ_ORDER)`` interleave the quantizer used, so
    this path is pure Torch and does not need the extension or AVX512-BF16
    (#5013).

    The result is verified bitwise against the pre-transform clone by
    ``finalize_fused_train_state`` before that clone is dropped, so an
    unexpected layout keeps the snapshot instead of exporting garbage.
    """

    from ...quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns

    packed = module.qweight
    out_features = int(module.out_features)
    in_features = int(module.in_features)
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.uint8
        or packed.dim() != 2
    ):
        raise RuntimeError(
            "AVX512 AWQ qweight must be a 2D uint8 tensor of packed nibbles"
        )
    if out_features % 32 != 0 or in_features % 2 != 0:
        raise RuntimeError(
            "AVX512 AWQ qweight requires out_features % 32 == 0 and an even "
            f"in_features, got out={out_features} in={in_features}"
        )
    groups = out_features // 32
    k_half = in_features // 2
    if tuple(packed.shape) != (out_features, k_half):
        raise RuntimeError(
            "AVX512 AWQ qweight shape "
            f"{tuple(packed.shape)} does not match {(out_features, k_half)}"
        )
    # ``pack_awq_int4`` writes, per (group, input-feature-pair, output-row):
    #   byte = code(low_col) | code(high_col) << 4
    # at ``group * k_half * 32 + (k // 2) * 32 + row * 2 + (k % 2)``, where
    # ``low_col = group * 32 + row`` and ``high_col = low_col + 16``.
    blocks = packed.reshape(groups, k_half, 16, 2)
    low = (blocks & 0x0F).permute(1, 3, 0, 2)  # [k_half, pair, group, row]
    high = torch.bitwise_right_shift(blocks, 4).permute(1, 3, 0, 2)
    codes = torch.stack((low, high), dim=3).reshape(in_features, out_features)
    return pack_awq_columns(codes.to(torch.int32), int(module.bits), order=AWQ_ORDER)


def avx512_checkpoint_byte_index(module) -> torch.Tensor | None:
    """Byte permutation that restores the checkpoint channel order, or ``None``.

    ``ret_idx`` maps runtime input channels to checkpoint channels. When every
    runtime pair ``(2k, 2k + 1)`` stays adjacent - true for the group-sorted
    ``g_idx`` the CPU packer consumes - the reorder can run on the checkpoint
    *bytes*, half the size of the unpacked codes. Any other permutation returns
    ``None`` so the caller uses the code-level path.
    """

    in_features = module.in_features
    ret_idx = module._build_ret_idx()
    if ret_idx.numel() != in_features:
        return None
    starts = ret_idx[0::2]
    if not torch.equal(ret_idx[1::2], starts + 1):
        return None
    if not bool(torch.all(starts % 2 == 0)):
        return None
    return (starts // 2).to(torch.long)


def _checkpoint_bytes_from_codes(
    codes: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """Pack ``[out, in]`` 4-bit codes into checkpoint bytes ``[out, in // 2]``."""

    pairs = codes.reshape(out_features, in_features // 2, 2)
    return torch.bitwise_or(
        pairs[..., 0],
        torch.bitwise_left_shift(pairs[..., 1], 4),
    )


def _checkpoint_words(
    checkpoint_bytes: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    """View checkpoint bytes as ``[in // 8, out]`` little-endian int32 words."""

    words_per_row = in_features // 8
    words = (
        checkpoint_bytes.reshape(out_features, words_per_row, 4)
        .permute(1, 0, 2)
        .contiguous()
    )
    return words.view(torch.int32).reshape(words_per_row, out_features)


def rebuild_avx512_qweight(module) -> torch.Tensor:
    """Rebuild the GPTQ checkpoint ``qweight`` from the AVX512 runtime words.

    ``TorchCpuAvx512Linear.transform_cpu`` sorts the input channels by GPTQ
    group (``ret_idx``) and packs with ``pack_hf_int4``. The checkpoint keeps
    the original channel order, so the inverse restores that order and packs
    the nibbles back into ``[in_features // pack_factor, out_features]``.

    ``unpack_hf_int4`` is the documented inverse of ``pack_hf_int4``, so this
    path is pure Torch and works without the ATen int4pack ops. The result is
    verified against the pre-transform bytes by
    ``finalize_fused_train_state`` before the duplicate is dropped.

    The checkpoint bytes are built in one pass over the packed tensor and four
    of them are viewed as one little-endian ``int32`` word, replacing the old
    ``codes.t().reshape(...)`` plus eight strided shift/OR passes (issue
    #4884). A ``desc_act`` channel order is applied at the byte level when it
    keeps input pairs adjacent and at the code level otherwise; both emit the
    same bytes.
    """

    from .torch_cpu_avx512 import avx512_checkpoint_bytes, unpack_hf_int4

    out_features = module.out_features
    in_features = module.in_features
    checkpoint_bytes = avx512_checkpoint_bytes(
        module.qweight, out_features=out_features, in_features=in_features
    )
    if not module._ret_idx_is_identity():
        # The transform gathers the input channels; the checkpoint stores them
        # in their original order, so scatter them back the same way
        # ``dequantize_weight`` does.
        byte_index = avx512_checkpoint_byte_index(module)
        if byte_index is not None:
            restored_bytes = torch.empty_like(checkpoint_bytes)
            restored_bytes.index_copy_(1, byte_index, checkpoint_bytes)
            checkpoint_bytes = restored_bytes
        else:
            ret_idx = module._build_ret_idx().to(
                device=checkpoint_bytes.device, dtype=torch.long
            )
            codes = unpack_hf_int4(
                module.qweight,
                out_features=out_features,
                in_features=in_features,
            )
            restored = torch.empty_like(codes)
            restored.index_copy_(1, ret_idx, codes)
            checkpoint_bytes = _checkpoint_bytes_from_codes(
                restored, out_features, in_features
            )
    return _checkpoint_words(checkpoint_bytes, out_features, in_features)


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
    "rebuild_avx512_awq_qweight",
    "rebuild_avx512_qweight",
    "rebuild_awq_qweight",
    "rebuild_gptq_qweight",
    "runtime_int4pack_dense_weight",
]
