# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Layout reference: MLX-LM (Apple Inc., MIT), mlx_lm/utils.py.
"""Transfer supported GPTQ/AWQ weights into MLX's affine quantized layout."""

import numpy as np


_SHIFTS = np.arange(8, dtype=np.uint32) * 4
_AWQ_SHIFTS = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32) * 4
_PLANES = {3: ((2, 0), (1, 2)), 5: ((4, 0), (1, 4)),
           6: ((4, 0), (2, 4)), 7: ((4, 0), (2, 4), (1, 6))}


def _pack_rows(codes, bits=4):
    out_features, in_features = codes.shape
    if in_features % 32 or bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError("MLX affine packing requires 32-aligned inputs and 2, 3, 4, 5, 6, or 8 bits")
    if bits in (2, 4, 8):
        per_word = 32 // bits
        shifts = np.arange(per_word, dtype=np.uint32) * bits
        return np.bitwise_or.reduce(
            codes.reshape(out_features, in_features // per_word, per_word).astype(np.uint32) << shifts,
            axis=-1,
        )
    # MLX stores consecutive code bits across 32-bit word boundaries, including
    # the boundaries crossed by 3-, 5-, and 6-bit codes.
    code_bits = np.unpackbits(codes.astype(np.uint8)[..., None], axis=-1, bitorder="little")[..., :bits]
    return np.packbits(code_bits.reshape(out_features, -1), axis=-1, bitorder="little").view(np.uint32)


def _unpack_gptq(packed, bits, count, planar=False):
    """Decode GPTQ v2 continuous or split-plane words along the final axis."""
    if bits not in (2, 3, 4, 5, 6, 7, 8) or count % 32 or packed.shape[-1] != count * bits // 32:
        raise ValueError("Unsupported GPTQ packed shape or bit width")
    words = packed.astype(np.uint32)
    if planar:
        if bits not in _PLANES:
            raise ValueError(f"Unsupported planar GPTQ width {bits}")
        blocks = words.reshape(-1, count // 32, bits)
        decoded = np.zeros((blocks.shape[0], count // 32, 32), dtype=np.uint32)
        start = 0
        for width, offset in _PLANES[bits]:
            factor = 32 // width
            plane = blocks[..., start:start + width]
            shifts = np.arange(factor, dtype=np.uint32) * width
            values = ((plane[..., None] >> shifts) & ((1 << width) - 1)).reshape(decoded.shape)
            decoded |= values << offset
            start += width
        return decoded.reshape(-1, count)
    offsets = np.arange(count, dtype=np.uint64) * bits
    indices = (offsets // 32).astype(np.intp)
    shifts = (offsets % 32).astype(np.uint64)
    low = words[:, indices].astype(np.uint64) >> shifts
    high = np.pad(words, ((0, 0), (0, 1)))[:, indices + 1].astype(np.uint64)
    return ((low | (high << (32 - shifts))) & ((1 << bits) - 1)).astype(np.uint32)


def repack_gptq(qweight, qzeros, scales, in_features, out_features, bits, planar=None):
    """Return exact MLX affine codes, scales, and biases from GPTQ v2 words."""
    if planar is None:
        planar = bits in (5, 6, 7)
    if in_features % 32 or out_features % 32:
        raise ValueError("GPTQ to MLX requires 32-aligned feature dimensions")
    if qweight.shape != (in_features * bits // 32, out_features):
        raise ValueError("Unsupported GPTQ qweight shape")
    if qzeros.shape != (scales.shape[0], out_features * bits // 32):
        raise ValueError("Unsupported GPTQ qzeros shape")
    if scales.shape[1] != out_features:
        raise ValueError("Unsupported GPTQ scales shape")
    zeros = _unpack_gptq(qzeros, bits, out_features, planar).T
    mlx_scales = np.ascontiguousarray(scales.T)
    biases = -zeros.astype(np.float32) * mlx_scales.astype(np.float32)
    target_bits = 8 if bits == 7 else bits
    weight = np.empty((out_features, in_features * target_bits // 32), dtype=np.uint32)
    for start in range(0, out_features, 64):
        stop = min(start + 64, out_features)
        codes = _unpack_gptq(qweight[:, start:stop].T, bits, in_features, planar)
        weight[start:stop] = _pack_rows(codes, target_bits)
    return weight, mlx_scales, biases


def repack_gptq_4bit(qweight, qzeros, scales, in_features, out_features):
    """Return (weight, scales, biases) for a GPTQ v2 linear layer."""
    return repack_gptq(qweight, qzeros, scales, in_features, out_features, 4)


def repack_awq_4bit(qweight, qzeros, scales, in_features, out_features, bits=4, planar=False):
    """Return (weight, scales, biases) for an AWQ GEMM linear layer."""
    if bits != 4 or planar:
        raise ValueError("AWQ GEMM to MLX supports 4-bit weights")
    if qweight.shape != (in_features, out_features // 8):
        raise ValueError("Unsupported AWQ qweight shape")
    if qzeros.shape != (scales.shape[0], out_features // 8):
        raise ValueError("Unsupported AWQ qzeros shape")
    if scales.shape[1] != out_features:
        raise ValueError("Unsupported AWQ scales shape")

    codes = ((qweight.astype(np.uint32)[:, :, None] >> _AWQ_SHIFTS) & 15)
    codes = codes.reshape(in_features, out_features).T
    zeros = ((qzeros.astype(np.uint32)[:, :, None] >> _AWQ_SHIFTS) & 15)
    zeros = zeros.reshape(scales.shape).T
    mlx_scales = np.ascontiguousarray(scales.T)
    biases = -zeros.astype(np.float32) * mlx_scales.astype(np.float32)
    return _pack_rows(codes), mlx_scales, biases


def repack_awq_gemv(qweight, qzeros, scales, in_features, out_features, group_size):
    """Decode AWQ GEMV's output-major INT4 words into MLX input-major words."""
    if in_features % 32 or out_features % 8:
        raise ValueError("AWQ GEMV to MLX needs input/output dimensions aligned to 32/8")
    groups = in_features // group_size
    if (qweight.shape != (out_features, in_features // 8)
            or qzeros.shape[0] != out_features or qzeros.shape[1] < (groups + 7) // 8):
        raise ValueError("Unsupported AWQ GEMV packed shape")
    shifts = np.arange(8, dtype=np.uint32) * 4
    codes = ((qweight.astype(np.uint32)[..., None] >> shifts) & 15).reshape(out_features, in_features)
    zeros = ((qzeros.astype(np.uint32)[..., None] >> shifts) & 15).reshape(out_features, -1)[:, :groups]
    if scales.shape[0] != out_features or scales.shape[1] < groups:
        raise ValueError("Unsupported AWQ GEMV scale shape")
    scales = np.ascontiguousarray(scales[:, :groups].astype(np.float16))
    return _pack_rows(codes), scales, -zeros.astype(np.float32) * scales.astype(np.float32)


def repack_awq_gemv_fast(qweight, scaled_zeros, scales, in_features, out_features, group_size):
    """Undo AWQ GEMV_FAST's 4-row interleave and 32-value input permutation."""
    if in_features % 64 or out_features % 8:
        raise ValueError("AWQ GEMV_FAST to MLX needs input/output dimensions aligned to 64/8")
    groups = in_features // group_size
    if qweight.shape != (out_features // 4, in_features):
        raise ValueError("Unsupported AWQ GEMV_FAST packed weight shape")
    packed = qweight.astype(np.uint16).reshape(out_features // 4, in_features // 64, 64)
    nibble = ((packed[..., None] >> (np.arange(4, dtype=np.uint16) * 4)) & 15)
    interleaved = nibble.reshape(out_features // 4, in_features // 64, 4, 64)
    reordered = interleaved.transpose(0, 2, 1, 3).reshape(out_features, in_features)
    step_two = reordered.reshape(out_features, in_features // 32, 4, 2, 4)
    step_one = step_two.transpose(0, 1, 2, 4, 3).reshape(out_features, in_features)
    original = step_one.reshape(out_features, in_features // 32, 4, 4, 2)
    codes = original.transpose(0, 1, 3, 2, 4).reshape(out_features, in_features)
    if scaled_zeros.shape != scales.shape or scales.shape[1] != out_features:
        raise ValueError("Unsupported AWQ GEMV_FAST scale/zero shape")
    if scales.shape[0] < groups:
        raise ValueError("AWQ GEMV_FAST has fewer scales than input groups")
    return _pack_rows(codes), scales[:groups].T.copy(), scaled_zeros[:groups].T.copy()
