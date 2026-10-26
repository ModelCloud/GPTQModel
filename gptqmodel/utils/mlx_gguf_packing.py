# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF layout reference: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
"""Transcode affine GGUF blocks to exact MLX packed quantized matmul inputs."""

import numpy as np

from ..nn_modules.qlinear.gguf import _unpack_q4_k_scale_min
from .mlx_packing import _pack_rows


MLX_GGUF_QTYPES = frozenset({
    "Q1_0", "Q1_0_g128", "Q2_0", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "TQ1_0", "TQ2_0",
    "MXFP4", "NVFP4",
})


def repack_gguf_float4(qweight: np.ndarray, qtype: str, in_features: int):
    """Reorder GGUF FP4 nibbles into MLX's native MXFP4 or NVFP4 layout."""
    if qtype not in {"MXFP4", "NVFP4"} or qweight.ndim != 2:
        raise ValueError("GGUF FP4 transfer requires MXFP4 or NVFP4 rows")
    block_size, block_bytes, group_size = (32, 17, 32) if qtype == "MXFP4" else (64, 36, 16)
    if in_features % block_size or qweight.shape[1] != in_features // block_size * block_bytes:
        raise ValueError("GGUF FP4 transfer requires complete, unpadded blocks")
    out_features = qweight.shape[0]
    blocks = np.ascontiguousarray(qweight, dtype=np.uint8).reshape(out_features, -1, block_bytes)
    if qtype == "MXFP4":
        scales = blocks[..., 0].reshape(out_features, -1)
        qs = blocks[..., 1:]
        codes = np.concatenate((qs & 15, qs >> 4), axis=-1)
    else:
        scales = blocks[..., :4].reshape(out_features, -1)
        qs = blocks[..., 4:].reshape(out_features, -1, 4, 8)
        codes = np.concatenate((qs & 15, qs >> 4), axis=-1).reshape(out_features, -1, 64)
    weight = _pack_rows(codes.reshape(out_features, in_features), 4)
    return weight, scales.astype(np.uint8), None, {
        "group_size": group_size, "bits": 4, "mode": qtype.lower(),
    }


def repack_gguf_affine(qweight: np.ndarray, qtype: str, in_features: int):
    """Return exact MLX code words, group scales, biases, and affine settings."""
    if qtype not in MLX_GGUF_QTYPES:
        raise ValueError(f"GGUF {qtype} does not have an exact MLX affine mapping")
    if qweight.ndim != 2 or in_features % 32:
        raise ValueError("GGUF MLX affine transfer needs 32-aligned, two-dimensional weights")
    out_features = qweight.shape[0]
    blocks_by_type = {
        "Q1_0": (128, 18), "Q1_0_g128": (128, 18), "Q2_0": (64, 18),
        "Q4_0": (32, 18), "Q8_0": (32, 34), "Q4_K": (256, 144),
        "Q5_K": (256, 176), "Q6_K": (256, 210), "TQ1_0": (256, 54),
        "TQ2_0": (256, 66),
    }
    source_group, block_bytes = blocks_by_type[qtype]
    if in_features % source_group or qweight.shape[1] != in_features // source_group * block_bytes:
        raise ValueError("GGUF padded or malformed weights require a different MLX runtime")
    blocks = np.ascontiguousarray(qweight, dtype=np.uint8).reshape(-1, block_bytes)
    count = len(blocks)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)

    if qtype in {"Q1_0", "Q1_0_g128"}:
        codes = np.unpackbits(blocks[:, 2:], axis=-1, bitorder="little").astype(np.uint32)
        scales = (2 * d)[:, None]
        biases = (-d)[:, None]
        bits, group_size = 2, 128
    elif qtype == "Q2_0":
        qs = blocks[:, 2:].reshape(count, 16, 1)
        codes = ((qs >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(count, 64).astype(np.uint32)
        scales, biases = d[:, None], (-d)[:, None]
        bits, group_size = 2, 64
    elif qtype == "Q4_0":
        qs = blocks[:, 2:]
        codes = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.uint32)
        scales, biases = d[:, None], (-8 * d)[:, None]
        bits, group_size = 4, 32
    elif qtype == "Q8_0":
        codes = (blocks[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint32)
        scales, biases = d[:, None], (-128 * d)[:, None]
        bits, group_size = 8, 32
    elif qtype in {"Q4_K", "Q5_K"}:
        dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
        sub_scales, sub_mins = _unpack_q4_k_scale_min(blocks[:, 4:16])
        scales = (d[:, None] * sub_scales).reshape(-1, 8)
        biases = -(dmin[:, None] * sub_mins).reshape(-1, 8)
        offset = 16 if qtype == "Q4_K" else 48
        qs = blocks[:, offset:]
        low = ((qs.reshape(count, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15)
        codes = low.reshape(count, 8, 32).astype(np.uint32)
        if qtype == "Q5_K":
            high = ((blocks[:, 16:48].reshape(count, 1, 1, 32)
                     >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1).reshape(count, 8, 32)
            codes |= high.astype(np.uint32) << 4
        codes = codes.reshape(count, 256)
        bits, group_size = (4 if qtype == "Q4_K" else 5), 32
    elif qtype == "Q6_K":
        low = ((blocks[:, :128].reshape(count, 2, 1, 64)
                >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & 15).reshape(count, 8, 32)
        high = ((blocks[:, 128:192].reshape(count, 2, 1, 32)
                 >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(count, 8, 32)
        codes = (low.astype(np.uint32) | (high.astype(np.uint32) << 4)).reshape(count, 256)
        d6 = blocks[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1)
        scales = (d6[:, None] * blocks[:, 192:208].view(np.int8).astype(np.float32)).reshape(count, 16)
        biases = -32 * scales
        bits, group_size = 6, 16
    elif qtype == "TQ1_0":
        qs, qh = blocks[:, :48], blocks[:, 48:52]
        a = (qs[:, :32].reshape(count, 1, 1, 32) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape(1, 1, 5, 1)).reshape(count, -1)
        b = (qs[:, 32:].reshape(count, 1, 1, 16) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape(1, 1, 5, 1)).reshape(count, -1)
        c = (qh.reshape(count, 1, 1, 4) * np.array([1, 3, 9, 27], dtype=np.uint8).reshape(1, 1, 4, 1)).reshape(count, -1)
        codes = ((np.concatenate([a, b, c], axis=-1).astype(np.uint16) * 3) >> 8).astype(np.uint32)
        scales, biases = blocks[:, 52:54].copy().view(np.float16).astype(np.float32), -blocks[:, 52:54].copy().view(np.float16).astype(np.float32)
        bits, group_size = 2, 128
        scales = np.repeat(scales, 2, axis=1)
        biases = np.repeat(biases, 2, axis=1)
    else:  # TQ2_0
        qs = blocks[:, :64].reshape(count, 2, 1, 32)
        codes = ((qs >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3).reshape(count, 256).astype(np.uint32)
        d2 = blocks[:, 64:66].copy().view(np.float16).astype(np.float32)
        scales, biases = np.repeat(d2, 2, axis=1), np.repeat(-d2, 2, axis=1)
        bits, group_size = 2, 128

    if group_size == 128 and source_group > 128 and qtype not in {"TQ1_0", "TQ2_0"}:
        scales = np.repeat(scales, source_group // 128, axis=1)
        biases = np.repeat(biases, source_group // 128, axis=1)
    weight = _pack_rows(codes.reshape(out_features, in_features), bits)
    scale_shape = (out_features, in_features // group_size)
    return (weight, scales.reshape(scale_shape).astype(np.float32),
            biases.reshape(scale_shape).astype(np.float32),
            {"group_size": max(32, group_size), "bits": bits, "mode": "affine"})
