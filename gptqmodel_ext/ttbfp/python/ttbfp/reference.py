# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for Tenstorrent BFP8_b / BFP4_b / BFP2_b.

This is a direct port of tt-metal's tt-llk helpers:

  tt_metal/tt-llk/tests/python_tests/helpers/pack.py
  tt_metal/tt-llk/tests/python_tests/helpers/unpack.py
  tt_metal/tt-llk/tests/python_tests/helpers/bfp_format_utils.py

It is intentionally written for clarity, not speed, and is used to validate
the C/C++ kernels bit-for-bit.
"""

from __future__ import annotations

import numpy as np

BLOCK = 16
BFP_MAX_ALIGNMENT_SHIFT = 8


def _fmt_bits(fmt: int) -> int:
    if fmt == 8:
        return 7
    if fmt == 4:
        return 3
    if fmt == 2:
        return 1
    raise ValueError(f"unsupported bfp format: {fmt}")


def _quantize_blocks(x: np.ndarray, mb: int):
    """Return (exps, q) with shapes (rows, nblk) and (rows, nblk, 16).

    Padding in a partial final block is encoded as datum zero, matching the
    C API (the tt-llk helpers only accept whole blocks).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected a 2-D array")
    rows, cols = x.shape
    nblk = (cols + BLOCK - 1) // BLOCK
    padded = np.zeros((rows, nblk * BLOCK), dtype=np.float32)
    padded[:, :cols] = x

    bits = padded.view(np.uint32)
    bf = (bits >> 16) & 0xFFFF
    # Quiet signaling NaNs (bit 22 of fp32 == bit 6 of bf16).
    is_nan = (((bf >> 7) & 0xFF) == 0xFF) & ((bf & 0x7F) != 0)
    bf = np.where(is_nan, bf | 0x0040, bf)

    sign = ((bf >> 15) & 1).astype(np.uint8)
    exp = ((bf >> 7) & 0xFF).astype(np.int32)
    mant = (((bf & 0x7F) >> 1) | 0x40).astype(np.int32)

    e = exp.reshape(rows, nblk, BLOCK)
    m = mant.reshape(rows, nblk, BLOCK)
    s = sign.reshape(rows, nblk, BLOCK)

    shared = e.max(axis=2)
    delta = shared[:, :, None] - e
    shift = np.minimum(delta, BFP_MAX_ALIGNMENT_SHIFT)
    guard_shift = np.clip(delta - 1, 0, BFP_MAX_ALIGNMENT_SHIFT)
    guard = ((m >> guard_shift) & 1) * (delta > 0)
    q = (((m >> shift) + guard) & 0x7F) >> (7 - mb)
    q = q.astype(np.uint8)

    # Sign-magnitude datum; a zero magnitude drops the sign bit.
    q = np.where(q != 0, ((s << mb) | q).astype(np.uint8), np.uint8(0))

    # Zero the padding datums (C API contract).
    valid = np.arange(nblk * BLOCK).reshape(nblk, BLOCK) < cols
    q = np.where(valid[None, :, :], q, 0).astype(np.uint8)
    return shared.astype(np.uint8), q


def pack(x: np.ndarray, fmt: int = 8):
    """Quantize a row-major [rows, cols] fp32 tensor.

    Returns (exps, data) as uint8 arrays with the flat layout used by the C
    API: exps (rows, nblk), data (rows, nblk*16) for BFP8 or
    (rows, nblk*8) for BFP4.
    """
    mb = _fmt_bits(fmt)
    exps, q = _quantize_blocks(x, mb)
    rows, nblk, _ = q.shape
    if fmt == 8:
        data = q.reshape(rows, nblk * BLOCK)
    elif fmt == 4:
        data = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).astype(np.uint8)
        data = data.reshape(rows, nblk * 8)
    else:
        data = (q[:, :, 0::4]
                | (q[:, :, 1::4] << 2)
                | (q[:, :, 2::4] << 4)
                | (q[:, :, 3::4] << 6)).astype(np.uint8)
        data = data.reshape(rows, nblk * 4)
    return exps, data


def dequantize(exps: np.ndarray, data: np.ndarray, fmt: int = 8,
               cols: int | None = None, ftz: bool = True) -> np.ndarray:
    """Decode the flat layout back to row-major fp32."""
    mb = _fmt_bits(fmt)
    exps = np.asarray(exps, dtype=np.uint8)
    rows, nblk = exps.shape
    if cols is None:
        cols = nblk * BLOCK
    if fmt == 8:
        d = np.asarray(data, dtype=np.uint8).reshape(rows, nblk, BLOCK)
    elif fmt == 4:
        b = np.asarray(data, dtype=np.uint8).reshape(rows, nblk, 8)
        d = np.empty((rows, nblk, BLOCK), dtype=np.uint8)
        d[:, :, 0::2] = b & 0x0F
        d[:, :, 1::2] = b >> 4
    else:
        b = np.asarray(data, dtype=np.uint8).reshape(rows, nblk, 4)
        d = np.empty((rows, nblk, BLOCK), dtype=np.uint8)
        for k in range(4):
            d[:, :, k::4] = (b >> (2 * k)) & 0x03

    mag = (d & ((1 << mb) - 1)).astype(np.float32)
    neg = ((d >> mb) & 1).astype(bool)
    scale = np.ldexp(np.float32(1.0), exps.astype(np.int32) - (126 + mb))
    val = mag * scale[:, :, None].astype(np.float32)
    val = np.where(neg, -val, val)
    val = np.where(exps[:, :, None] == 0, np.float32(0.0), val).astype(np.float32)
    if ftz:
        val = np.where(np.abs(val) < np.float32(2.0 ** -126),
                       np.float32(0.0), val).astype(np.float32)
    return np.ascontiguousarray(val.reshape(rows, nblk * BLOCK)[:, :cols])


def matmul(a: np.ndarray, exps: np.ndarray, data: np.ndarray, fmt: int = 8,
           cols: int | None = None, ftz: bool = True) -> np.ndarray:
    """Reference C = A @ dequant(B), accumulated in float64."""
    b = dequantize(exps, data, fmt, cols=cols, ftz=ftz)
    return np.asarray(a, dtype=np.float64) @ b.astype(np.float64)


def tilize(tile: np.ndarray) -> np.ndarray:
    """32x32 row-major -> TT face order (4 x 16x16 row-major faces)."""
    t = np.asarray(tile, dtype=np.float32)
    if t.shape != (32, 32):
        raise ValueError("expected a 32x32 tile")
    faces = np.empty(1024, dtype=np.float32)
    for f in range(4):
        fr, fc = divmod(f, 2)
        faces[f * 256:(f + 1) * 256] = t[fr * 16:(fr + 1) * 16,
                                          fc * 16:(fc + 1) * 16].reshape(-1)
    return faces


def untilize(faces: np.ndarray) -> np.ndarray:
    """TT face order -> 32x32 row-major."""
    f = np.asarray(faces, dtype=np.float32)
    if f.size != 1024:
        raise ValueError("expected 1024 elements")
    t = np.empty((32, 32), dtype=np.float32)
    for fi in range(4):
        fr, fc = divmod(fi, 2)
        t[fr * 16:(fr + 1) * 16, fc * 16:(fc + 1) * 16] = \
            f[fi * 256:(fi + 1) * 256].reshape(16, 16)
    return t


def tile_pack(tile: np.ndarray, fmt: int = 8) -> np.ndarray:
    """Pack a 32x32 tile into the exact TT L1 byte layout.

    Returns 64 exponent bytes followed by 1024 (BFP8), 512 (BFP4) or 256
    (BFP2) datum bytes, i.e. 1088, 576 or 320 bytes.
    """
    exps, data = pack(tilize(tile).reshape(64, 16), fmt)
    return np.concatenate([exps.reshape(-1), data.reshape(-1)])


def tile_unpack(blob: np.ndarray, fmt: int = 8, ftz: bool = True) -> np.ndarray:
    blob = np.asarray(blob, dtype=np.uint8)
    exps = blob[:64].reshape(64, 1)
    db = 16 if fmt == 8 else (8 if fmt == 4 else 4)
    data = blob[64:].reshape(64, db)
    faces = dequantize(exps, data, fmt, cols=16, ftz=ftz).reshape(-1)
    return untilize(faces)
