# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ctypes + NumPy bindings for the ttbfp CPU kernels.

The C library is built by the repository Makefile (``make``); set TTBFP_LIB
to point at a different libttbfp.so if needed.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

__all__ = [
    "BF8_B", "BF4_B", "BF2_B", "TTBF8", "TTBF4", "TTBF2",
    "BFP8", "BFP4", "BFP2", "pack", "unpack", "matmul", "matmul_bf16",
    "pack_transposed", "transpose_packed", "matmul_prepacked",
    "matmul_prepacked_bf16", "convert_s8", "prepack_s8",
    "matmul_prepacked_s8", "matmul_prepacked_s8_bf16",
    "tile_pack", "tile_unpack", "packed_bytes", "blocks_per_row",
    "tile_bytes", "version", "cpu_features", "lib",
]

# Canonical names follow hw/dtypes.md.
BF8_B = 8
BF4_B = 4
BF2_B = 2
# Deprecated spellings, kept as aliases.
TTBF8 = BF8_B
TTBF4 = BF4_B
TTBF2 = BF2_B
BFP8 = BF8_B
BFP4 = BF4_B
BFP2 = BF2_B

_FMT = {8: 8, 4: 4, 2: 2,
        "bf8_b": 8, "bf4_b": 4, "bf2_b": 2,
        "ttbf8": 8, "ttbf4": 4, "ttbf2": 2,
        "bfp8_b": 8, "bfp8": 8, "BFP8_B": 8,
        "bfp4_b": 4, "bfp4": 4, "BFP4_B": 4,
        "bfp2_b": 2, "bfp2": 2, "BFP2_B": 2}


def _fmt(fmt) -> int:
    try:
        return _FMT[fmt]
    except KeyError:
        raise ValueError(f"unknown bfp format {fmt!r}") from None


def _db(f: int) -> int:
    """Datum bytes per 16-element block (16 / 8 / 4 for bf8_b / bf4_b / bf2_b)."""
    return 16 if f == 8 else (8 if f == 4 else 4)


def _flat_packed(exps, data, fmt):
    """Validate the row-major (flat) layout and return its dimensions."""
    f = _fmt(fmt)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if exps.ndim != 2:
        raise ValueError("exps must be a 2-D (rows, nblk) array")
    if data.ndim != 2:
        raise ValueError("data must be a 2-D (rows, nblk*db) array")
    rows, nblk = exps.shape
    want = (rows, nblk * _db(f))
    if data.shape != want:
        raise ValueError(f"data must have shape {want}, got {data.shape}")
    return f, exps, data, rows, nblk


def _block_major(exps_t, data_t, fmt, K, *, s8=False):
    """Validate the block-major layout and return its dimensions."""
    f = _fmt(fmt)
    exps_t = np.ascontiguousarray(exps_t, dtype=np.uint8)
    data_t = np.ascontiguousarray(data_t, dtype=np.uint8)
    if exps_t.ndim != 2:
        raise ValueError("exps_t must be a 2-D (nblk, K) array")
    if exps_t.shape[1] != K:
        raise ValueError(f"exps_t must be (nblk, {K}), got {exps_t.shape}")
    nblk = exps_t.shape[0]
    width = 16 if s8 else _db(f)
    want = (nblk, K * width)
    if data_t.ndim != 2 or data_t.shape != want:
        raise ValueError(f"packed data must have shape {want}, got {data_t.shape}")
    return f, exps_t, data_t, nblk


def _cols(cols, nblk, *, flat):
    """Resolve ``cols`` and reject counts that change a flat row stride."""
    if cols is None:
        return nblk * 16
    cols = int(cols)
    if cols <= 0 or cols > nblk * 16:
        raise ValueError(f"cols must be in 1..{nblk * 16}, got {cols}")
    if flat and blocks_per_row(cols) != nblk:
        raise ValueError(
            f"cols={cols} would change the stored row stride: the packed "
            f"buffer has {nblk} blocks per row, but cols={cols} needs "
            f"{blocks_per_row(cols)}. Pass cols > {(nblk - 1) * 16} or "
            "repack with the smaller column count.")
    return cols


def _find_lib() -> str:
    env = os.environ.get("TTBFP_LIB")
    if env:
        return env
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "libttbfp.so",          # repo root
        here.parent / "libttbfp.so",              # installed next to package
        Path.cwd() / "libttbfp.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise OSError(
        "libttbfp.so not found; run `make` in the ttbfp repository or set "
        "TTBFP_LIB"
    )


_lib = C.CDLL(_find_lib())

_u8p = C.POINTER(C.c_uint8)
_f32p = C.POINTER(C.c_float)
_u16p = C.POINTER(C.c_uint16)
_i64 = C.c_int64

_lib.ttbfp_blocks_per_row.argtypes = [_i64]
_lib.ttbfp_blocks_per_row.restype = C.c_size_t
_lib.ttbfp_exps_bytes.argtypes = [C.c_int, _i64, _i64]
_lib.ttbfp_exps_bytes.restype = C.c_size_t
_lib.ttbfp_data_bytes.argtypes = [C.c_int, _i64, _i64]
_lib.ttbfp_data_bytes.restype = C.c_size_t
_lib.ttbfp_packed_bytes.argtypes = [C.c_int, _i64, _i64]
_lib.ttbfp_packed_bytes.restype = C.c_size_t
_lib.ttbfp_tile_bytes.argtypes = [C.c_int]
_lib.ttbfp_tile_bytes.restype = C.c_size_t

_lib.ttbfp_quantize.argtypes = [C.c_int, _f32p, _i64, _i64, _u8p, _u8p]
_lib.ttbfp_quantize.restype = None
_lib.ttbfp_dequantize.argtypes = [C.c_int, _u8p, _u8p, _i64, _i64, _f32p, C.c_int]
_lib.ttbfp_dequantize.restype = None

_lib.ttbfp_matmul.argtypes = [C.c_int, _f32p, _i64, _u8p, _u8p,
                              _i64, _i64, _i64, _f32p, _i64, C.c_int]
_lib.ttbfp_matmul.restype = None
_lib.ttbfp_matmul_bf16.argtypes = [C.c_int, _u16p, _i64, _u8p, _u8p,
                                   _i64, _i64, _i64, _f32p, _i64, C.c_int]
_lib.ttbfp_matmul_bf16.restype = None

_lib.ttbfp_transpose_packed.argtypes = [C.c_int, _u8p, _u8p, _i64, _i64,
                                        _u8p, _u8p]
_lib.ttbfp_transpose_packed.restype = None
_lib.ttbfp_matmul_prepacked.argtypes = [C.c_int, _f32p, _i64, _u8p, _u8p,
                                        _i64, _i64, _i64, _f32p, _i64, C.c_int]
_lib.ttbfp_matmul_prepacked.restype = None
_lib.ttbfp_matmul_prepacked_bf16.argtypes = [C.c_int, _u16p, _i64, _u8p, _u8p,
                                             _i64, _i64, _i64, _f32p, _i64,
                                             C.c_int]
_lib.ttbfp_matmul_prepacked_bf16.restype = None
_lib.ttbfp_convert_s8.argtypes = [C.c_int, _u8p, _u8p, _i64, _i64, _u8p, _u8p]
_lib.ttbfp_convert_s8.restype = None
_lib.ttbfp_matmul_prepacked_s8.argtypes = [C.c_int, _f32p, _i64, _u8p, _u8p,
                                           _i64, _i64, _i64, _f32p, _i64,
                                           C.c_int]
_lib.ttbfp_matmul_prepacked_s8.restype = None
_lib.ttbfp_matmul_prepacked_s8_bf16.argtypes = [C.c_int, _u16p, _i64, _u8p,
                                                _u8p, _i64, _i64, _i64, _f32p,
                                                _i64, C.c_int]
_lib.ttbfp_matmul_prepacked_s8_bf16.restype = None

_lib.ttbfp_tile_pack.argtypes = [C.c_int, _f32p, _u8p]
_lib.ttbfp_tile_pack.restype = None
_lib.ttbfp_tile_unpack.argtypes = [C.c_int, _u8p, _f32p, C.c_int]
_lib.ttbfp_tile_unpack.restype = None

_lib.ttbfp_version.restype = C.c_char_p
_lib.ttbfp_cpu_features.restype = C.c_char_p

lib = _lib


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_u8p)


def blocks_per_row(cols: int) -> int:
    return int(_lib.ttbfp_blocks_per_row(int(cols)))


def packed_bytes(fmt, rows: int, cols: int) -> int:
    return int(_lib.ttbfp_packed_bytes(_fmt(fmt), int(rows), int(cols)))


def tile_bytes(fmt) -> int:
    return int(_lib.ttbfp_tile_bytes(_fmt(fmt)))


def version() -> str:
    return _lib.ttbfp_version().decode()


def cpu_features() -> str:
    return _lib.ttbfp_cpu_features().decode()


def pack(x, fmt=BF8_B):
    """Quantize a row-major [rows, cols] fp32 array.

    Returns (exps, data): uint8 arrays of shape (rows, nblk) and
    (rows, nblk*16) / (rows, nblk*8).
    """
    f = _fmt(fmt)
    x = _f32(x)
    if x.ndim != 2:
        raise ValueError("pack expects a 2-D array")
    rows, cols = x.shape
    nblk = blocks_per_row(cols)
    exps = np.zeros((rows, nblk), dtype=np.uint8)
    db = _db(f)
    data = np.zeros((rows, nblk * db), dtype=np.uint8)
    _lib.ttbfp_quantize(f, x.ctypes.data_as(_f32p), rows, cols,
                        _ptr(exps), _ptr(data))
    return exps, data


def unpack(exps, data, fmt=BF8_B, cols: int | None = None, ftz: bool = True):
    """Decode packed BFP data to a row-major fp32 array.

    ``cols`` may trim the final partial block; it must not change the
    stored block count, because the flat row stride depends on it.
    """
    f, exps, data, rows, nblk = _flat_packed(exps, data, fmt)
    cols = _cols(cols, nblk, flat=True)
    out = np.empty((rows, cols), dtype=np.float32)
    _lib.ttbfp_dequantize(f, _ptr(exps), _ptr(data), rows, cols,
                          out.ctypes.data_as(_f32p), int(bool(ftz)))
    return out


def matmul(a, exps, data, fmt=BF8_B, cols: int | None = None, ftz: bool = True):
    """C = A @ dequant(B) with B kept packed (fused dequant + GEMM).

    a: (M, K) float32; exps: (K, nblk); data: (K, nblk*db).
    Returns (M, N) float32, N = nblk*16 unless ``cols`` is given.
    ``cols`` has the same final-partial-block restriction as unpack().
    """
    a = _f32(a)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    f, exps, data, rows, nblk = _flat_packed(exps, data, fmt)
    if rows != K:
        raise ValueError(f"exps has {rows} rows, expected K={K}")
    N = _cols(cols, nblk, flat=True)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul(f, a.ctypes.data_as(_f32p), K, _ptr(exps), _ptr(data),
                      M, K, N, out.ctypes.data_as(_f32p), N, int(bool(ftz)))
    return out


def matmul_bf16(a_bf16, exps, data, fmt=BF8_B, cols: int | None = None,
                ftz: bool = True):
    """Like matmul(), with A given as bf16 bit patterns (uint16)."""
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("a_bf16 must be 2-D")
    M, K = a.shape
    f, exps, data, rows, nblk = _flat_packed(exps, data, fmt)
    if rows != K:
        raise ValueError(f"exps has {rows} rows, expected K={K}")
    N = _cols(cols, nblk, flat=True)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_bf16(f, a.ctypes.data_as(_u16p), K, _ptr(exps),
                           _ptr(data), M, K, N, out.ctypes.data_as(_f32p), N,
                           int(bool(ftz)))
    return out


def transpose_packed(exps, data, fmt=BF8_B):
    """Convert the flat packed layout to the block-major (streaming) layout.

    exps (K, nblk) -> (nblk, K);  data (K, nblk*db) -> (nblk, K*db).
    """
    f, exps, data, K, nblk = _flat_packed(exps, data, fmt)
    db = _db(f)
    exps_t = np.empty((nblk, K), dtype=np.uint8)
    data_t = np.empty((nblk, K * db), dtype=np.uint8)
    _lib.ttbfp_transpose_packed(f, _ptr(exps), _ptr(data), K, nblk * 16,
                                _ptr(exps_t), _ptr(data_t))
    return exps_t, data_t


def pack_transposed(x, fmt=BF8_B):
    """Quantize and return the block-major packed layout (best for matmul)."""
    e, d = pack(x, fmt)
    return transpose_packed(e, d, fmt)


def matmul_prepacked(a, exps_t, data_t, fmt=BF8_B, cols: int | None = None,
                     ftz: bool = True):
    """C = A @ dequant(B) with B in the block-major layout.

    exps_t: (nblk, K) uint8;  data_t: (nblk, K*db) uint8.
    """
    a = _f32(a)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    f, exps_t, data_t, nblk = _block_major(exps_t, data_t, fmt, K)
    N = _cols(cols, nblk, flat=False)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked(f, a.ctypes.data_as(_f32p), K, _ptr(exps_t),
                                _ptr(data_t), M, K, N,
                                out.ctypes.data_as(_f32p), N, int(bool(ftz)))
    return out


def matmul_prepacked_bf16(a_bf16, exps_t, data_t, fmt=BF8_B,
                          cols: int | None = None, ftz: bool = True):
    """Like matmul_prepacked(), with A as bf16 bit patterns (uint16)."""
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("a_bf16 must be 2-D")
    M, K = a.shape
    f, exps_t, data_t, nblk = _block_major(exps_t, data_t, fmt, K)
    N = _cols(cols, nblk, flat=False)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_bf16(f, a.ctypes.data_as(_u16p), K,
                                     _ptr(exps_t), _ptr(data_t), M, K, N,
                                     out.ctypes.data_as(_f32p), N,
                                     int(bool(ftz)))
    return out


def convert_s8(exps, data, fmt=BF8_B):
    """Convert the flat packed layout to the int8 kernel-native layout.

    Returns (exps_t, s8) with shapes (nblk, K) and (nblk, K*16).  The values
    are unchanged: sign*magnitude is stored as a two's-complement int8, which
    the matmul kernel decodes in 3 instructions instead of 8-12.
    """
    f, exps, data, K, nblk = _flat_packed(exps, data, fmt)
    exps_t = np.empty((nblk, K), dtype=np.uint8)
    s8 = np.empty((nblk, K * 16), dtype=np.uint8)
    _lib.ttbfp_convert_s8(f, _ptr(exps), _ptr(data), K, nblk * 16,
                          _ptr(exps_t), _ptr(s8))
    return exps_t, s8


def prepack_s8(x, fmt=BF8_B):
    """Quantize straight into the int8 kernel-native block-major layout."""
    e, d = pack(x, fmt)
    return convert_s8(e, d, fmt)


def matmul_prepacked_s8(a, exps_t, s8, fmt=BF8_B, cols: int | None = None,
                        ftz: bool = True):
    """C = A @ dequant(B) with B in the int8 kernel-native layout."""
    a = _f32(a)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    f, exps_t, s8, nblk = _block_major(exps_t, s8, fmt, K, s8=True)
    N = _cols(cols, nblk, flat=False)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_s8(f, a.ctypes.data_as(_f32p), K,
                                   _ptr(exps_t), _ptr(s8), M, K, N,
                                   out.ctypes.data_as(_f32p), N,
                                   int(bool(ftz)))
    return out


def matmul_prepacked_s8_bf16(a_bf16, exps_t, s8, fmt=BF8_B,
                             cols: int | None = None, ftz: bool = True):
    """Like matmul_prepacked_s8(), with A as bf16 bit patterns (uint16)."""
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("a_bf16 must be 2-D")
    M, K = a.shape
    f, exps_t, s8, nblk = _block_major(exps_t, s8, fmt, K, s8=True)
    N = _cols(cols, nblk, flat=False)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_s8_bf16(f, a.ctypes.data_as(_u16p), K,
                                        _ptr(exps_t), _ptr(s8), M, K, N,
                                        out.ctypes.data_as(_f32p), N,
                                        int(bool(ftz)))
    return out


def tile_pack(tile, fmt=BF8_B):
    """Pack a logical 32x32 fp32 tile into the TT L1 byte layout."""
    f = _fmt(fmt)
    t = _f32(tile)
    if t.shape != (32, 32):
        raise ValueError("tile must be 32x32")
    n = int(_lib.ttbfp_tile_bytes(f))
    out = np.zeros(n, dtype=np.uint8)
    _lib.ttbfp_tile_pack(f, t.ctypes.data_as(_f32p), _ptr(out))
    return out


def tile_unpack(blob, fmt=BF8_B, ftz: bool = True):
    """Unpack a TT L1 byte-layout tile back to a 32x32 fp32 tile."""
    f = _fmt(fmt)
    b = np.ascontiguousarray(blob, dtype=np.uint8)
    want = tile_bytes(f)
    if b.size != want:
        raise ValueError(f"blob must contain {want} bytes, got {b.size}")
    out = np.empty((32, 32), dtype=np.float32)
    _lib.ttbfp_tile_unpack(f, _ptr(b), out.ctypes.data_as(_f32p), int(bool(ftz)))
    return out
