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
    "TTBF8", "TTBF4", "BFP8", "BFP4", "pack", "unpack", "matmul", "matmul_bf16",
    "pack_transposed", "transpose_packed", "matmul_prepacked",
    "matmul_prepacked_bf16", "convert_s8", "prepack_s8",
    "matmul_prepacked_s8", "matmul_prepacked_s8_bf16",
    "tile_pack", "tile_unpack", "packed_bytes", "blocks_per_row",
    "tile_bytes", "version", "cpu_features", "lib",
]

TTBF8 = 8
TTBF4 = 4
# Deprecated TT-internal spellings, kept as aliases.
BFP8 = TTBF8
BFP4 = TTBF4

_FMT = {8: 8, 4: 4,
        "ttbf8": 8, "ttbf4": 4,
        "bfp8_b": 8, "bfp8": 8, "BFP8_B": 8,
        "bfp4_b": 4, "bfp4": 4, "BFP4_B": 4}


def _fmt(fmt) -> int:
    try:
        return _FMT[fmt]
    except KeyError:
        raise ValueError(f"unknown bfp format {fmt!r}") from None


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


def pack(x, fmt=TTBF8):
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
    db = 16 if f == 8 else 8
    data = np.zeros((rows, nblk * db), dtype=np.uint8)
    _lib.ttbfp_quantize(f, x.ctypes.data_as(_f32p), rows, cols,
                        _ptr(exps), _ptr(data))
    return exps, data


def unpack(exps, data, fmt=TTBF8, cols: int | None = None, ftz: bool = True):
    """Decode packed BFP data to a row-major fp32 array."""
    f = _fmt(fmt)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if exps.ndim != 2:
        raise ValueError("exps must be (rows, nblk)")
    rows, nblk = exps.shape
    if cols is None:
        cols = nblk * 16
    if cols > nblk * 16:
        raise ValueError("cols exceeds the packed block count")
    out = np.empty((rows, cols), dtype=np.float32)
    _lib.ttbfp_dequantize(f, _ptr(exps), _ptr(data), rows, cols,
                          out.ctypes.data_as(_f32p), int(bool(ftz)))
    return out


def matmul(a, exps, data, fmt=TTBF8, cols: int | None = None, ftz: bool = True):
    """C = A @ dequant(B) with B kept packed (fused dequant + GEMM).

    a: (M, K) float32; exps: (K, nblk); data: (K, nblk*db).
    Returns (M, N) float32, N = nblk*16 unless ``cols`` is given.
    """
    f = _fmt(fmt)
    a = _f32(a)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    if exps.shape[0] != K:
        raise ValueError(f"exps has {exps.shape[0]} rows, expected K={K}")
    nblk = exps.shape[1]
    N = nblk * 16 if cols is None else int(cols)
    if N > nblk * 16:
        raise ValueError("cols exceeds the packed block count")
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul(f, a.ctypes.data_as(_f32p), K, _ptr(exps), _ptr(data),
                      M, K, N, out.ctypes.data_as(_f32p), N, int(bool(ftz)))
    return out


def matmul_bf16(a_bf16, exps, data, fmt=TTBF8, cols: int | None = None,
                ftz: bool = True):
    """Like matmul(), with A given as bf16 bit patterns (uint16)."""
    f = _fmt(fmt)
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    M, K = a.shape
    if exps.shape[0] != K:
        raise ValueError(f"exps has {exps.shape[0]} rows, expected K={K}")
    nblk = exps.shape[1]
    N = nblk * 16 if cols is None else int(cols)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_bf16(f, a.ctypes.data_as(_u16p), K, _ptr(exps),
                           _ptr(data), M, K, N, out.ctypes.data_as(_f32p), N,
                           int(bool(ftz)))
    return out


def transpose_packed(exps, data, fmt=TTBF8):
    """Convert the flat packed layout to the block-major (streaming) layout.

    exps (K, nblk) -> (nblk, K);  data (K, nblk*db) -> (nblk, K*db).
    """
    f = _fmt(fmt)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if exps.ndim != 2:
        raise ValueError("exps must be (K, nblk)")
    K, nblk = exps.shape
    db = 16 if f == 8 else 8
    if data.shape != (K, nblk * db):
        raise ValueError(f"data must be (K, {nblk * db})")
    exps_t = np.empty((nblk, K), dtype=np.uint8)
    data_t = np.empty((nblk, K * db), dtype=np.uint8)
    _lib.ttbfp_transpose_packed(f, _ptr(exps), _ptr(data), K, nblk * 16,
                                _ptr(exps_t), _ptr(data_t))
    return exps_t, data_t


def pack_transposed(x, fmt=TTBF8):
    """Quantize and return the block-major packed layout (best for matmul)."""
    e, d = pack(x, fmt)
    return transpose_packed(e, d, fmt)


def matmul_prepacked(a, exps_t, data_t, fmt=TTBF8, cols: int | None = None,
                     ftz: bool = True):
    """C = A @ dequant(B) with B in the block-major layout.

    exps_t: (nblk, K) uint8;  data_t: (nblk, K*db) uint8.
    """
    f = _fmt(fmt)
    a = _f32(a)
    exps_t = np.ascontiguousarray(exps_t, dtype=np.uint8)
    data_t = np.ascontiguousarray(data_t, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    if exps_t.ndim != 2 or exps_t.shape[1] != K:
        raise ValueError(f"exps_t must be (nblk, {K})")
    nblk = exps_t.shape[0]
    N = nblk * 16 if cols is None else int(cols)
    if N > nblk * 16:
        raise ValueError("cols exceeds the packed block count")
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked(f, a.ctypes.data_as(_f32p), K, _ptr(exps_t),
                                _ptr(data_t), M, K, N,
                                out.ctypes.data_as(_f32p), N, int(bool(ftz)))
    return out


def matmul_prepacked_bf16(a_bf16, exps_t, data_t, fmt=TTBF8,
                          cols: int | None = None, ftz: bool = True):
    """Like matmul_prepacked(), with A as bf16 bit patterns (uint16)."""
    f = _fmt(fmt)
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    exps_t = np.ascontiguousarray(exps_t, dtype=np.uint8)
    data_t = np.ascontiguousarray(data_t, dtype=np.uint8)
    M, K = a.shape
    nblk = exps_t.shape[0]
    N = nblk * 16 if cols is None else int(cols)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_bf16(f, a.ctypes.data_as(_u16p), K,
                                     _ptr(exps_t), _ptr(data_t), M, K, N,
                                     out.ctypes.data_as(_f32p), N,
                                     int(bool(ftz)))
    return out


def convert_s8(exps, data, fmt=TTBF8):
    """Convert the flat packed layout to the int8 kernel-native layout.

    Returns (exps_t, s8) with shapes (nblk, K) and (nblk, K*16).  The values
    are unchanged: sign*magnitude is stored as a two's-complement int8, which
    the matmul kernel decodes in 3 instructions instead of 8-12.
    """
    f = _fmt(fmt)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if exps.ndim != 2:
        raise ValueError("exps must be (K, nblk)")
    K, nblk = exps.shape
    exps_t = np.empty((nblk, K), dtype=np.uint8)
    s8 = np.empty((nblk, K * 16), dtype=np.uint8)
    _lib.ttbfp_convert_s8(f, _ptr(exps), _ptr(data), K, nblk * 16,
                          _ptr(exps_t), _ptr(s8))
    return exps_t, s8


def prepack_s8(x, fmt=TTBF8):
    """Quantize straight into the int8 kernel-native block-major layout."""
    e, d = pack(x, fmt)
    return convert_s8(e, d, fmt)


def matmul_prepacked_s8(a, exps_t, s8, fmt=TTBF8, cols: int | None = None,
                        ftz: bool = True):
    """C = A @ dequant(B) with B in the int8 kernel-native layout."""
    f = _fmt(fmt)
    a = _f32(a)
    exps_t = np.ascontiguousarray(exps_t, dtype=np.uint8)
    s8 = np.ascontiguousarray(s8, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError("a must be 2-D")
    M, K = a.shape
    if exps_t.ndim != 2 or exps_t.shape[1] != K:
        raise ValueError(f"exps_t must be (nblk, {K})")
    nblk = exps_t.shape[0]
    N = nblk * 16 if cols is None else int(cols)
    if N > nblk * 16:
        raise ValueError("cols exceeds the packed block count")
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_s8(f, a.ctypes.data_as(_f32p), K,
                                   _ptr(exps_t), _ptr(s8), M, K, N,
                                   out.ctypes.data_as(_f32p), N,
                                   int(bool(ftz)))
    return out


def matmul_prepacked_s8_bf16(a_bf16, exps_t, s8, fmt=TTBF8,
                             cols: int | None = None, ftz: bool = True):
    """Like matmul_prepacked_s8(), with A as bf16 bit patterns (uint16)."""
    f = _fmt(fmt)
    a = np.ascontiguousarray(a_bf16, dtype=np.uint16)
    exps_t = np.ascontiguousarray(exps_t, dtype=np.uint8)
    s8 = np.ascontiguousarray(s8, dtype=np.uint8)
    M, K = a.shape
    nblk = exps_t.shape[0]
    N = nblk * 16 if cols is None else int(cols)
    out = np.empty((M, N), dtype=np.float32)
    _lib.ttbfp_matmul_prepacked_s8_bf16(f, a.ctypes.data_as(_u16p), K,
                                        _ptr(exps_t), _ptr(s8), M, K, N,
                                        out.ctypes.data_as(_f32p), N,
                                        int(bool(ftz)))
    return out


def tile_pack(tile, fmt=TTBF8):
    """Pack a logical 32x32 fp32 tile into the TT L1 byte layout."""
    f = _fmt(fmt)
    t = _f32(tile)
    if t.shape != (32, 32):
        raise ValueError("tile must be 32x32")
    n = int(_lib.ttbfp_tile_bytes(f))
    out = np.zeros(n, dtype=np.uint8)
    _lib.ttbfp_tile_pack(f, t.ctypes.data_as(_f32p), _ptr(out))
    return out


def tile_unpack(blob, fmt=TTBF8, ftz: bool = True):
    """Unpack a TT L1 byte-layout tile back to a 32x32 fp32 tile."""
    f = _fmt(fmt)
    b = np.ascontiguousarray(blob, dtype=np.uint8)
    out = np.empty((32, 32), dtype=np.float32)
    _lib.ttbfp_tile_unpack(f, _ptr(b), out.ctypes.data_as(_f32p), int(bool(ftz)))
    return out
