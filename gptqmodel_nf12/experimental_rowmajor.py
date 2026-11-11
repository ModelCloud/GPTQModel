# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Experimental row-major NF12 embedding layout; production dispatch is unchanged.

``from_nf12`` converts a canonical packed NF12 weight on its current device.
The CPU path uses NumPy and the SM80 CUDA path uses bounded native passes.
Only packed bytes, row escape bits, and scalar metadata live in the result.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import torch

from .api import NF12Weight, check_weight_shape


_LAYOUT_VERSION = 1
_KEY_PACKED = "nf12_row_packed"
_KEY_ESCAPES = "nf12_row_escapes"
_KEY_BASE = "nf12_exp_base"
_KEY_VERSION = "nf12_row_layout_version"
_KEY_ORIGINAL = "nf12_row_original_escapes"
_KEY_COLLISIONS = "nf12_row_half_collisions"


@dataclass(eq=False)
class NF12RowMajorWeight:
    """Packed embedding-only layout, with no decoded or expanded weight cache."""

    packed: torch.Tensor  # uint8 [N, 3K/2], 96 bytes per row and 64 columns
    escapes: torch.Tensor  # int16 [E, 4], exact BF16 escape bits
    exp_base: int
    original_escapes: int
    selected_half_collisions: int
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        p, e = self.packed, self.escapes
        if p.dtype != torch.uint8 or p.ndim != 2 or p.shape[1] % 3 or not p.is_contiguous():
            raise ValueError("row-major packed must be contiguous uint8 [N, 3K/2]")
        if e.dtype != torch.int16 or e.ndim != 2 or e.shape[1] != 4 or not e.is_contiguous():
            raise ValueError("row-major escapes must be contiguous int16 [E, 4]")
        if p.device != e.device:
            raise ValueError("row-major packed and escapes must share a device")
        self.shape = (p.shape[0], p.shape[1] * 2 // 3)
        check_weight_shape(*self.shape)
        self.exp_base = int(self.exp_base)
        self.original_escapes = int(self.original_escapes)
        self.selected_half_collisions = int(self.selected_half_collisions)
        if not 0 <= self.exp_base <= 120:
            raise ValueError("row-major exp_base must be in [0, 120]")
        if self.original_escapes < 0 or self.selected_half_collisions < 0:
            raise ValueError("row-major escape counts must be nonnegative")
        if e.shape[0] != 2 * self.original_escapes + self.selected_half_collisions:
            raise ValueError("row-major escape table length disagrees with its counts")
        if e.shape[0] > 1 << 32:
            raise ValueError("row-major escape index exceeds uint32 capacity")

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.escapes.nbytes

    def to(self, device: torch.device | str | int) -> NF12RowMajorWeight:
        return NF12RowMajorWeight(self.packed.to(device), self.escapes.to(device), self.exp_base,
                                  self.original_escapes, self.selected_half_collisions)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        """Explicitly tagged layout; canonical NF12 loaders must not consume it."""
        return {
            prefix + _KEY_PACKED: self.packed,
            prefix + _KEY_ESCAPES: self.escapes,
            prefix + _KEY_BASE: torch.tensor([self.exp_base], dtype=torch.uint8),
            prefix + _KEY_VERSION: torch.tensor([_LAYOUT_VERSION], dtype=torch.uint8),
            prefix + _KEY_ORIGINAL: torch.tensor([self.original_escapes], dtype=torch.int64),
            prefix + _KEY_COLLISIONS: torch.tensor([self.selected_half_collisions], dtype=torch.int64),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor], prefix: str = "") -> NF12RowMajorWeight:
        def scalar(key: str, dtype: torch.dtype) -> int:
            value = state[prefix + key]
            if value.dtype != dtype or value.numel() != 1:
                raise ValueError(f"row-major {prefix + key} must be a one-element {dtype} tensor")
            return int(value.item())

        if scalar(_KEY_VERSION, torch.uint8) != _LAYOUT_VERSION:
            raise ValueError("unsupported row-major layout version")
        return cls(state[prefix + _KEY_PACKED], state[prefix + _KEY_ESCAPES],
                   scalar(_KEY_BASE, torch.uint8), scalar(_KEY_ORIGINAL, torch.int64),
                   scalar(_KEY_COLLISIONS, torch.int64))


def _check_source(source: NF12Weight, device_type: str) -> None:
    if not isinstance(source, NF12Weight) or source.device.type != device_type:
        raise ValueError(f"row-major transform requires a canonical NF12Weight on {device_type}")
    if not source.packed.is_contiguous() or not source.escapes.is_contiguous():
        raise ValueError("row-major transform requires contiguous canonical packed and escapes")
    if source.num_escapes > (1 << 31) - 1:
        raise ValueError("canonical escape index exceeds uint32 transform range")


def _from_nf12_cpu(source: NF12Weight, chunk_blocks: int) -> NF12RowMajorWeight:
    _check_source(source, "cpu")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    n, k = source.shape
    blocks, kb = n // 16, k // 64
    words = source.packed.detach().numpy().view(np.uint32).reshape(blocks, kb, 3, 32, 4)
    old_esc = source.escapes.detach().numpy().view(np.uint16).reshape(-1, 8)
    orig_e = len(old_esc)
    collision_count = 0
    for start in range(0, blocks, chunk_blocks):
        meta = words[start:start + chunk_blocks, :, 0, :, :]
        nonzero = meta != 0
        collision_count += int(np.count_nonzero(nonzero & ((meta & np.uint32(0x00ff00ff)) == 0)))
        collision_count += int(np.count_nonzero(nonzero & ((meta & np.uint32(0xff00ff00)) == 0)))
    total_e = 2 * orig_e + collision_count
    if total_e > 1 << 32:
        raise OverflowError("row-major escape index exceeds uint32 capacity")

    row = np.empty((blocks, 16, kb, 96), dtype=np.uint8)
    row_esc = np.empty((total_e, 4), dtype=np.uint16)
    row_esc[0:2 * orig_e:2] = old_esc[:, (0, 1, 4, 5)]
    row_esc[1:2 * orig_e:2] = old_esc[:, (2, 3, 6, 7)]
    next_collision = 2 * orig_e
    for start in range(0, blocks, chunk_blocks):
        end = min(start + chunk_blocks, blocks)
        for ri in range(16):
            upper = ri // 8
            for atom in range(4):
                plane = 1 if atom < 2 else 2
                half_atom = atom & 1
                for t in range(4):
                    lane = 4 * (ri % 8) + t
                    group = 4 * atom + t
                    meta = words[start:end, :, 0, lane, atom]
                    lo0 = words[start:end, :, plane, lane, 2 * half_atom]
                    lo1 = words[start:end, :, plane, lane, 2 * half_atom + 1]
                    selected = ((meta >> (8 * upper)) & 255) | (((meta >> (16 + 8 * upper)) & 255) << 8)
                    original_escape = meta == 0
                    collision = (selected == 0) & ~original_escape
                    normal_lows = ((lo0 >> (16 * upper)) & 65535) | (((lo1 >> (16 * upper)) & 65535) << 16)
                    lows = normal_lows.copy()
                    if np.any(original_escape):
                        if np.any(lo0[original_escape] >= orig_e):
                            raise ValueError("canonical escape index out of range")
                        lows[original_escape] = lo0[original_escape] * 2 + upper
                    if np.any(collision):
                        count = int(np.count_nonzero(collision))
                        indices = np.arange(next_collision, next_collision + count, dtype=np.uint32)
                        lows[collision] = indices
                        collision_lows = normal_lows[collision]
                        for j in range(4):
                            row_esc[next_collision:next_collision + count, j] = (
                                (source.exp_base << 8) | ((collision_lows >> (8 * j)) & 255)
                            ).astype(np.uint16)
                        next_collision += count
                    row[start:end, ri, :, 2 * group] = selected & 255
                    row[start:end, ri, :, 2 * group + 1] = selected >> 8
                    row[start:end, ri, :, 32 + 4 * group:36 + 4 * group] = (
                        np.ascontiguousarray(lows).view(np.uint8).reshape(end - start, kb, 4)
                    )
    assert next_collision == total_e
    return NF12RowMajorWeight(torch.from_numpy(row.reshape(n, k * 3 // 2)),
                               torch.from_numpy(row_esc.view(np.int16)), source.exp_base,
                               orig_e, collision_count)


def _check_sm80(device: torch.device) -> None:
    if device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("experimental row-major native path requires NVIDIA CUDA")
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("experimental row-major native path is validated only on SM80")


@lru_cache(maxsize=1)
def _native() -> ctypes.CDLL:
    source = Path(__file__).resolve().parents[1] / "gptqmodel_ext/nf12/experiments/rowmajor_embedding.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library_path = Path(tempfile.gettempdir()) / f"nf12-rowmajor-native-{digest}.so"
    if not library_path.is_file():
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        temporary = library_path.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run([nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
                        "-gencode", "arch=compute_80,code=sm_80", str(source),
                        "-o", str(temporary)], check=True)
        os.replace(temporary, library_path)
    lib = ctypes.CDLL(str(library_path))
    lib.nf12_rowmajor_count.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_void_p, ctypes.c_void_p]
    lib.nf12_rowmajor_count.restype = ctypes.c_int
    lib.nf12_rowmajor_pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
    lib.nf12_rowmajor_pack.restype = ctypes.c_int
    lib.nf12_rowmajor_lookup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_uint32, ctypes.c_int64, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_void_p]
    lib.nf12_rowmajor_lookup.restype = ctypes.c_int
    return lib


def _cuda_error(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f"row-major {operation} CUDA error {code}")


def _from_nf12_gpu(source: NF12Weight) -> NF12RowMajorWeight:
    _check_source(source, "cuda")
    _check_sm80(source.device)
    n, k = source.shape
    tiles = (n // 16) * (k // 64)
    lib = _native()
    with torch.cuda.device(source.device):
        stream = torch.cuda.current_stream(source.device).cuda_stream
        counts = torch.empty((tiles,), dtype=torch.int32, device=source.device)
        _cuda_error(lib.nf12_rowmajor_count(source.packed.data_ptr(), n, k,
                                            counts.data_ptr(), stream), "collision count")
        prefix = torch.cumsum(counts, dim=0, dtype=torch.int64)
        collisions = int(prefix[-1].item())
        original_count = source.num_escapes
        total_escapes = 2 * original_count + collisions
        if total_escapes > 1 << 32:
            raise OverflowError("row-major escape index exceeds uint32 capacity")
        packed = torch.empty((n, k * 3 // 2), dtype=torch.uint8, device=source.device)
        escapes = torch.empty((total_escapes, 4), dtype=torch.int16, device=source.device)
        invalid = torch.zeros((), dtype=torch.int32, device=source.device)
        _cuda_error(lib.nf12_rowmajor_pack(source.packed.data_ptr(), source.escapes.data_ptr(),
                                           original_count, n, k, source.exp_base,
                                           prefix.data_ptr(), packed.data_ptr(),
                                           escapes.data_ptr(), invalid.data_ptr(), stream), "pack")
        if int(invalid.item()):
            raise ValueError("canonical escape index out of range")
    return NF12RowMajorWeight(packed, escapes, source.exp_base, original_count, collisions)


def from_nf12(source: NF12Weight, *, backend: Literal["cpu", "cuda"] | None = None,
              chunk_blocks: int = 128) -> NF12RowMajorWeight:
    """Convert canonical packed NF12 on CPU or NVIDIA SM80 to row-major packed NF12.

    This is a copy, not a mutation-aware cache. Release the canonical weight
    after conversion if only embedding lookup is needed. Tied linear weights
    still require canonical NF12 or a separately qualified linear layout.
    """
    if not isinstance(source, NF12Weight):
        raise TypeError("row-major transform requires a canonical NF12Weight")
    if backend is None:
        backend = source.device.type
    if backend == "cpu":
        return _from_nf12_cpu(source, chunk_blocks)
    if backend == "cuda":
        return _from_nf12_gpu(source)
    raise ValueError("backend must be 'cpu' or 'cuda'")


def embedding(ids: torch.Tensor, weight: NF12RowMajorWeight, *, threads: int = 256,
              ids_per_cta: int = 8, cache_global_stores: bool = True) -> torch.Tensor:
    """Look up exact BF16 bits, including +0 rows for invalid int32/int64 IDs."""
    if not isinstance(weight, NF12RowMajorWeight):
        raise TypeError("row-major embedding requires NF12RowMajorWeight")
    _check_sm80(weight.device)
    if (ids.device != weight.device or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous()):
        raise ValueError("ids must be contiguous int32/int64 on the row-major CUDA device")
    if threads not in (64, 128, 256, 512) or ids_per_cta not in (1, 2, 4, 8, 16):
        raise ValueError("unsupported row-major CTA geometry")
    if ids_per_cta > threads // 4:
        raise ValueError("IDs per CTA exceed available four-lane groups")
    n, k = weight.shape
    output = torch.empty((*ids.shape, k), dtype=torch.bfloat16, device=ids.device)
    if ids.numel():
        mode = 2 if cache_global_stores else 1
        with torch.cuda.device(ids.device):
            stream = torch.cuda.current_stream(ids.device).cuda_stream
            _cuda_error(_native().nf12_rowmajor_lookup(
                ids.data_ptr(), weight.packed.data_ptr(), weight.escapes.data_ptr(),
                weight.escapes.shape[0], n, k, weight.exp_base, ids.numel(),
                output.data_ptr(), ids.element_size(), threads, ids_per_cta, mode, stream), "lookup")
    return output
