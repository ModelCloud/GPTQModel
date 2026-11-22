# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU-side preparation and provenance for Tenstorrent NF12 weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from gptqmodel_nf12.api import ESCAPES_KEY, EXP_BASE_KEY, PACKED_KEY, NF12Weight, validate_escape_indices

from .nf12_codec import MAX_EXP_BASE, encode, escape_counts_all_bases


# Keep two pairs of BF16 K tiles queued. Reader scratch holds two raw atom
# pairs, one aligned escape page per group, and the exponent lookup table.
LINEAR_PACKED_BUFFER_DEPTH = 4
LINEAR_PACKED_SCRATCH_BYTES = 4 * 1536 + 2 * 4 * 32 * 64 + 256 * 4


def tensor_sha256(value: torch.Tensor) -> str:
    """SHA-256 of the contiguous tensor bytes, including raw BF16 payloads."""
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()


@dataclass(frozen=True)
class NF12Descriptor:
    tensor_name: str
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    layout: str
    source_sha256: str
    packed_sha256: str
    escapes_sha256: str
    exp_base_sha256: str
    escape_groups: int

    @property
    def packed_bytes(self) -> int:
        n, k = self.padded_shape
        return n * k * 3 // 2 + 16 * self.escape_groups + 1


def describe_nf12(tensor_name, logical_shape, weight, *, source_sha256):
    return NF12Descriptor(
        tensor_name=tensor_name,
        logical_shape=tuple(logical_shape),
        padded_shape=weight.shape,
        layout="nf12-canonical-v1",
        source_sha256=source_sha256,
        packed_sha256=tensor_sha256(weight.packed),
        escapes_sha256=tensor_sha256(weight.escapes),
        exp_base_sha256=tensor_sha256(torch.tensor([weight.exp_base], dtype=torch.uint8)),
        escape_groups=weight.num_escapes,
    )


def verify_nf12(descriptor: NF12Descriptor, weight: NF12Weight) -> None:
    """Validate on the host before a kernel can dereference an escape index."""
    if weight.device.type != "cpu":
        raise ValueError("Tenstorrent NF12 preparation expects CPU checkpoint tensors")
    n, k = descriptor.logical_shape
    if not (0 < n <= weight.shape[0] and 0 < k <= weight.shape[1]):
        raise ValueError("NF12 logical shape is outside the stored matrix")
    if descriptor != describe_nf12(
        descriptor.tensor_name, descriptor.logical_shape, weight, source_sha256=descriptor.source_sha256
    ):
        raise ValueError(f"NF12 descriptor mismatch for {descriptor.tensor_name}")
    for rows in weight.packed.split(512):
        validate_escape_indices(rows, weight.escapes)


def read_nf12(path, descriptor: NF12Descriptor) -> NF12Weight:
    """Read the canonical safetensors triplet bound to this descriptor."""
    from safetensors import safe_open

    prefix = descriptor.tensor_name + "."
    with safe_open(path, framework="pt", device="cpu") as tensors:
        packed = tensors.get_tensor(prefix + PACKED_KEY)
        escapes = tensors.get_tensor(prefix + ESCAPES_KEY)
        base = tensors.get_tensor(prefix + EXP_BASE_KEY)
    if base.dtype != torch.uint8 or tuple(base.shape) != (1,):
        raise ValueError("NF12 exponent base must be uint8 [1]")
    weight = NF12Weight(packed, escapes, int(base.item()))
    verify_nf12(descriptor, weight)
    return weight


def prepare_nf12(source: torch.Tensor, tensor_name: str, *, exp_base: int | None = None):
    """Encode a BF16 checkpoint matrix using bounded host codec workspaces."""
    if source.device.type != "cpu" or source.dtype != torch.bfloat16 or source.ndim != 2:
        raise TypeError("NF12 expects a CPU BF16 [N, K] matrix")
    n, k = source.shape
    if not n or not k:
        raise ValueError("NF12 expects a nonempty matrix")
    padded_n, padded_k = (n + 15) // 16 * 16, (k + 63) // 64 * 64

    def slabs():
        for start in range(0, padded_n, 512):
            rows = min(512, padded_n - start)
            source_rows = source[start : start + rows]
            yield start, torch.nn.functional.pad(source_rows, (0, padded_k - k, 0, rows - len(source_rows)))

    if exp_base is None:
        counts = torch.zeros(MAX_EXP_BASE + 1, dtype=torch.int64)
        for _, slab in slabs():
            counts += escape_counts_all_bases(slab)
        exp_base = int(counts.argmin())

    packed = torch.empty((padded_n, padded_k * 3 // 2), dtype=torch.uint8)
    escape_parts = []
    count = 0
    for start, slab in slabs():
        payload, escapes, _ = encode(slab, exp_base)
        # The canonical encoder numbers escapes from zero within a slab.
        # Convert the two low planes to [atom, lane, (L0, L1)] views.
        if count and len(escapes):
            planes = payload.view(-1, 3, 512).view(torch.int32)
            meta = planes[:, 0].reshape(-1, 32, 4)
            low = planes[:, 1:].reshape(-1, 2, 32, 2, 2)
            for plane in range(2):
                indices = low[:, plane, :, :, 0]
                indices.add_((meta[:, :, plane * 2 : plane * 2 + 2] == 0) * count)
        packed[start : start + len(slab)].copy_(payload)
        escape_parts.append(escapes)
        count += len(escapes)
    weight = NF12Weight(packed, torch.cat(escape_parts), exp_base)
    descriptor = describe_nf12(tensor_name, (n, k), weight, source_sha256=tensor_sha256(source))
    return descriptor, weight


def normalize_ids(ids: torch.Tensor, n: int, padding_idx: int | None = None):
    """Mask signed index values before narrowing them for device transport."""
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("NF12 embedding expects signed int32 or int64 IDs")
    valid = (ids >= 0) & (ids < n)
    if padding_idx is not None:
        if not -n <= padding_idx < n:
            raise ValueError("padding_idx is outside the NF12 embedding table")
        valid = valid & (ids != padding_idx % n)
    return torch.where(valid, ids, -1).to(torch.int32)


_KERNELS = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "nf12_tenstorrent"


@lru_cache(maxsize=None)
def kernel_source(name):
    source = (_KERNELS / f"{name}.cpp").read_text()
    if name in {"gather", "gather_xla", "linear_reader"}:
        source = (_KERNELS / "decode.h").read_text() + "\n" + source
    for header in ("program.h", "sfpu_decode_math.h", "sfpu_escape_patch.h"):
        include = f'#include "{header}"'
        if include in source:
            source = source.replace(include, (_KERNELS / header).read_text())
    return source


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple
