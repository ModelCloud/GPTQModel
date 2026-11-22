# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Lossless row-oriented NF12 embedding for native Blackhole TTNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import pad

from .nf12_codec import rowmajor_pack_cpu
from .nf12_tenstorrent import normalize_ids, round_up, tensor_sha256, verify_nf12
from .nf12_ttnn import _cb, _cores, _kernel
from .ttnn_runtime import import_ttnn


_STRIPE_COLUMNS = 128


@dataclass(frozen=True)
class _RowStorageDescriptor:
    layout: str
    padded_columns: int
    packed_sha256: str
    escapes_sha256: str


def _transcode_rows(descriptor, weight):
    verify_nf12(descriptor, weight)
    n, _ = descriptor.logical_shape
    packed, escapes = rowmajor_pack_cpu(weight.packed, weight.escapes, weight.exp_base)
    # Blackhole NoC reads two adjacent 96-byte CUDA/HIP row blocks as one
    # aligned stripe. Add transport padding only when K has an odd block.
    _, k = descriptor.padded_shape
    padding = (round_up(k, _STRIPE_COLUMNS) - k) * 3 // 2
    rows = pad(packed[:n], (0, padding)) if padding else packed[:n]
    escape_pages = pad(escapes, (0, 0, 0, max(8, round_up(len(escapes), 8)) - len(escapes))).reshape(-1, 32)
    return rows, escape_pages, len(escapes)


class TTNNNF12RowEmbedding:
    """One derived packed copy for native embedding. Checkpoint input is canonical NF12."""

    def __init__(self, device, descriptor, weight):
        ttnn = import_ttnn()
        if device.arch() != ttnn.Arch.BLACKHOLE or device.get_num_devices() != 1:
            raise ValueError("Row-oriented NF12 embedding requires one Blackhole chip")
        self.device = device
        self.descriptor = descriptor
        self.exp_base = weight.exp_base
        packed, escapes, self.escape_groups = _transcode_rows(descriptor, weight)
        self.storage_descriptor = _RowStorageDescriptor(
            "nf12-row-major-v1", packed.shape[1] * 2 // 3, tensor_sha256(packed), tensor_sha256(escapes),
        )
        self.packed, self.escapes = (
            ttnn.from_torch(tensor.view(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in (packed, escapes)
        )

    @property
    def resident_bytes(self):
        return (self.packed.volume() + self.escapes.volume()) * 4

    def upload_ids(self, ids, padding_idx=None):
        ttnn = import_ttnn()
        ids = normalize_ids(ids, self.descriptor.logical_shape[0], padding_idx).flatten()
        ids = pad(ids, (0, round_up(len(ids), 16) - len(ids)), value=-1)
        return ttnn.from_torch(ids.reshape(-1, 16), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def prepare_gather(self, ids, count, *, stripes_per_task=None, ids_per_task=1):
        """Bind a resident normalized ID tensor to one gather dispatch."""
        ttnn = import_ttnn()
        n, _ = self.descriptor.logical_shape
        k = self.storage_descriptor.padded_columns
        if (count <= 0 or tuple(ids.shape) != (round_up(count, 16) // 16, 16)
                or ids.dtype != ttnn.int32 or ids.layout != ttnn.ROW_MAJOR_LAYOUT
                or ids.device() != self.device or ids.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError("Expected nonempty normalized int32 IDs on the embedding's device")
        stripes = k // _STRIPE_COLUMNS
        if stripes_per_task is None:
            # The released 2,304-wide table is the measured geometry. Keep
            # one stripe per task for other row widths and request counts.
            stripes_per_task = {1: 2, 32: 9}.get(count, 1) if stripes == 18 else 1
        if not 1 <= stripes_per_task <= stripes or not 1 <= ids_per_task <= count:
            raise ValueError("NF12 gather task is outside the physical row or ID count")
        output = ttnn.allocate_tensor_on_device(ttnn.Shape((count, k)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                               self.device, ttnn.DRAM_MEMORY_CONFIG)
        operands = [self.packed, self.escapes, ids, output]
        addresses = [tensor.buffer_address() for tensor in operands]
        stripe_tasks = (stripes + stripes_per_task - 1) // stripes_per_task
        row_tasks = (count + ids_per_task - 1) // ids_per_task
        scratch = stripes_per_task * (192 + 256 + 32 * 8 + 32 * 64) + 64
        cores, assignments = _cores(ttnn, self.device, row_tasks * stripe_tasks)
        reader = _kernel(ttnn, "gather_rows", ttnn.ReaderConfigDescriptor(), cores, assignments,
                         (n, stripes, self.exp_base, count, stripes_per_task, self.descriptor.padded_shape[1] // 64,
                          ids_per_task), operands,
                         lambda start, end: [*addresses, start, end])
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[],
                                         cbs=[_cb(ttnn, cores, 24, ttnn.uint32, scratch)])
        return lambda: ttnn.generic_op(operands, program)

    def gather_device(self, ids, count):
        return self.prepare_gather(ids, count)()

    def embedding(self, ids, padding_idx=None):
        if ids.device.type != "cpu":
            raise ValueError("Row-oriented NF12 requires CPU IDs in its Torch adapter")
        n, k = self.descriptor.logical_shape
        count = ids.numel()
        if not count:
            normalize_ids(ids, n, padding_idx)
            return torch.empty((*ids.shape, k), dtype=torch.bfloat16)
        resident_ids = self.upload_ids(ids, padding_idx)
        stripes_per_task = None
        ids_per_task = 1
        if count == 32 and self.storage_descriptor.padded_columns == 2304:
            flat = ids.flatten()
            if torch.equal(flat, flat[0].expand_as(flat)):
                stripes_per_task = 1
                ids_per_task = 32
        result = self.prepare_gather(resident_ids, count, stripes_per_task=stripes_per_task, ids_per_task=ids_per_task)()
        return import_ttnn().to_torch(result)[:, :k].reshape(*ids.shape, k)
