# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native Blackhole NF12 gather and fused, tile-streamed linear."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from .nf12_tenstorrent import normalize_ids, verify_nf12
from .ttnn_runtime import import_ttnn

_KERNELS = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "nf12_tenstorrent"
_GATHER_SCRATCH = 1536 + 64 + 128 + 16 * 64
_LINEAR_SCRATCH = 1536 + 64 * 64


@lru_cache(maxsize=None)
def kernel_source(name):
    source = (_KERNELS / f"{name}.cpp").read_text()
    if name in {"gather", "linear_reader"}:
        return (_KERNELS / "decode.h").read_text() + "\n" + source
    return source


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _cores(ttnn, device, tasks):
    grid = device.compute_with_storage_grid_size()
    count = min(tasks, grid.x * grid.y)
    assignments = []
    for index in range(count):
        x, y = index % grid.x, index // grid.x
        start, end = tasks * index // count, tasks * (index + 1) // count
        assignments.append((x, y, start, end))
    ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
        for x, y, _, _ in assignments
    ]
    return ttnn.CoreRangeSet(ranges), assignments


def _kernel(ttnn, name, config, cores, assignments, args, tensors, runtime):
    compile_args = list(args)
    for tensor in tensors:
        compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime_args = ttnn.RuntimeArgs()
    for x, y, start, end in assignments:
        runtime_args[x][y] = runtime(start, end)
    return ttnn.KernelDescriptor(
        kernel_source=kernel_source(name),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=compile_args,
        runtime_args=runtime_args,
        config=config,
    )


def _cb(ttnn, cores, index, dtype, page, count=1):
    return ttnn.CBDescriptor(
        total_size=page * count,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


class TTNNNF12:
    """A single-chip DRAM weight. The caller owns the open TTNN device."""

    def __init__(self, device, descriptor, weight):
        ttnn = import_ttnn()
        if device.arch() != ttnn.Arch.BLACKHOLE or device.get_num_devices() != 1:
            raise ValueError("Native NF12 requires a single Blackhole chip")
        verify_nf12(descriptor, weight)
        self.device = device
        self.descriptor = descriptor
        self.exp_base = weight.exp_base
        words = weight.packed.reshape(-1, 1536).view(torch.uint32)
        escapes = weight.escapes
        if len(escapes) % 4 or not len(escapes):
            escapes = torch.nn.functional.pad(escapes, (0, 0, 0, max(4, _round_up(len(escapes), 4)) - len(escapes)))
        self.packed = self._upload(words, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.escapes = self._upload(escapes.reshape(-1, 32).view(torch.uint32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    @property
    def resident_bytes(self):
        return self.packed.volume() * 4 + self.escapes.volume() * 4

    def _upload(self, value, dtype, layout):
        ttnn = import_ttnn()
        return ttnn.from_torch(value, dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _allocate(self, shape, dtype, layout):
        ttnn = import_ttnn()
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, self.device, ttnn.DRAM_MEMORY_CONFIG)

    def upload_ids(self, ids, padding_idx=None):
        ttnn = import_ttnn()
        ids = normalize_ids(ids, self.descriptor.logical_shape[0], padding_idx).flatten()
        ids = torch.nn.functional.pad(ids, (0, _round_up(len(ids), 16) - len(ids)), value=-1)
        return self._upload(ids.reshape(-1, 16), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def gather_device(self, ids, count):
        """Take DRAM pages of 16 normalized int32 IDs and return BF16 rows."""
        ttnn = import_ttnn()
        n, _ = self.descriptor.logical_shape
        _, k = self.descriptor.padded_shape
        if count <= 0 or tuple(ids.shape) != ((_round_up(count, 16) // 16), 16):
            raise ValueError("Unexpected NF12 ID page shape")
        output = self._allocate((count, k), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        tensors = [self.packed, self.escapes, ids, output]
        addresses = [tensor.buffer_address() for tensor in tensors]
        cores, assignments = _cores(ttnn, self.device, count * (k // 64))
        reader = _kernel(
            ttnn, "gather", ttnn.ReaderConfigDescriptor(), cores, assignments,
            (n, k // 64, self.exp_base), tensors, lambda start, end: [*addresses, start, end],
        )
        program = ttnn.ProgramDescriptor(
            kernels=[reader], semaphores=[], cbs=[_cb(ttnn, cores, 24, ttnn.uint32, _GATHER_SCRATCH)],
        )
        return ttnn.generic_op(tensors, program)

    def embedding(self, ids, padding_idx=None):
        if ids.device.type != "cpu":
            raise ValueError("Native TTNN's Torch adapter accepts CPU tensors")
        k = self.descriptor.logical_shape[1]
        if not ids.numel():
            normalize_ids(ids, self.descriptor.logical_shape[0], padding_idx)
            return torch.empty((*ids.shape, k), dtype=torch.bfloat16)
        out = self.gather_device(self.upload_ids(ids, padding_idx), ids.numel())
        return import_ttnn().to_torch(out)[:, :k].reshape(*ids.shape, k)

    def upload_activation(self, x):
        ttnn = import_ttnn()
        logical_k = self.descriptor.logical_shape[1]
        _, k = self.descriptor.padded_shape
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != logical_k or x.device.type != "cpu":
            raise ValueError("Native NF12 expects CPU BF16 activations [M, logical K]")
        padded = torch.nn.functional.pad(x, (0, k - logical_k, 0, _round_up(len(x), 32) - len(x)))
        return self._upload(padded, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def upload_bias(self, bias):
        ttnn = import_ttnn()
        n, _ = self.descriptor.logical_shape
        padded_n, _ = self.descriptor.padded_shape
        if bias.ndim != 1 or len(bias) != n or bias.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Native NF12 bias must be BF16 or FP32 [N]")
        padded = torch.nn.functional.pad(bias.float(), (0, _round_up(padded_n, 32) - n))
        return self._upload(padded[None, :], ttnn.float32, ttnn.TILE_LAYOUT)

    def upload_baseline(self, weight):
        ttnn = import_ttnn()
        n, k = self.descriptor.logical_shape
        padded_n, padded_k = self.descriptor.padded_shape
        if weight.shape != (n, k) or weight.dtype != torch.bfloat16:
            raise ValueError("Matched baseline expects the original BF16 weight")
        padded = torch.nn.functional.pad(weight, (0, padded_k - k, 0, _round_up(padded_n, 32) - n))
        # Store [N tile, K tile] page order, with each page in [K, N] face order.
        tiled = padded.reshape(-1, 32, padded_k // 32, 32).permute(0, 2, 3, 1).contiguous()
        return self._upload(tiled.reshape(-1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def linear_device(self, x, bias=None, *, baseline=None, m_block=4):
        """Run one fixed K-reduction schedule on NF12 or a matched BF16 twin."""
        ttnn = import_ttnn()
        n, k = self.descriptor.padded_shape
        if not 1 <= m_block <= 4 or x.shape[1] != k or x.shape[0] % 32:
            raise ValueError("Unexpected NF12 activation tile shape or M block")
        m_tiles, k_tiles, n_tiles = x.shape[0] // 32, k // 32, _round_up(n, 32) // 32
        m_block = min(m_block, m_tiles)
        dense = baseline is not None
        w = baseline if dense else self.packed
        cores, assignments = _cores(ttnn, self.device, n_tiles)
        output_dtype = ttnn.bfloat16 if bias is None else ttnn.float32
        page = 2048 if bias is None else 4096
        output = self._allocate((m_tiles * 32, n_tiles * 32), output_dtype, ttnn.TILE_LAYOUT)
        readers = [x, w, self.escapes]
        addresses = [tensor.buffer_address() for tensor in readers]
        reader = _kernel(
            ttnn, "linear_reader", ttnn.ReaderConfigDescriptor(), cores, assignments,
            (n // 16, k_tiles, m_tiles, m_block, self.exp_base, int(dense)),
            readers, lambda start, end: [*addresses, start, end],
        )
        compute = _kernel(
            ttnn, "linear_compute", ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True), cores, assignments,
            (k_tiles, m_tiles, m_block), (), lambda start, end: [end - start],
        )
        writer = _kernel(
            ttnn, "linear_writer", ttnn.WriterConfigDescriptor(), cores, assignments,
            (n_tiles, m_tiles, page), (output,), lambda start, end: [output.buffer_address(), start, end],
        )
        program = ttnn.ProgramDescriptor(
            kernels=[reader, compute, writer], semaphores=[], cbs=[
                _cb(ttnn, cores, 0, ttnn.bfloat16, 2048, 2),
                _cb(ttnn, cores, 1, ttnn.bfloat16, 2048, 2 * m_block),
                _cb(ttnn, cores, 16, output_dtype, page, 2),
                _cb(ttnn, cores, 24, ttnn.uint32, _LINEAR_SCRATCH),
            ],
        )
        result = ttnn.generic_op([*readers, output], program)
        if bias is not None:
            result = ttnn.add(result, bias, dtype=ttnn.bfloat16)
        return result

    def linear(self, x, bias=None):
        if x.ndim == 0 or x.shape[-1] != self.descriptor.logical_shape[1]:
            raise ValueError("NF12 linear input does not match logical K")
        n = self.descriptor.logical_shape[0]
        rows = x.reshape(-1, x.shape[-1])
        if not rows.numel():
            if x.dtype != torch.bfloat16 or x.device.type != "cpu":
                raise ValueError("Native NF12 expects CPU BF16 activations")
            return x.new_empty((*x.shape[:-1], n))
        out = self.linear_device(self.upload_activation(rows), None if bias is None else self.upload_bias(bias))
        return import_ttnn().to_torch(out)[: len(rows), :n].reshape(*x.shape[:-1], n)
