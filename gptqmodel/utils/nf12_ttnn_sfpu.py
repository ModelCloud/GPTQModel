# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Blackhole bias-free NF12 LM head with on-worker SFPU reconstruction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn.functional import pad

from .nf12_codec import decode
from .nf12_tenstorrent import NF12Descriptor, round_up, tensor_sha256, verify_nf12
from .nf12_ttnn import _cb, _cores, _kernel
from .ttnn_runtime import import_ttnn


_MAX_ACTIVATION_TILES = 8


@dataclass(frozen=True)
class _SFPUStorageDescriptor:
    layout: str
    planes_sha256: str
    patch_headers_sha256: str | None
    patch_values_sha256: str | None


def _to_tiles(matrix):
    n, k = matrix.shape
    return matrix.reshape(n // 32, 32, k // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)


def _plane_components(tiles, base):
    assert tiles.dtype == torch.int32
    assert tiles.shape[0] % 4 == 0
    delta = ((tiles >> 8) & 0x7f) - base
    escaped = (delta < 0) | (delta > 7)
    high = (delta.clamp(0, 7) | ((tiles >> 12) & 8)).reshape(-1, 4, 32, 32)
    low = (tiles & 0xff).reshape_as(high)
    a, b, c, d = high.unbind(1)
    planes = torch.stack((low[:, 0] | (low[:, 1] << 8), low[:, 2] | (low[:, 3] << 8),
                          a | (b << 4) | (c << 8) | (d << 12)), dim=1).reshape(-1, 32, 32)
    block, tile, row, col = escaped.reshape_as(high).nonzero(as_tuple=True)
    # Each patch names a physical slot in four 32 x 32 tiles and its raw BF16 word.
    slot = (row // 16 * 2 + col // 16) * 256 + row % 16 * 16 + col % 16
    patches = ((tiles.reshape_as(high)[block, tile, row, col] << 12) | (tile * 1024 + slot)).int()
    counts = torch.bincount(block, minlength=high.shape[0])
    offsets = counts.cumsum(0) - counts
    return planes, torch.stack((offsets, counts), dim=1).int(), patches


def _metadata_pages(headers, patches):
    if not len(patches):
        return ()
    return tuple(pad(data, (0, round_up(len(data), 16) - len(data))).reshape(-1, 16)
                 for data in (headers.flatten(), patches))


def _pack_four(tiles, base):
    planes, headers, patches = _plane_components(tiles, base)
    return planes, _metadata_pages(headers, patches)


def _transcode(descriptor, weight):
    n, k = descriptor.logical_shape
    if (n, k) != descriptor.padded_shape or n % 32 or k % 128:
        raise ValueError("SFPU NF12 requires unpadded N divisible by 32 and K divisible by 128")
    verify_nf12(descriptor, weight)
    group_count = n // 32 * (k // 128)
    planes = torch.empty((group_count * 3, 32, 32), dtype=torch.uint16)
    headers = torch.empty((group_count, 2), dtype=torch.int32)
    patch_parts = []
    source_hash = hashlib.sha256()
    plane_offset = header_offset = patch_count = 0
    for packed in weight.packed.split(512):
        decoded = decode(packed, weight.escapes, weight.exp_base)
        source_hash.update(decoded.contiguous().view(torch.uint8).numpy())
        bits = decoded.view(torch.int16).int() & 0xffff
        tiles = _to_tiles(bits).transpose(-2, -1).contiguous()
        current_planes, current_headers, patches = _plane_components(tiles, weight.exp_base)
        current_headers[:, 0].add_(patch_count)
        planes[plane_offset : plane_offset + len(current_planes)].copy_(current_planes)
        headers[header_offset : header_offset + len(current_headers)].copy_(current_headers)
        patch_parts.append(patches)
        plane_offset += len(current_planes)
        header_offset += len(current_headers)
        patch_count += len(patches)
    if source_hash.hexdigest() != descriptor.source_sha256:
        raise ValueError(f"NF12 source hash mismatch after decoding {descriptor.tensor_name}")
    return planes, _metadata_pages(headers, torch.cat(patch_parts))


def _packed_pages(k_tiles, prefetch_groups):
    return min(prefetch_groups * 6, k_tiles // 4 * 3)


def _linear_l1_bytes(k_tiles, activation_tiles, *, dense, metadata, prefetch_groups, single_row=False):
    pages = (activation_tiles + 1) * k_tiles + 2
    if not dense:
        pages += _packed_pages(k_tiles, prefetch_groups)
    return pages * 2048 + 128 * int(bool(metadata)) + 64 * int(single_row)


def _linear_operation(ttnn, device, words, activation, n, k, base, *, dense, metadata, prefetch_groups, single_row=False):
    m, activation_k = activation.shape
    activation_tiles = m // 32
    if (activation_k != k or m % 32 or not 1 <= activation_tiles <= _MAX_ACTIVATION_TILES
            or (single_row and activation_tiles != 1)):
        raise ValueError("SFPU activation group must fit between one and eight full input tiles")
    shape = (n // 32, 32) if single_row else (m, n)
    layout = ttnn.ROW_MAJOR_LAYOUT if single_row else ttnn.TILE_LAYOUT
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, layout,
                                           device, ttnn.DRAM_MEMORY_CONFIG)
    cores, assignments = _cores(ttnn, device, n // 32)
    k_tiles = k // 32
    geometry = (k_tiles, int(dense))
    reader = _kernel(ttnn, "sfpu_linear_reader", ttnn.ReaderConfigDescriptor(), cores, assignments,
                     (*geometry, prefetch_groups, activation_tiles), [words, activation],
                     lambda start, end: [words.buffer_address(), activation.buffer_address(), start, end])
    compute = _kernel(ttnn, "sfpu_linear_compute", ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
                      cores, assignments, (*geometry, base, activation_tiles), (), lambda start, end: [end - start])
    metadata_addresses = [tensor.buffer_address() for tensor in metadata]
    writer = _kernel(ttnn, "sfpu_linear_writer", ttnn.WriterConfigDescriptor(), cores, assignments,
                     (*geometry, int(bool(metadata)), int(single_row), n // 32, activation_tiles), [output, *metadata],
                     lambda start, end: [output.buffer_address(), start, end, *metadata_addresses])
    weight_views = [ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=2048)]
    if not dense:
        weight_views.append(ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.uint16, page_size=2048))
    buffers = [
        _cb(ttnn, cores, 1, ttnn.bfloat16, 2048, activation_tiles * k_tiles),
        ttnn.CBDescriptor(total_size=k_tiles * 2048, core_ranges=cores, format_descriptors=weight_views),
        _cb(ttnn, cores, 17, ttnn.bfloat16, 2048, 2),
    ]
    if not dense:
        buffers.append(_cb(ttnn, cores, 0, ttnn.uint16, 2048, _packed_pages(k_tiles, prefetch_groups)))
    if metadata:
        buffers.append(_cb(ttnn, cores, 24, ttnn.uint32, 128))
    if single_row:
        buffers.append(_cb(ttnn, cores, 25, ttnn.uint32, 64))
    program = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=buffers)
    return lambda: ttnn.generic_op([words, activation, *metadata, output], program)


class TTNNSFPUNF12Head:
    """One derived NF12 copy for a bias-free Inkling LM head on Blackhole."""

    def __init__(self, device, descriptor: NF12Descriptor, weight):
        ttnn = import_ttnn()
        if device.arch() != ttnn.Arch.BLACKHOLE or device.get_num_devices() != 1:
            raise ValueError("SFPU NF12 requires a single Blackhole chip")
        planes, metadata = _transcode(descriptor, weight)
        metadata_hashes = tuple(tensor_sha256(tensor) for tensor in metadata) or (None, None)
        self.storage_descriptor = _SFPUStorageDescriptor(
            "nf12-tt-sfpu-planes-v1", tensor_sha256(planes), *metadata_hashes,
        )
        self.device = device
        self.descriptor = descriptor
        self.exp_base = weight.exp_base
        self.planes = ttnn.from_torch(planes, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.metadata = tuple(ttnn.from_torch(tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                             device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                              for tensor in metadata)

    @property
    def resident_bytes(self):
        return self.planes.volume() * 2 + sum(tensor.volume() * 4 for tensor in self.metadata)

    def linear(self, x, bias=None):
        n, k = self.descriptor.logical_shape
        if bias is not None:
            raise ValueError("SFPU NF12 LM head expects no bias")
        if x.device.type != "cpu" or x.dtype != torch.bfloat16 or not x.ndim or x.shape[-1] != k:
            raise ValueError("SFPU NF12 expects CPU BF16 activations with the logical K")
        ttnn = import_ttnn()
        rows = x.reshape(-1, k)
        result = x.new_empty((len(rows), n))
        for start in range(0, len(rows), 32):
            slab = rows[start : start + 32]
            activation = ttnn.from_torch(pad(slab, (0, 0, 0, 32 - len(slab))),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            call = _linear_operation(ttnn, self.device, self.planes, activation, n, k, self.exp_base,
                                     dense=False, metadata=self.metadata, prefetch_groups=1, single_row=len(slab) == 1)
            host_output = ttnn.to_torch(call()).reshape(-1, n)
            result[start : start + len(slab)].copy_(host_output[: len(slab)])
        return result.reshape(*x.shape[:-1], n)
