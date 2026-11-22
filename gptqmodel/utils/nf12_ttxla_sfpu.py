# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Single-copy Blackhole SFPU LM head for the bundled TT-XLA artifact ABI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import pad

from .nf12_tenstorrent import NF12Descriptor, round_up, tensor_sha256
from .nf12_ttnn_sfpu import _to_tiles, _transcode
from .nf12_ttxla import _blackhole_device, _call, _install_resolver, _tiled_words


@dataclass(frozen=True)
class _TTXLASFPUStorageDescriptor:
    layout: str
    planes_sha256: str
    metadata_sha256: str


def _pack_xla_planes(planes, patches, base):
    if planes.dtype not in (torch.int16, torch.uint16):
        raise TypeError("TT-XLA SFPU expects the raw 16-bit plane words")
    # Place the UInt16 tile faces in raw byte order, then carry two planes
    # per physical INT32 page in PJRT.
    raw = planes.view(torch.int16).reshape(-1, 2, 16, 2, 16).transpose(2, 3).contiguous().view(torch.int32)
    packed = _tiled_words(raw)
    if patches:
        headers, escapes = patches
        metadata = torch.cat((headers.flatten(), escapes.flatten()))
    else:
        metadata = torch.zeros(round_up(2 * (planes.shape[0] // 3), 16), dtype=torch.int32)
    carried = _tiled_words(metadata)
    # The logical height residue makes per-matrix compile-time bases distinct
    # in Torch/XLA's same-shape executable cache. It sits after the payload.
    height = round_up(carried.shape[0], 128) + base
    return packed, pad(carried, (0, 0, 0, height - carried.shape[0]))


def _launch(activation, words, metadata, n, base, groups=1):
    m, k = activation.shape
    out = torch.empty((m, n), dtype=torch.bfloat16, device=activation.device)
    return _call("sfpu_linear", (n, k, m, base, groups, int(words.dtype == torch.bfloat16)),
                 (activation, words, metadata), out)


class TTXLASFPUNF12Head:
    """Bias-free head with compressed DRAM weights and one decoded slab per worker."""

    def __init__(self, device, descriptor: NF12Descriptor, weight):
        self.device = _blackhole_device(device)
        _install_resolver()
        planes, patches = _transcode(descriptor, weight)
        packed, metadata = _pack_xla_planes(planes, patches, weight.exp_base)
        self.storage_descriptor = _TTXLASFPUStorageDescriptor(
            "nf12-tt-sfpu-pjrt-int32-v1", tensor_sha256(packed), tensor_sha256(metadata),
        )
        self.descriptor = descriptor
        self.exp_base = weight.exp_base
        self.planes = packed.to(self.device)
        self.metadata = metadata.to(self.device)

    @property
    def resident_bytes(self):
        return self.planes.nbytes + round_up(self.metadata.shape[0], 32) * 128

    def upload_baseline(self, weight):
        if tuple(weight.shape) != self.descriptor.logical_shape or weight.dtype != torch.bfloat16:
            raise ValueError("Matched SFPU baseline expects the original BF16 weight")
        return _to_tiles(weight).transpose(-2, -1).reshape(-1, 32).to(self.device)

    def linear(self, x, bias=None, *, baseline=None):
        n, k = self.descriptor.logical_shape
        if bias is not None:
            raise ValueError("SFPU NF12 LM head expects no bias")
        if x.device != self.device or x.dtype != torch.bfloat16 or not x.ndim or x.shape[-1] != k:
            raise ValueError("SFPU NF12 expects BF16 activations with the logical K on the packed weight's device")
        if baseline is not None and (tuple(baseline.shape) != (n * k // 32, 32)
                                     or baseline.dtype != torch.bfloat16 or baseline.device != self.device):
            raise ValueError("Unexpected TT-XLA SFPU baseline shape, dtype, or device")
        if not x.numel():
            return x.new_empty((*x.shape[:-1], n))

        words = self.planes if baseline is None else baseline
        rows = x.reshape(-1, k)
        activation = pad(rows, (0, 0, 0, round_up(len(rows), 32) - len(rows)))
        # One executable traverses the token slabs. Separate lazy Generic
        # operations can lose their outputs when TT-XLA lowers concatenation.
        result = _launch(activation, words, self.metadata, n, self.exp_base)[:len(rows)]
        return result.reshape(*x.shape[:-1], n)
