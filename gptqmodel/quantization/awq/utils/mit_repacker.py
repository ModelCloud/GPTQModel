# SPDX-FileCopyrightText: 2023 MIT Han Lab
# SPDX-License-Identifier: Apache-2.0

"""Helpers mirrored from MIT Han Lab's AWQ offline weight repacker.

Source:
https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/offline-weight-repacker.py

These utilities convert legacy AWQ v1 tensors into the layout expected by the
v2 CUDA kernels. The implementations are intentionally close to the reference
script to minimise drift; only minimal Torch wrappers are added so they can run
on tensors that may already live on device memory.
"""

from __future__ import annotations

import torch


def qweight_unpack(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack int4 weights into individual nibbles (reference implementation)."""
    if qweight.dtype != torch.int32:
        qweight = qweight.to(torch.int32)
    n = qweight.shape[0]
    k = qweight.shape[1] * 8
    unpacked = torch.zeros((n, k), dtype=torch.int32, device=qweight.device)
    mask = torch.tensor(0x0000000F, dtype=torch.int32, device=qweight.device)
    for kk in range(k):
        ele_offset = kk // 8
        bit_offset = (kk % 8) * 4
        unpacked[:, kk] = (qweight[:, ele_offset] >> bit_offset) & mask
    return unpacked


def packing_v2_from_unpacked(
    unpacked_qweight: torch.Tensor, interleave: int, kstride: int
) -> torch.Tensor:
    """Pack unpacked weights into the v2 kernel layout (reference implementation)."""
    n = unpacked_qweight.shape[0]
    k = unpacked_qweight.shape[1]

    packed_kernel = (
        unpacked_qweight.detach()
        .cpu()
        .numpy()
        .reshape(n, k // 32, 32)
    )
    packed_kernel = packed_kernel.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    packed_kernel = packed_kernel.reshape(n, k // 32, 32)

    packed_kernel = packed_kernel.reshape(n, k // 32, 4, 8)
    packed_kernel = packed_kernel.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    packed_kernel = packed_kernel.reshape(n, k)

    packed_kernel = packed_kernel.reshape(n // interleave, interleave, k // kstride, kstride)
    packed_kernel = packed_kernel.transpose(0, 2, 1, 3)
    packed_kernel = packed_kernel.reshape(n // interleave, k // kstride, kstride, interleave)
    packed_kernel = (
        packed_kernel[..., 0]
        | (packed_kernel[..., 1] << 4)
        | (packed_kernel[..., 2] << 8)
        | (packed_kernel[..., 3] << 12)
    )
    packed_kernel = packed_kernel.reshape(n // interleave, k)
    qweight_v2 = torch.tensor(packed_kernel.astype("int16"), device=unpacked_qweight.device).contiguous()
    return qweight_v2


def multiply_scale_qzero_negative(
    scales: torch.Tensor, qzeros: torch.Tensor, zp_shift: int = 0
) -> torch.Tensor:
    """Compute scaled zero-points in the format consumed by v2 kernels."""
    pack_size = 8
    k_groups = scales.shape[1]
    scaled_zeros = torch.zeros_like(scales)
    qzeros = qzeros.to(torch.int32)
    for group_idx in range(k_groups):
        zero_idx = group_idx // pack_size
        zero_offset = group_idx % pack_size
        zero = (qzeros[:, zero_idx] >> (4 * zero_offset)) & 0x0000000F
        scaled_zeros[:, group_idx] = scales[:, group_idx] * zero.to(scales.dtype)
    return -(scaled_zeros + (zp_shift * scales))


__all__ = [
    "qweight_unpack",
    "packing_v2_from_unpacked",
    "multiply_scale_qzero_negative",
]
