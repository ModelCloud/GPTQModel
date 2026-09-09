# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch


def awq_gemv_codes(weight, scales, zeros, g_idx):
    """GEMV's source-offset/stored-FP16-scale inverse, with bounded int4 codes."""
    if (weight.ndim != 2 or scales.ndim != 2 or scales.shape != zeros.shape
            or scales.shape[0] != weight.shape[0] or g_idx.shape != (weight.shape[1],)
            or not weight.numel() or not scales.numel()):
        raise ValueError("AWQ GEMV requires matching weight, group-scale and zero-point shapes")
    if any(t.device != weight.device for t in (scales, zeros, g_idx)):
        raise ValueError("AWQ GEMV packing tensors must share a device")
    if g_idx.dtype not in (torch.int32, torch.int64) or g_idx.min() < 0 or g_idx.max() >= scales.shape[1]:
        raise ValueError("AWQ GEMV group indices must be in-range integers")
    if any(not torch.isfinite(t).all() for t in (weight, scales, zeros)):
        raise ValueError("AWQ GEMV weights, scales and zero-points must be finite")
    stored = scales.half()
    if (scales <= 0).any() or (stored <= 0).any() or not torch.isfinite(stored).all():
        raise ValueError("AWQ GEMV scales must be positive and finite in FP16 storage")
    if (zeros != zeros.round()).any() or (zeros < 0).any() or (zeros > 15).any():
        raise ValueError("AWQ GEMV zero-points must be integers in [0,15]")
    offset = zeros * scales
    if not torch.isfinite(offset).all():
        raise ValueError("AWQ GEMV source offsets must be finite")
    # Preserve the producer's dtype and operation order. Clamp before the
    # integer cast so endpoint drift cannot wrap into unrelated packed codes.
    return ((weight + offset[:, g_idx]) / stored[:, g_idx]).round().clamp(0, 15)


def dequantize_awq_gemv(qweight, scales, zeros, *, group_size, in_features, out_features, fast=False):
    """Decode GEMV or GEMV_FAST storage to [in,out], in the scale dtype.

    Passing FP32 scale values gives the canonical FP32 operator; ordinary
    checkpoint scales produce FP16 weights. Fast zeros are additive offsets.
    """
    n, k = out_features, in_features
    groups = torch.arange(k, device=qweight.device) // group_size
    if fast:
        shifts = torch.arange(0, 16, 4, device=qweight.device)
        codes = ((qweight.long().unsqueeze(-1) >> shifts) & 15)
        codes = codes.reshape(n//4, k//64, 4, 64).transpose(1, 2).reshape(n, k)
        codes = codes.reshape(n, k//32, 4, 2, 4).transpose(3, 4).reshape(n, k)
        codes = codes.reshape(n, k//32, 4, 4, 2).transpose(2, 3).reshape(n, k)
        weight = scales.T.float()[:, groups]*codes + zeros.T.float()[:, groups]
    else:
        shifts = torch.arange(0, 32, 4, device=qweight.device)
        codes = ((qweight.long().unsqueeze(-1) >> shifts) & 15).reshape(n, k)
        zero_codes = ((zeros.long().unsqueeze(-1) >> shifts) & 15).reshape(n, -1)
        weight = scales.float()[:, groups]*(codes-zero_codes[:, groups])
    return weight.T.to(scales.dtype).contiguous()

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width
