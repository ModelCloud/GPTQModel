"""Frozen exported ParoQuant coordinate system for a GSQ candidate adapter.

This module prepares the reconstruction problem; lifecycle integration and
native validation are separate. It does not enable GSQ in ParoConfig.
"""

import torch

from .paroquant.optimization import _apply_rotation


def paro_gsq_basis(weight, inputs, pairs, theta, channel_scales, *, group_size, storage_dtype=torch.float16):
    """Return teacher W S^-1 R and calibration X S R in export coordinates.

    Runtime applies X S R, where S contains *exported reciprocal* channel
    scales. Do not substitute the optimizer's channel-scale parameters for S.
    Freeze angles/scales after their export dtype conversion. Accumulate the
    reference transform in at least FP32; this is not native FP16 kernel parity.
    """
    if weight.ndim != 2 or inputs.ndim != 2 or weight.shape[1] != inputs.shape[1]:
        raise ValueError("Paro GSQ requires [out,in] weights and [tokens,in] inputs")
    width = weight.shape[1]
    if min(weight.shape) == 0 or inputs.shape[0] == 0:
        raise ValueError("Paro GSQ requires nonempty weights and calibration")
    group = width if group_size == -1 else group_size
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0 or group % 2 or width % group:
        raise ValueError("Paro GSQ requires an even group size dividing input width")
    if storage_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Paro GSQ requires a supported floating export dtype")
    if pairs.ndim != 2 or pairs.shape[1] != width or theta.shape != (pairs.shape[0], width // 2):
        raise ValueError("Paro GSQ rotation metadata dimensions do not match input width")
    if pairs.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("Paro GSQ pairs must be integer permutations")
    grouped_pairs = pairs.reshape(pairs.shape[0], width // group, group).long()
    expected = torch.arange(group, device=pairs.device).expand_as(grouped_pairs)
    if not torch.equal(grouped_pairs.sort(dim=-1).values, expected):
        raise ValueError("Paro GSQ pairs must be disjoint permutations within each group")
    if channel_scales.numel() != width:
        raise ValueError("Paro GSQ requires one exported channel scale per input")
    if any(t.device != weight.device for t in (inputs, pairs, theta, channel_scales)):
        raise ValueError("Paro GSQ tensors must share a device")
    if any(not t.is_floating_point() or not torch.isfinite(t).all()
           for t in (weight, inputs, theta, channel_scales)):
        raise ValueError("Paro GSQ requires finite floating weights, inputs and transform metadata")
    dtype = torch.float64 if torch.float64 in (weight.dtype, inputs.dtype) else torch.float32
    angles = theta.to(storage_dtype).to(dtype)
    scales = channel_scales.reshape(-1).to(storage_dtype).to(dtype)
    if not torch.isfinite(angles).all() or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Paro GSQ exported transform must remain finite with positive channel scales")
    transformed_teacher = _apply_rotation(weight.to(dtype) / scales, pairs, angles,
                                          scales=None, group_size=group, fused_rotation=False)
    transformed_inputs = _apply_rotation(inputs.to(dtype), pairs, angles,
                                         scales=scales, group_size=group, fused_rotation=False)
    if not torch.isfinite(transformed_teacher).all() or not torch.isfinite(transformed_inputs).all():
        raise ValueError("Paro GSQ transformed reconstruction problem is nonfinite")
    return transformed_teacher, transformed_inputs
