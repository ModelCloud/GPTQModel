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


def refine_paro_export(result, *, teacher, inputs, group_size, config=None, storage_dtype=torch.float16,
                       teacher_inputs=None):
    """Fit the affine grid after rotation export; return tensors and diagnostics.

    The returned dictionary deliberately excludes initializer train/validation
    losses: callers must not relabel those as post-GSQ losses. Rotation metadata
    remains owned by ``result``. Optional row-aligned ``teacher_inputs`` encode
    clean targets while ``inputs`` encode noisy runtime activations. In that
    case losses omit the candidate-independent asymmetric constant; they can
    be negative and are not normalized clean-target MSE. This low-level adapter
    is not processor wiring.
    """
    from .config import normalize_gsq_config
    from .gsq_scalar import affine_codes, refine_affine_scalar
    from .paroquant.optimization import _apply_inverse_rotation

    config = normalize_gsq_config(config)

    def retained(before=None, after=None, history=None):
        return dict(pack_weight=result.pack_weight.clone(), pseudo_weight=result.pseudo_weight.clone(),
                    q_scales=result.q_scales.clone(), q_zeros=result.q_zeros.clone(),
                    before=before, after=after, history=[] if history is None else history)

    if config is None or not config.enabled:
        return retained()
    if storage_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Paro GSQ fitting requires FP16 or BF16 exported metadata")
    target, features = paro_gsq_basis(teacher, inputs, result.pairs, result.theta,
                                     result.channel_scales, group_size=group_size, storage_dtype=storage_dtype)
    cross = None
    if teacher_inputs is not None:
        if teacher_inputs.shape != inputs.shape:
            raise ValueError("Paro GSQ clean/noisy inputs must have identical row-aligned shapes")
        _, clean_features = paro_gsq_basis(teacher, teacher_inputs, result.pairs, result.theta,
                                          result.channel_scales, group_size=group_size, storage_dtype=storage_dtype)
        # E = candidate - teacher; expand ||X_noisy E^T -
        # (X_clean-X_noisy) teacher^T||^2 and omit its constant.
        cross = (clean_features.float() - features.float()).T @ features.float()
    width = teacher.shape[1]
    group = width if group_size == -1 else group_size
    groups = torch.arange(width, device=teacher.device) // group
    fitted = refine_affine_scalar(
        result.pack_weight.to(storage_dtype), result.q_scales.to(storage_dtype), result.q_zeros, groups,
        target=target, bits=4, config=config, inputs=features, cross_moment=cross,
        packing="awq_gemm", scale_dtype=storage_dtype)
    if fitted.after >= fitted.before:
        return retained(fitted.before, fitted.after, fitted.history)
    # Replay must use the actual packed grid, not merely the floating transport
    # weights: export casts can otherwise desynchronize replay and inference.
    codes = affine_codes(fitted.weight, fitted.scales, fitted.zeros, groups, 4,
                         packing="awq_gemm", scale_dtype=storage_dtype)
    packed_grid = fitted.scales.to(storage_dtype).float()[:, groups] * (codes.float() - fitted.zeros[:, groups].float())
    angles = result.theta.to(storage_dtype).float()
    channel = result.channel_scales.to(storage_dtype).float()
    pseudo = _apply_inverse_rotation(packed_grid, result.pairs, angles,
                                     group_size=group, fused_rotation=False) * channel.reshape(-1)
    return dict(pack_weight=fitted.weight, pseudo_weight=pseudo, q_scales=fitted.scales, q_zeros=fitted.zeros,
                before=fitted.before, after=fitted.after, history=fitted.history)
