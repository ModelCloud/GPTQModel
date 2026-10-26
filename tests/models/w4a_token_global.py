# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Experimental per-token outer NVFP4 scales for disjoint selection traces."""

from collections import defaultdict

import torch


def token_global_scale(source: torch.Tensor) -> torch.Tensor:
    """Choose FP32 row scales with room for both the M=4 and M=6 candidates."""
    if source.ndim < 2 or not source.shape[-1] or not source.is_floating_point():
        raise ValueError("Token global scales require floating token tensors")
    values = source.float().reshape(-1, source.shape[-1])
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Token global scales require finite values")
    maximum = values.abs().amax(-1)
    # CUDA scalar division may use a rounded reciprocal. A one-ULP difference
    # in this outer scale can move normalized inputs across an FP4 tie. Use
    # the higher-precision quotient for this diagnostic and store only FP32;
    # a future fused implementation must reproduce it with div.rn.f32.
    quotient = (maximum.double() / (448. * 4.)).float()
    scale = torch.where(maximum > 0, quotient, torch.ones_like(maximum))
    if not bool(torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("Token global scale is not representable in positive FP32")
    return scale


def pack_token_global(source: torch.Tensor, *, model_dtype, rotation_applied=False):
    """Use native E2M1/E4M3 operands and one outer FP32 multiplier per token."""
    from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation

    scale = token_global_scale(source)
    rows = source.float().reshape(-1, source.shape[-1])
    normalized = (rows / scale[:, None]).reshape_as(source)
    packed = pack_activation(normalized, "w4a_nvfp4", model_dtype=model_dtype,
                             recipe="least_squares", rotation_applied=rotation_applied,
                             global_scale=torch.ones((), device=source.device, dtype=torch.float32))
    return packed.rescale_tokens(scale)


def install_token_global_diagnostic(core):
    """Replace producer results without changing checkpoint parameters or policy.

    Repacking in a post-hook is intentionally diagnostic. A selected production
    recipe would need its own fused packer, replay, and serialized policy.
    """
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer

    modules = [m for m in core.modules() if isinstance(m, NVFP4BoundaryQuantizer)]
    if not modules or any(m.calibrated or m.observer is not None or m._forward_hooks for m in modules):
        raise ValueError("Token global diagnostics require idle, uncalibrated version-4 producers")
    stats = defaultdict(lambda: {"tokens": 0, "minimum": float("inf"), "maximum": 0.})

    def replace(_module, args, output):
        if (not isinstance(output, W4AActivation) or output.mode != "w4a_nvfp4"
                or output.recipe != "least_squares"):
            raise ValueError("Token global diagnostics require least_squares NVFP4")
        result = pack_token_global(args[0], model_dtype=output.model_dtype,
                                   rotation_applied=output.rotation_applied)
        row = stats[_module.key]
        row["tokens"] += result.token_scale.numel()
        if result.token_scale.numel():
            row["minimum"] = min(row["minimum"], float(result.token_scale.min()))
            row["maximum"] = max(row["maximum"], float(result.token_scale.max()))
        return result

    handles = []
    try:
        for module in modules:
            handles.append(module.register_forward_hook(replace))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles, stats
