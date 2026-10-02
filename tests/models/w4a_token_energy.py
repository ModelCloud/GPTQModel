# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Experimental token-energy correction for disjoint-corpus diagnostics only.

This uses the existing outer FP32 token multiplier. It never stores a dense
residual, changes FP4 codes/block scales, or writes model/checkpoint settings.
"""

from collections import defaultdict

import torch


def token_energy_gain(source: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
    """Return ||source|| / ||decoded|| per token, in FP32.

    Normalize both operands by their shared maximum before accumulation to
    avoid squaring very small/large FP32 values. Zero/zero needs no correction;
    a nonzero token rounded entirely to zero cannot be corrected by a scalar.
    """
    if (source.shape != decoded.shape or source.ndim < 2 or not source.shape[-1]
            or source.device != decoded.device or not source.is_floating_point()
            or not decoded.is_floating_point()):
        raise ValueError("Energy correction requires matching floating token tensors")
    source = source.float().reshape(-1, source.shape[-1])
    decoded = decoded.float().reshape_as(source)
    if not bool(torch.isfinite(source).all() and torch.isfinite(decoded).all()):
        raise ValueError("Energy correction requires finite activations")
    maximum = torch.maximum(source.abs().amax(-1), decoded.abs().amax(-1))
    divisor = torch.where(maximum > 0, maximum, torch.ones_like(maximum))[:, None]
    left = (source / divisor).square().sum(-1)
    right = (decoded / divisor).square().sum(-1)
    if bool(((left > 0) & (right == 0)).any()):
        raise ValueError("Scalar correction cannot recover an entirely zero encoded token")
    gain = torch.sqrt(left / torch.where(right > 0, right, torch.ones_like(right)))
    gain = torch.where((left == 0) & (right == 0), torch.ones_like(gain), gain)
    if not bool(torch.isfinite(gain).all()):
        raise ValueError("Nonfinite token-energy correction")
    return gain


def install_token_energy_diagnostic(core, scope: str):
    """Attach removable producer hooks; caller must remove every returned hook."""
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
    from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer

    if scope not in {"all", "residual"}:
        raise ValueError("Energy diagnostic scope must be all or residual")
    modules = [module for module in core.modules() if isinstance(module, NVFP4BoundaryQuantizer)]
    if not modules:
        raise ValueError("Energy diagnostics need explicit version-4 NVFP4 producers")
    selected = [module for module in modules if scope == "all" or (
        module.key.endswith((".input", ".post_attention_residual", ".output"))
        and ".self_attn." not in module.key and ".mlp." not in module.key)]
    if not selected:
        raise ValueError("No producers selected for energy diagnostics")
    stats = defaultdict(lambda: {"tokens": 0, "gain_sum": 0., "gain_min": float("inf"), "gain_max": 0.})

    def correct(module, args, output):
        if not isinstance(output, W4AActivation) or output.mode != "w4a_nvfp4":
            raise TypeError("Energy correction requires an encoded NVFP4 producer result")
        gain = token_energy_gain(args[0], output.decode(torch.float32))
        result = output.rescale_tokens(gain)
        if result.codes is not output.codes or result.scales is not output.scales:
            raise AssertionError("Token correction must preserve packed codes and hardware scales")
        row = stats[module.key]
        row["tokens"] += gain.numel()
        row["gain_sum"] += float(gain.double().sum())
        if gain.numel():
            row["gain_min"] = min(row["gain_min"], float(gain.min()))
            row["gain_max"] = max(row["gain_max"], float(gain.max()))
        return result

    handles = []
    try:
        for module in selected:
            handles.append(module.register_forward_hook(correct))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles, stats
