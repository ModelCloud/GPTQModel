# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""QQQ deployed candidate values for GSQ.

QQQ grouped W4 codes undergo a second INT8 rounding step. Decode each hard
candidate before mixing probabilities; decoding an expected nibble would erase
both the true grid and its useful gradient.
"""

import torch


def qqq_candidate_values(codes, scales, *, group_size, in_features, channel_scales=None):
    """Decode [..., candidate] nibbles with weight axes [out,in].

    ``scales`` is the raw [out,groups] producer table. Grouped exports also
    require the FP32 per-output-channel scale. Returns FP32 effective weights,
    including the channel multiplier, before activation quantization. This
    models the Torch runtime grid, not a native-kernel accuracy guarantee.
    """
    if codes.ndim not in (2, 3) or codes.shape[1] != in_features or in_features <= 0:
        raise ValueError("QQQ codes must be [out,in] or [out,in,candidates]")
    if group_size == -1:
        group_size = in_features
    if group_size <= 0 or in_features % group_size:
        raise ValueError("QQQ group size must divide input width")
    if scales.shape != (codes.shape[0], in_features // group_size):
        raise ValueError("QQQ scales must be [out,groups]")
    if scales.device != codes.device or not scales.is_floating_point():
        raise ValueError("QQQ scales must be floating and on the codes device")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("QQQ scales must be finite and positive")
    if not torch.isfinite(codes).all() or (codes < 0).any() or (codes > 15).any():
        raise ValueError("QQQ nibbles must be finite and in [0,15]")
    if codes.is_floating_point() and (codes != codes.round()).any():
        raise ValueError("QQQ candidates must be integer nibbles")
    indices = torch.arange(in_features, device=codes.device) // group_size
    extra_axis = codes.ndim == 3
    values = codes.float()
    if group_size == in_features:
        # Packer divides in source scale precision before FP32 storage.
        channel = (scales / 16).float()
        if not torch.isfinite(channel).all() or (channel <= 0).any():
            raise ValueError("QQQ channel scales must remain finite and positive after storage")
        signed = torch.where(values >= 8, values - 16, values)
        multiplier = channel.unsqueeze(-1) if extra_axis else channel
        return signed * 16 * multiplier
    if channel_scales is None or channel_scales.numel() != codes.shape[0]:
        raise ValueError("grouped QQQ requires one channel scale per output")
    if channel_scales.device != codes.device or not channel_scales.is_floating_point():
        raise ValueError("QQQ channel scales must be floating and on the codes device")
    channel = channel_scales.reshape(-1, 1).float()
    if not torch.isfinite(channel).all() or (channel <= 0).any():
        raise ValueError("QQQ channel scales must be finite and positive")
    ratio = (scales / channel).half().float()
    if not torch.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("QQQ group ratios must remain finite and positive in FP16")
    ratio = ratio[:, indices]
    if extra_axis:
        ratio = ratio.unsqueeze(-1)
        channel = channel.unsqueeze(-1)
    return ((values - 8) * ratio).round().clamp(-128, 127) * channel


def qqq_calibration_moments(inputs, *, teacher_inputs=None):
    """Return unnormalized (Hq, D, tokens) for deployed-input reconstruction.

    Inputs are [tokens,in] in the same smoothing basis as the weights. Runtime
    casts to FP16 before computing token maxima and dividing by 127; retain
    that precision order. D = (Xteacher-Xq).T @ Xq permits asymmetric fitting
    without an inverse. This objective excludes final output casts/roundoff.
    Add returned moments across batches and normalize both by the same count.
    """
    if inputs.ndim != 2 or min(inputs.shape) <= 0 or not inputs.is_floating_point():
        raise ValueError("QQQ calibration inputs must be nonempty floating [tokens,in]")
    teacher = inputs if teacher_inputs is None else teacher_inputs
    if teacher.shape != inputs.shape or teacher.device != inputs.device or not teacher.is_floating_point():
        raise ValueError("QQQ teacher inputs must match input shape and device")
    if not torch.isfinite(inputs).all() or not torch.isfinite(teacher).all():
        raise ValueError("QQQ calibration inputs must be finite")
    with torch.no_grad():
        runtime = inputs.half()
        if not torch.isfinite(runtime).all():
            raise ValueError("QQQ calibration inputs overflow runtime FP16")
        maximum = runtime.abs().amax(-1, keepdim=True)
        scale = (maximum / 127.0).float()
        if ((maximum > 0) & (scale == 0)).any():
            raise ValueError("QQQ runtime token scale underflows for nonzero inputs")
        # Zero rows contribute exactly zero to both moments. Avoid NaN-to-int
        # conversion, whose behavior is not a portable numerical contract.
        divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = (runtime / divisor).round().clamp(-128, 127).to(torch.int8)
        deployed = codes.float() * scale
        hessian = deployed.T @ deployed
        cross = (teacher.float() - deployed).T @ deployed
        return hessian, cross, inputs.shape[0]


def refine_qqq_codes(codes, scales, *, target, group_size, hessian, cross_moment,
                     channel_scales=None, config=None):
    """Fit fixed-scale stored nibbles; return (best_codes, before, after, history).

    This lower-level optimizer returns codes, not fake-quantized weights. The
    lifecycle must preserve these assignments through its producer/packer
    boundary before this is exposed as supported QQQ quantization.
    """
    from .config import normalize_gsq_config
    from .gsq_scalar import _metric_factor, asymmetric_error_term

    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return codes.clone(), None, None, []
    if config.learn_scales:
        raise ValueError("QQQ GSQ scale learning requires a separate two-scale optimizer")
    if codes.ndim != 2 or target.shape != codes.shape:
        raise ValueError("QQQ GSQ codes and teacher must be matching [out,in]")
    width = codes.shape[1]
    if hessian.shape != (width, width) or cross_moment.shape != (width, width):
        raise ValueError("QQQ GSQ requires matching [in,in] calibration moments")
    if any(t.device != codes.device for t in (target, hessian, cross_moment)):
        raise ValueError("QQQ GSQ calibration and teacher must share the codes device")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (target, hessian, cross_moment)):
        raise ValueError("QQQ GSQ teacher and calibration must be finite floating tensors")
    count = min(config.candidates, 16)
    if codes.numel() * count * 4 > config.max_candidate_bytes:
        raise ValueError("QQQ GSQ decoded candidates exceed max_candidate_bytes")
    with torch.inference_mode(False), torch.enable_grad():
        baseline = codes.detach().clone()
        teacher = target.detach().float().clone()
        fixed_scales = scales.detach().clone()
        fixed_channel = None if channel_scales is None else channel_scales.detach().clone()

        def decode(candidate):
            return qqq_candidate_values(candidate, fixed_scales, group_size=group_size,
                                        in_features=width, channel_scales=fixed_channel)

        base_weight = decode(baseline)
        factor = _metric_factor(hessian.detach())
        cross = cross_moment.detach().float()
        correction = teacher @ cross
        asymmetric_error_term(torch.zeros_like(teacher), teacher, cross)
        energy = (teacher @ factor).square().sum()
        denominator = torch.where(energy > torch.finfo(torch.float32).eps, energy, torch.ones_like(energy))

        def loss(weight):
            error = weight - teacher
            return ((error @ factor).square().sum() - 2 * (error * correction).sum()) / denominator

        channelwise = group_size in (-1, width)
        logical = baseline.float()
        if channelwise:
            logical = torch.where(logical >= 8, logical - 16, logical)
        lo, hi = (-8, 7) if channelwise else (0, 15)
        if count == 16:
            candidate_logical = torch.arange(lo, hi + 1, device=codes.device).float().expand(*codes.shape, 16)
            valid = torch.ones_like(candidate_logical, dtype=torch.bool)
        else:
            offsets = [0] + [(-1)**i * ((i + 1)//2) for i in range(1, count)]
            candidate_logical = logical.unsqueeze(-1) + torch.tensor(offsets, device=codes.device)
            valid = (candidate_logical >= lo) & (candidate_logical <= hi)
            candidate_logical = candidate_logical.clamp(lo, hi)
        candidates = candidate_logical.remainder(16).long()
        values = decode(candidates).detach()
        logits = (-0.5 * (candidate_logical - logical.unsqueeze(-1)).square()).requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
        rng = torch.Generator(device=codes.device).manual_seed(config.seed)
        before = float(loss(base_weight))
        if not torch.isfinite(torch.tensor(before)):
            raise ValueError("non-finite QQQ GSQ baseline objective")
        best, selected_codes, history = before, baseline, [before]
        for step in range(config.steps):
            tau = config.temperature_start * (config.temperature_end / config.temperature_start) ** (
                step / max(config.steps - 1, 1))
            uniform = torch.rand(logits.shape, device=codes.device, generator=rng).clamp(1e-6, 1 - 1e-6)
            noise = -(-uniform.log()).log()
            probabilities = ((logits.masked_fill(~valid, -torch.inf) + noise) / tau).softmax(-1)
            objective = loss((probabilities * values).sum(-1))
            if not torch.isfinite(objective):
                raise ValueError("non-finite QQQ GSQ relaxed objective")
            optimizer.zero_grad()
            objective.backward()
            if not torch.isfinite(logits.grad).all():
                raise ValueError("non-finite QQQ GSQ gradient")
            optimizer.step()
            with torch.no_grad():
                selected = logits.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
                hard = candidates.gather(-1, selected).squeeze(-1)
                score = float(loss(decode(hard)))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError("non-finite QQQ GSQ hard objective")
                history.append(score)
                if score < best:
                    best, selected_codes = score, hard.clone().to(codes.dtype)
        return selected_codes, before, best, history


def qqq_codes_to_packer_weight(codes, scales, *, group_size, dtype):
    """Build a producer weight that the existing QQQ packer encodes exactly.

    This is a transport value, not the deployed INT8-decoded weight. Refuse
    casts that change an assignment; callers must never score one set of codes
    and silently export another.
    """
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("QQQ producer weight dtype must be FP16, BF16 or FP32")
    if codes.ndim != 2:
        raise ValueError("QQQ producer codes must be [out,in]")
    width = codes.shape[1]
    resolved = width if group_size == -1 else group_size
    if width <= 0 or resolved <= 0 or width % resolved:
        raise ValueError("QQQ group size must divide input width")
    if scales.shape != (codes.shape[0], width // resolved) or scales.device != codes.device:
        raise ValueError("QQQ producer scales must match code shape and device")
    if not scales.is_floating_point() or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("QQQ producer scales must be finite positive floating values")
    if not torch.isfinite(codes).all() or (codes < 0).any() or (codes > 15).any():
        raise ValueError("QQQ producer codes must be valid nibbles")
    if codes.is_floating_point() and (codes != codes.round()).any():
        raise ValueError("QQQ producer codes must be integer nibbles")
    groups = torch.arange(width, device=codes.device) // resolved
    raw = scales[:, groups]
    if resolved == width:
        logical = torch.where(codes >= 8, codes - 16, codes).float()
    else:
        logical = codes.float() - 8
    weight = (logical * raw).to(dtype)
    if not torch.isfinite(weight).all():
        raise ValueError("QQQ producer reconstruction overflows its dtype")
    rounded = (weight / raw).round()
    if not torch.isfinite(rounded).all():
        raise ValueError("QQQ producer inverse contains non-finite codes")
    recovered = (rounded + 8).clamp(0, 15) if resolved != width else rounded.clamp(-15, 15).remainder(16)
    if not torch.equal(recovered, codes):
        raise ValueError("QQQ producer dtype cannot preserve the selected codes")
    return weight
