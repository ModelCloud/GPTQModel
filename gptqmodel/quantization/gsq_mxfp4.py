"""Exact MXFP4 candidate preparation; optimizer/lifecycle binding is separate."""

import torch

from ..utils.mxfp4_cpu import FP4_TABLE


def mxfp4_candidates(qweight, scales, count=3):
    """Return logical nibble candidates and FP32 decoded weights, baseline first.

    Stored low nibbles own even columns; high nibbles own odd columns. E8M0
    scales are shared by 32 values, frozen, and never treated as affine zeros.
    Candidate zero retains signed zero; alternatives use adjacent numeric levels.
    """
    if (qweight.ndim != 2 or qweight.dtype != torch.uint8 or min(qweight.shape) == 0
            or qweight.shape[1] % 16):
        raise ValueError('MXFP4 GSQ requires nonempty uint8 [out,in/2] with in divisible by 32')
    rows, half = qweight.shape
    if (scales.dtype != torch.uint8 or scales.shape != (rows, half//16)
            or scales.device != qweight.device or (scales == 255).any()):
        raise ValueError('MXFP4 GSQ requires matching finite E8M0 block scales')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
        raise ValueError('MXFP4 GSQ candidate count must be in [1,16]')
    table = FP4_TABLE.to(qweight.device)
    baseline = torch.stack((qweight & 15, qweight >> 4), dim=-1).reshape(rows, half*2)
    order = table.argsort(stable=True)
    values = table[order]
    unique = torch.ones_like(values, dtype=torch.bool)
    unique[1:] = values[1:] != values[:-1]
    order, values = order[unique], values[unique]
    center = torch.searchsorted(values, table[baseline.long()].contiguous())
    codes = [baseline]
    for index in range(1, count):
        offset = ((index+1)//2)*(-1 if index % 2 else 1)
        codes.append(order[(center+offset).clamp(0, len(values)-1)].to(torch.uint8))
    codes = torch.stack(codes)
    expanded = torch.exp2(scales.float()-127).repeat_interleave(32, dim=1)
    decoded = table[codes.long()]*expanded.unsqueeze(0)
    if not torch.isfinite(decoded).all():
        raise ValueError('MXFP4 GSQ decoded candidates overflow FP32')
    return codes, decoded


def pack_mxfp4_codes(codes):
    """Pack logical four-bit assignments into the existing MXFP4 byte layout."""
    if (codes.dtype != torch.uint8 or codes.ndim != 2 or min(codes.shape) == 0
            or codes.shape[1] % 32 or (codes > 15).any()):
        raise ValueError('MXFP4 GSQ requires uint8 [out,in] nibble codes with in divisible by 32')
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def refine_mxfp4_weight(weight, scales, *, target, config=None, inputs=None, hessian=None):
    """Fit finite MXFP4 nibble choices with frozen E8M0 scales and a hard guard.

    Optional input activations or their PSD Gram matrix define output
    reconstruction; without either the objective is weight reconstruction. This is GSQ-inspired categorical Adam,
    not the paper's full staged optimizer. Public lifecycle binding is separate.
    """
    from .config import normalize_gsq_config

    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return dict(weight=weight.clone(), scales=scales.clone(), before=None, after=None, history=[])
    if config.learn_scales:
        raise ValueError('MXFP4 GSQ scale learning is not implemented')
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError('MXFP4 GSQ requires nonempty rank-2 weights')
    if (target.shape != (weight.shape[0], weight.shape[1]*2) or not target.is_floating_point() or target.device != weight.device
            or not torch.isfinite(target).all()):
        raise ValueError('MXFP4 GSQ requires a finite floating teacher matching the weight shape and device')
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]*2 or inputs.shape[0] == 0
                               or inputs.device != weight.device or not inputs.is_floating_point()
                               or not torch.isfinite(inputs).all()):
        raise ValueError('MXFP4 GSQ inputs must be finite nonempty [tokens,in] on the weight device')
    if hessian is not None:
        if inputs is not None:
            raise ValueError('MXFP4 GSQ accepts inputs or Hessian, not both')
        if (hessian.shape != (weight.shape[1]*2, weight.shape[1]*2) or not hessian.is_floating_point()
                or hessian.device != weight.device or not torch.isfinite(hessian).all()):
            raise ValueError('MXFP4 GSQ Hessian must be finite floating [in,in] on the weight device')
        if not torch.allclose(hessian, hessian.T, rtol=1e-5, atol=1e-7):
            raise ValueError('MXFP4 GSQ Hessian must be symmetric')
    count = min(config.candidates, 16)
    if count * weight.numel() * 8 > config.max_candidate_bytes:
        raise ValueError('MXFP4 GSQ decoded candidates exceed max_candidate_bytes')
    with torch.inference_mode(False), torch.enable_grad():
        payloads, values = mxfp4_candidates(weight.detach().clone(), scales.detach().clone(), count=count)
        teacher = target.detach().float().clone()
        features = None if inputs is None else inputs.detach().float().T.contiguous().clone()
        if hessian is not None:
            from .gsq_scalar import _metric_factor

            metric = hessian.detach().float().clone()
            if not torch.isfinite(metric).all():
                raise ValueError('MXFP4 GSQ Hessian overflows FP32')
            features = _metric_factor(metric)

        def project(x):
            return x if features is None else x @ features

        energy = project(teacher).square().sum()
        if not torch.isfinite(energy):
            raise ValueError('MXFP4 GSQ teacher energy overflows FP32')
        denominator = torch.where(energy > torch.finfo(torch.float32).eps, energy, torch.ones_like(energy))

        def loss(x):
            return project(x-teacher).square().sum() / denominator

        before = float(loss(values[0]))
        if not torch.isfinite(torch.tensor(before)):
            raise ValueError('MXFP4 GSQ baseline objective is nonfinite')
        best, best_payload = before, payloads[0].clone()
        history = [before]
        if before == 0:
            return dict(weight=pack_mxfp4_codes(best_payload), scales=scales.clone(),
                        before=before, after=best, history=history)
        offsets = torch.tensor([0] + [((i+1)//2)*(-1 if i % 2 else 1) for i in range(1, count)],
                               device=weight.device, dtype=torch.float32)
        logits = (-.5 * offsets.square())[:, None, None].expand_as(values).clone().requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
        rng = torch.Generator(device=weight.device).manual_seed(config.seed)
        for step in range(config.steps):
            tau = config.temperature_start * (config.temperature_end/config.temperature_start) ** (
                step/max(config.steps-1, 1))
            uniform = torch.rand(logits.shape, generator=rng, device=weight.device).clamp_(1e-6, 1-1e-6)
            noise = -(-uniform.log()).log()
            probabilities = ((logits + noise)/tau).softmax(0)
            objective = loss((probabilities * values).sum(0))
            if not torch.isfinite(objective):
                raise ValueError('MXFP4 GSQ relaxed objective is nonfinite')
            optimizer.zero_grad()
            objective.backward()
            if not torch.isfinite(logits.grad).all():
                raise ValueError('MXFP4 GSQ gradient is nonfinite')
            optimizer.step()
            with torch.no_grad():
                selected = logits.argmax(0, keepdim=True)
                hard = values.gather(0, selected).squeeze(0)
                score = float(loss(hard))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError('MXFP4 GSQ hard objective is nonfinite')
                if score < best:
                    best = score
                    best_payload = payloads.gather(0, selected).squeeze(0).clone()
                history.append(best)
        return dict(weight=pack_mxfp4_codes(best_payload), scales=scales.clone(),
                    before=before, after=best, history=history)
