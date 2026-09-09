"""Finite FP8 payload candidates for optional GSQ format-aware refinement.

Candidate preparation and fixed-scale fitting preserve exact storage bytes.
Public configuration and processor export remain separate contracts.
"""

import torch


def fp8_payload_candidates(weight, count=4):
    """Return [candidate,out,in] uint8 payloads and their normalized FP32 values.

    Candidate zero preserves every baseline byte, including signed zero. Other
    candidates alternate lower/upper neighbors on the sorted finite FP8 grid.
    Endpoint clamping may duplicate candidates. Scales are deliberately absent:
    callers must apply the exact stored inverse-scale geometry before fitting.
    """
    supported = tuple(getattr(torch, name) for name in
                      ('float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz')
                      if hasattr(torch, name))
    if weight.dtype not in supported or weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError('FP8 GSQ requires a nonempty rank-2 E4M3/E5M2 weight payload')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
        raise ValueError('FP8 GSQ candidate count must be an integer in [1,256]')
    baseline = weight.float()
    if not torch.isfinite(baseline).all():
        raise ValueError('FP8 GSQ baseline must contain only finite values')
    # Enumerate real storage bytes rather than assuming exponent/mantissa bit
    # semantics are identical across FN/FNUZ variants.
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    decoded = raw.view(weight.dtype).float()
    finite = torch.isfinite(decoded)
    order = decoded[finite].argsort(stable=True)
    values = decoded[finite][order]
    codes = raw[finite][order]
    unique = torch.ones_like(values, dtype=torch.bool)
    unique[1:] = values[1:] != values[:-1]
    values, codes = values[unique].to(weight.device), codes[unique].to(weight.device)
    center = torch.searchsorted(values, baseline.contiguous())
    if not torch.equal(values[center], baseline):
        raise ValueError('FP8 baseline does not lie on its finite storage grid')
    payloads = [weight.contiguous().view(torch.uint8)]
    for index in range(1, count):
        offset = (index + 1) // 2 * (-1 if index % 2 else 1)
        payloads.append(codes[(center + offset).clamp(0, len(values)-1)])
    payloads = torch.stack(payloads)
    return payloads, payloads.view(weight.dtype).float()


def fp8_inverse_scale_grid(weight, scale_inv, *, method='row', block_size=None):
    """Validate and expand inverse scales to the logical [out,in] grid."""
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError('FP8 GSQ scale expansion requires nonempty rank-2 weights')
    if (not scale_inv.is_floating_point() or not torch.isfinite(scale_inv).all()
            or (scale_inv <= 0).any() or scale_inv.device != weight.device):
        raise ValueError('FP8 GSQ inverse scales must be finite positive floating values on the weight device')
    rows, columns = weight.shape
    scales = scale_inv.float()
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('FP8 GSQ inverse scales must remain finite and positive in FP32')
    if method == 'tensor':
        if scales.numel() != 1 or block_size is not None:
            raise ValueError('FP8 tensor scaling requires one scale and no block size')
        return scales.reshape(1, 1).expand(rows, columns)
    if method == 'row':
        if scales.shape != (rows,) or block_size is not None:
            raise ValueError('FP8 row scaling requires one scale per output row and no block size')
        return scales[:, None].expand(rows, columns)
    if method != 'block' or not isinstance(block_size, (tuple, list)) or len(block_size) != 2:
        raise ValueError('FP8 scale method must be tensor, row, or block with two block dimensions')
    br, bc = block_size
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (br, bc)):
        raise ValueError('FP8 block dimensions must be positive integers')
    if rows % br or columns % bc or scales.shape != (rows // br, columns // bc):
        raise ValueError('FP8 block scales do not match the divisible weight geometry')
    return scales.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)


def fp8_decoded_candidates(weight, scale_inv, *, method='row', block_size=None, count=4):
    """Return exact candidate bytes and decoded FP32 weights using frozen scales."""
    scales = fp8_inverse_scale_grid(weight, scale_inv, method=method, block_size=block_size)
    payloads, values = fp8_payload_candidates(weight, count)
    decoded = values / scales.unsqueeze(0)
    if not torch.isfinite(decoded).all():
        raise ValueError('FP8 GSQ decoded candidates overflow FP32')
    return payloads, decoded


def refine_fp8_weight(weight, scale_inv, *, target, config=None, inputs=None, method='row', block_size=None):
    """Fit finite FP8 payload choices with frozen inverse scales and a hard guard.

    Optional input activations define output reconstruction; without them the
    objective is weight reconstruction. This is GSQ-inspired categorical Adam,
    not the paper's full staged optimizer. Public lifecycle binding is separate.
    """
    from .config import normalize_gsq_config

    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return dict(weight=weight.clone(), scale_inv=scale_inv.clone(), before=None, after=None, history=[])
    if config.learn_scales:
        raise ValueError('FP8 GSQ scale learning is not implemented')
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError('FP8 GSQ requires nonempty rank-2 weights')
    if (target.shape != weight.shape or not target.is_floating_point() or target.device != weight.device
            or not torch.isfinite(target).all()):
        raise ValueError('FP8 GSQ requires a finite floating teacher matching the weight shape and device')
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[1] != weight.shape[1] or inputs.shape[0] == 0
                               or inputs.device != weight.device or not inputs.is_floating_point()
                               or not torch.isfinite(inputs).all()):
        raise ValueError('FP8 GSQ inputs must be finite nonempty [tokens,in] on the weight device')
    count = min(config.candidates, 256)
    if count * weight.numel() * 4 > config.max_candidate_bytes:
        raise ValueError('FP8 GSQ decoded candidates exceed max_candidate_bytes')
    with torch.inference_mode(False), torch.enable_grad():
        payloads, values = fp8_decoded_candidates(weight.detach().clone(), scale_inv.detach().clone(),
                                                 method=method, block_size=block_size, count=count)
        teacher = target.detach().float().clone()
        features = None if inputs is None else inputs.detach().float().T.contiguous().clone()

        def project(x):
            return x if features is None else x @ features

        energy = project(teacher).square().sum()
        if not torch.isfinite(energy):
            raise ValueError('FP8 GSQ teacher energy overflows FP32')
        denominator = torch.where(energy > torch.finfo(torch.float32).eps, energy, torch.ones_like(energy))

        def loss(x):
            return project(x-teacher).square().sum() / denominator

        before = float(loss(values[0]))
        if not torch.isfinite(torch.tensor(before)):
            raise ValueError('FP8 GSQ baseline objective is nonfinite')
        best, best_payload = before, payloads[0].clone()
        history = [before]
        if before == 0:
            return dict(weight=best_payload.view(weight.dtype), scale_inv=scale_inv.clone(),
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
                raise ValueError('FP8 GSQ relaxed objective is nonfinite')
            optimizer.zero_grad()
            objective.backward()
            if not torch.isfinite(logits.grad).all():
                raise ValueError('FP8 GSQ gradient is nonfinite')
            optimizer.step()
            with torch.no_grad():
                selected = logits.argmax(0, keepdim=True)
                hard = values.gather(0, selected).squeeze(0)
                score = float(loss(hard))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError('FP8 GSQ hard objective is nonfinite')
                if score < best:
                    best = score
                    best_payload = payloads.gather(0, selected).squeeze(0).clone()
                history.append(best)
        return dict(weight=best_payload.view(weight.dtype), scale_inv=scale_inv.clone(),
                    before=before, after=best, history=history)
