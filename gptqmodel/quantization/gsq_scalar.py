"""GSQ-inspired refinement on an affine scalar grid, with export-aware scoring.

The integer assignments and optional group scales are trainable; zero-points,
group ownership and the caller's packing convention remain fixed. This module
does not imply that every scalar checkpoint format has a lifecycle adapter.
"""

from dataclasses import dataclass

import pcre
import torch

from ..utils.gemv import awq_gemv_codes
from .config import normalize_gsq_config


@dataclass
class ScalarGSQResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor
    before: float | None
    after: float | None
    history: list[float]


def asymmetric_error_term(weight_error, teacher, cross_moment, alpha=1.0):
    """Linear term for native-versus-propagated reconstruction (unnormalized).

    With feature-by-token X, D = (X_native-X) X.T and E = Wq-W,
    ||Wq X - W (X + alpha*(X_native-X))||² differs from
    ||E X||² - 2*alpha*<E, W D> only by a candidate-independent constant.
    H and D must use the same sample normalization at the caller boundary.
    Keeping this term explicit avoids an inverse/pseudoinverse of H.
    """
    if weight_error.ndim != 2 or teacher.shape != weight_error.shape:
        raise ValueError("asymmetric GSQ requires matching [out,in] errors and teachers")
    width = teacher.shape[1]
    if cross_moment.shape != (width, width):
        raise ValueError("asymmetric GSQ cross moment must be [in,in]")
    if any(t.device != teacher.device for t in (weight_error, cross_moment)):
        raise ValueError("asymmetric GSQ tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all()
               for t in (weight_error, teacher, cross_moment)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("asymmetric GSQ requires finite floating inputs and alpha")
    return -2 * alpha * (weight_error * (teacher @ cross_moment)).sum()


def gsq_enabled_for(config, module_name):
    config = normalize_gsq_config(config)
    return config is not None and config.enabled and (
        config.modules is None or any(pcre.search(pattern, module_name) for pattern in config.modules))


def affine_codes(weight, scales, zeros, g_idx, bits, *, packing, scale_dtype=torch.float16):
    """Match the producer-to-packer arithmetic before saturation and conversion."""
    scale = scales[:, g_idx].float()
    zero = zeros[:, g_idx].float()
    if packing == "gptq":
        values = (weight.float() + zero * scale) / scale
    elif packing == "awq_gemm":
        # GEMM casts its scale table before reconstructing codes and performs
        # the addition/division in the weight/scale promoted dtype, not FP32.
        stored = scales.to(scale_dtype)[:, g_idx]
        offset = (zeros[:, g_idx] * stored).to(weight.dtype)
        values = (weight + offset) / stored
    elif packing in ("awq_gemv", "awq_gemv_fast"):
        if bits != 4 or scale_dtype != torch.float16:
            raise ValueError("AWQ GEMV requires int4 codes and FP16 stored scales")
        return awq_gemv_codes(weight, scales, zeros, g_idx)
    else:
        raise ValueError("scalar GSQ packing must be gptq, awq_gemm, awq_gemv or awq_gemv_fast")
    return values.round().clamp(0, 2**bits - 1)


def _metric_factor(hessian):
    """Factor a calibration Gram without adding an unrequested damping term."""
    h = (hessian.float() + hessian.float().T) * 0.5
    factor, info = torch.linalg.cholesky_ex(h)
    if int(info) == 0:
        return factor
    values, vectors = torch.linalg.eigh(h.double())
    tolerance = torch.finfo(torch.float32).eps * h.shape[0] * values.abs().max()
    if values.min() < -tolerance:
        raise ValueError("scalar GSQ requires a positive-semidefinite calibration Hessian")
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(0)).float()


def refine_affine_scalar(
    weight, scales, zeros, g_idx, *, target, bits, config=None,
    hessian=None, inputs=None, packing="gptq", scale_dtype=torch.float16,
    cross_moment=None, cross_alpha=1.0,
):
    """Refine [out,in] weights and [out,groups] scales on their actual grid.

    ``g_idx`` maps original columns to groups, including GPTQ activation-order
    permutations. Supply either real activations or their Hessian; neither means
    weight reconstruction (the explicit calibration-free RTN objective).
    Hard checkpoints are scored after weight casting, code reconstruction by the
    selected packer convention, and scale casting to the checkpoint dtype.
    The original baseline tensors are returned unchanged unless that score improves.
    An optional native-minus-current cross moment adds the asymmetric linear
    term. Scores then omit a candidate-independent constant and can be negative;
    they are not absolute normalized reconstruction errors.
    """
    config = normalize_gsq_config(config)
    if config is None or not config.enabled:
        return ScalarGSQResult(weight.clone(), scales.clone(), zeros.clone(), g_idx.clone(), None, None, [])
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError("scalar GSQ bits must be an integer in [1,8]")
    if weight.ndim != 2 or target.shape != weight.shape or not weight.is_floating_point():
        raise ValueError("scalar GSQ requires floating [out,in] weights and a matching teacher")
    n, k = weight.shape
    if scales.ndim != 2 or scales.shape[0] != n or zeros.shape != scales.shape:
        raise ValueError("scalar GSQ scales/zeros must have shape [out,groups]")
    if g_idx.shape != (k,) or g_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("scalar GSQ g_idx must be an integer vector of input width")
    if g_idx.min() < 0 or g_idx.max() >= scales.shape[1]:
        raise ValueError("scalar GSQ group index is outside the scale table")
    tensors = (weight, target, scales, zeros, g_idx)
    if any(t.device != weight.device for t in tensors):
        raise ValueError("scalar GSQ tensors must share a device")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in tensors[:-1]):
        raise ValueError("scalar GSQ weights, target, scales and zeros must be finite floating point")
    if (scales <= 0).any() or (zeros != zeros.round()).any() or (zeros < 0).any() or (zeros > 2**bits-1).any():
        raise ValueError("scalar GSQ requires positive scales and in-range integer zero-points")
    if hessian is not None and inputs is not None:
        raise ValueError("supply inputs or hessian, not both")
    if hessian is not None and (hessian.shape not in ((k, k), (k,)) or not torch.isfinite(hessian).all()):
        raise ValueError("scalar GSQ Hessian must be finite [in,in] or diagonal [in]")
    if hessian is not None and hessian.ndim == 1 and (hessian < 0).any():
        raise ValueError("scalar GSQ diagonal Hessian must be nonnegative")
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[1] != k or inputs.shape[0] == 0
                               or not torch.isfinite(inputs).all()):
        raise ValueError("scalar GSQ inputs must be finite nonempty [tokens,in]")
    count = min(2**bits, config.candidates)
    if count * weight.numel() * 4 > config.max_candidate_bytes:
        raise ValueError("scalar GSQ decoded candidates exceed max_candidate_bytes")

    with torch.inference_mode(False), torch.enable_grad():
        teacher = target.detach().float().clone()
        groups = g_idx.detach().long().clone()
        base_scales = scales.detach().float().clone()
        zero = zeros.detach().float().clone()
        baseline = weight.detach().clone()
        factor = None
        diagonal = None
        if hessian is not None:
            h = hessian.detach().to(device=weight.device, dtype=torch.float32).clone()
            if h.ndim == 1:
                diagonal = h.sqrt()
            else:
                factor = _metric_factor(h)
        elif inputs is not None:
            factor = inputs.detach().to(device=weight.device, dtype=torch.float32).T.contiguous().clone()
        def project(matrix):
            if diagonal is not None:
                return matrix * diagonal
            return matrix if factor is None else matrix @ factor

        teacher_output = project(teacher)
        energy = teacher_output.square().sum()
        # Near-zero teacher energy is not a useful normalization scale. Use
        # the unnormalized quadratic, retaining any asymmetric linear term.
        denominator = torch.where(energy > torch.finfo(torch.float32).eps, energy, torch.ones_like(energy))
        correction = None
        if cross_moment is not None:
            if hessian is None and inputs is None:
                raise ValueError("asymmetric GSQ requires matching calibration moments or inputs")
            cross = cross_moment.detach().to(device=weight.device, dtype=torch.float32).clone()
            asymmetric_error_term(torch.zeros_like(teacher), teacher, cross, cross_alpha)
            correction = cross_alpha * (teacher @ cross)

        def loss(matrix):
            error = project(matrix - teacher)
            value = error.square().sum()
            if correction is not None:
                value = value - 2 * ((matrix-teacher) * correction).sum()
            return value / denominator

        gemv = packing in ("awq_gemv", "awq_gemv_fast")

        def recover_codes(candidate_weight, candidate_scales):
            source_scales = candidate_scales.to(scales.dtype) if gemv else candidate_scales
            source_zeros = zero.to(zeros.dtype) if gemv else zero
            values = affine_codes(candidate_weight, source_scales, source_zeros, groups, bits,
                                  packing=packing, scale_dtype=scale_dtype)
            # Packing arithmetic keeps its native dtype above. Optimization
            # needs FP32: Adam's epsilon underflows in FP16 for local grids.
            return values.float()

        def decode(codes, candidate_scales):
            stored_scales = candidate_scales.to(scale_dtype).float()
            if not torch.isfinite(stored_scales).all() or (stored_scales <= 0).any():
                raise ValueError("scalar GSQ scales are not finite positive in the checkpoint dtype")
            if packing == "awq_gemv_fast":
                offset = -(stored_scales * zero).half().float()
                if not torch.isfinite(offset).all():
                    raise ValueError("scalar GSQ GEMV_FAST offsets must be finite in FP16 storage")
                return stored_scales[:, groups] * codes + offset[:, groups]
            return stored_scales[:, groups] * (codes - zero[:, groups])

        def exported(candidate_weight, candidate_scales):
            source_scales = candidate_scales.to(scales.dtype) if gemv else candidate_scales
            return decode(recover_codes(candidate_weight, source_scales), source_scales)

        base_codes = recover_codes(baseline, base_scales)
        if count == 2**bits:
            codes = torch.arange(count, device=weight.device).float().expand(n, k, count)
            valid = torch.ones((n, k, count), device=weight.device, dtype=torch.bool)
        else:
            offsets = torch.tensor([0] + [((-1)**i) * ((i+1)//2) for i in range(1, count)], device=weight.device)
            codes = base_codes.unsqueeze(-1) + offsets
            valid = (codes >= 0) & (codes < 2**bits)
            codes = codes.clamp(0, 2**bits-1)
        # A scalar grid has a distance structure: distant codes must not start
        # as equally plausible alternatives to adjacent rounding choices. Use
        # the author's quadratic-distance prior, without its random initial
        # perturbation (sampling below already uses the private seeded RNG).
        logits = (-0.5 * (codes - base_codes.unsqueeze(-1)).square()).requires_grad_()
        scale_delta = torch.zeros_like(base_scales, requires_grad=config.learn_scales)
        parameters = [logits] + ([scale_delta] if config.learn_scales else [])
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        rng = torch.Generator(device=weight.device).manual_seed(config.seed)
        before = float(loss(exported(baseline, base_scales)))
        if not torch.isfinite(torch.tensor(before)):
            raise ValueError("non-finite scalar GSQ baseline objective")
        best, best_weight, best_scales = before, baseline, scales.detach().clone()
        history = [before]
        for step in range(config.steps):
            tau = config.temperature_start * (config.temperature_end / config.temperature_start) ** (
                step / max(config.steps-1, 1))
            uniform = torch.rand(logits.shape, device=weight.device, generator=rng).clamp_(1e-6, 1-1e-6)
            noise = -(-uniform.log()).log()
            probabilities = ((logits.masked_fill(~valid, -torch.inf) + noise) / tau).softmax(-1)
            current_scales = base_scales * scale_delta.exp()
            soft_codes = (probabilities * codes).sum(-1)
            objective = loss(current_scales[:, groups] * (soft_codes - zero[:, groups]))
            if not torch.isfinite(objective):
                raise ValueError("non-finite scalar GSQ relaxed objective")
            optimizer.zero_grad()
            objective.backward()
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("non-finite scalar GSQ gradient")
            optimizer.step()
            with torch.no_grad():
                selected = logits.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
                hard_codes = codes.gather(-1, selected).squeeze(-1)
                hard_scales = (base_scales * scale_delta.exp()).to(scale_dtype).float()
                if gemv:
                    hard_scales = hard_scales.to(scales.dtype)
                candidate = decode(hard_codes, hard_scales).to(weight.dtype)
                score = float(loss(exported(candidate, hard_scales)))
                if not torch.isfinite(torch.tensor(score)):
                    raise ValueError("non-finite scalar GSQ hard objective")
                history.append(score)
                if score < best:
                    best, best_weight, best_scales = score, candidate.clone(), hard_scales.clone()
        return ScalarGSQResult(best_weight.detach().to(weight.dtype), best_scales.detach().to(scales.dtype),
                               zeros.detach().clone(), g_idx.detach().clone(), before, best, history)
