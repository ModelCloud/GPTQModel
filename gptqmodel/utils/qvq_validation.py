# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Accuracy and argument gates shared by QVQ lifecycle validators."""

import math
from argparse import Namespace

import torch
import torch.nn.functional as F


def _finite_threshold(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def qvq_accuracy_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise AssertionError(f"Logit shape mismatch: dense={reference.shape}, QVQ={candidate.shape}")
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    reference_finite = bool(torch.isfinite(reference).all())
    candidate_finite = bool(torch.isfinite(candidate).all())
    if not reference_finite or not candidate_finite:
        raise AssertionError(
            f"Logits must be finite: reference_finite={reference_finite}, candidate_finite={candidate_finite}"
        )

    error = candidate - reference
    reference64 = reference.flatten().double()
    candidate64 = candidate.flatten().double()
    error64 = error.flatten().double()
    epsilon = torch.finfo(torch.float64).eps
    reference_energy = reference64.square().sum().clamp_min(epsilon)
    error_energy = error64.square().sum()

    reference_log_prob = reference.double().log_softmax(dim=-1)
    candidate_log_prob = candidate.double().log_softmax(dim=-1)
    reference_prob = reference_log_prob.exp()
    candidate_prob = candidate_log_prob.exp()
    midpoint = (reference_prob + candidate_prob) * 0.5
    midpoint_log = midpoint.clamp_min(1e-30).log()
    if torch.equal(reference, candidate):
        forward_kld = reverse_kld = jsd = torch.zeros(
            reference.shape[:-1], dtype=torch.float64, device=reference.device
        )
    else:
        forward_kld = (reference_prob * (reference_log_prob - candidate_log_prob)).sum(dim=-1).clamp_min(0)
        reverse_kld = (candidate_prob * (candidate_log_prob - reference_log_prob)).sum(dim=-1).clamp_min(0)
        jsd = (
            0.5
            * (
                (reference_prob * (reference_log_prob - midpoint_log)).sum(dim=-1)
                + (candidate_prob * (candidate_log_prob - midpoint_log)).sum(dim=-1)
            )
        ).clamp_min(0)
    top5 = min(5, reference.shape[-1])
    reference_top5 = reference.topk(top5, dim=-1).indices
    candidate_top5 = candidate.topk(top5, dim=-1).indices
    top5_overlap = (
        (reference_top5.unsqueeze(-1) == candidate_top5.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    )
    top10 = min(10, reference.shape[-1])
    reference_top10 = reference.topk(top10, dim=-1).indices
    candidate_top10 = candidate.topk(top10, dim=-1).indices
    top10_overlap = (
        (reference_top10.unsqueeze(-1) == candidate_top10.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    )
    return {
        "finite": candidate_finite,
        "mae": float(error.abs().mean().item()),
        "mse": float(F.mse_loss(candidate, reference).item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "relative_l2": float((error_energy / reference_energy).sqrt().item()),
        "sqnr_db": float((10.0 * torch.log10(reference_energy / error_energy.clamp_min(epsilon))).item()),
        "cosine": float(F.cosine_similarity(reference64, candidate64, dim=0).clamp(-1.0, 1.0).item()),
        "forward_kld": float(forward_kld.mean().item()),
        "reverse_kld": float(reverse_kld.mean().item()),
        "jensen_shannon": float(jsd.mean().item()),
        "top1_agreement": float((candidate.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean().item()),
        "top5_overlap": float(top5_overlap.mean().item()),
        "top10_overlap": float(top10_overlap.mean().item()),
        "max_abs": float(error.abs().max().item()),
    }


def assert_qvq_dense_accuracy(metrics: dict[str, float | bool], args: Namespace) -> None:
    max_forward_kld = _finite_threshold(args.max_forward_kld, name="--max-forward-kld")
    min_top1_agreement = _finite_threshold(args.min_top1_agreement, name="--min-top1-agreement")
    if metrics["forward_kld"] > max_forward_kld:
        raise AssertionError(
            f"Forward KLD {metrics['forward_kld']:.8f} exceeds --max-forward-kld={args.max_forward_kld:.8f}"
        )
    if metrics["top1_agreement"] < min_top1_agreement:
        raise AssertionError(
            f"Top-1 agreement {metrics['top1_agreement']:.8f} is below "
            f"--min-top1-agreement={args.min_top1_agreement:.8f}"
        )


def assert_qvq_reload_parity(
    live_logits: torch.Tensor,
    reloaded_logits: torch.Tensor,
    args: Namespace,
) -> dict[str, float | bool]:
    _finite_threshold(args.reload_rtol, name="--reload-rtol")
    _finite_threshold(args.reload_atol, name="--reload-atol")
    metrics = qvq_accuracy_metrics(live_logits, reloaded_logits)
    torch.testing.assert_close(
        reloaded_logits,
        live_logits,
        rtol=args.reload_rtol,
        atol=args.reload_atol,
        msg=f"QVQ live and reloaded logits diverged: {metrics}",
    )
    return metrics


def validate_qvq_lifecycle_args(args: Namespace) -> None:
    if args.rows <= 0 or args.layers <= 0:
        raise ValueError("--rows and --layers must be positive")
    max_forward_kld = _finite_threshold(args.max_forward_kld, name="--max-forward-kld")
    min_top1_agreement = _finite_threshold(args.min_top1_agreement, name="--min-top1-agreement")
    reload_rtol = _finite_threshold(args.reload_rtol, name="--reload-rtol")
    reload_atol = _finite_threshold(args.reload_atol, name="--reload-atol")
    if max_forward_kld < 0:
        raise ValueError("--max-forward-kld must be nonnegative")
    if not 0 <= min_top1_agreement <= 1:
        raise ValueError("--min-top1-agreement must be in [0, 1]")
    if reload_rtol < 0 or reload_atol < 0:
        raise ValueError("reload tolerances must be nonnegative")


__all__ = [
    "assert_qvq_dense_accuracy",
    "assert_qvq_reload_parity",
    "qvq_accuracy_metrics",
    "validate_qvq_lifecycle_args",
]
