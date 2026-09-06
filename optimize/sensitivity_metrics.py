# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Streaming FP64 diagnostics. Numerical drift is not task accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ErrorStats:
    count: int = 0
    reference_sq: float = 0.0
    error_sq: float = 0.0
    absolute_sum: float = 0.0
    max_abs: float = 0.0
    finite: bool = True

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError("Reference and candidate shapes differ")
        # Bound temporary double tensors even for large projection outputs.
        left = reference.detach().reshape(-1)
        right = candidate.detach().reshape(-1)
        for start in range(0, left.numel(), 65536):
            ref = left[start : start + 65536].to(device="cpu", dtype=torch.float64)
            got = right[start : start + 65536].to(device="cpu", dtype=torch.float64)
            self.count += ref.numel()
            if not bool(torch.isfinite(ref).all() and torch.isfinite(got).all()):
                self.finite = False
                continue
            delta = got - ref
            self.reference_sq += ref.square().sum().item()
            self.error_sq += delta.square().sum().item()
            self.absolute_sum += delta.abs().sum().item()
            self.max_abs = max(self.max_abs, delta.abs().max().item())

    def report(self) -> dict:
        finite = self.finite and all(
            math.isfinite(v) for v in (self.reference_sq, self.error_sq, self.absolute_sum, self.max_abs)
        )
        valid = self.count > 0 and finite
        relative = math.sqrt(self.error_sq) / math.sqrt(self.reference_sq) if valid and self.reference_sq > 0 else None
        return {
            "count": self.count,
            "finite": finite,
            "max_abs": self.max_abs if valid else None,
            "mean_abs": self.absolute_sum / self.count if valid else None,
            "rmse": math.sqrt(self.error_sq / self.count) if valid else None,
            "relative_l2": relative if relative is None or math.isfinite(relative) else None,
        }


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Inputs are selected positions x full vocabulary; no logit centering."""
    if reference.ndim != 2 or reference.shape[-1] < 2 or reference.shape[0] == 0:
        raise ValueError("Expected nonempty positions x vocabulary logits with vocabulary >= 2")
    stats = ErrorStats()
    stats.update(reference, candidate)
    report = stats.report()
    report.update(kl_mean=None, kl_max=None, top1_agreement=None, reference_margin_min=None)
    if not report["finite"]:
        return report
    kl_sum, kl_max, agreements, margin_min = 0.0, 0.0, 0, math.inf
    for start in range(0, reference.shape[0], 32):
        ref = reference[start : start + 32].to(device="cpu", dtype=torch.float64)
        got = candidate[start : start + 32].to(device="cpu", dtype=torch.float64)
        log_p, log_q = ref.log_softmax(-1), got.log_softmax(-1)
        # Remove only FP64 roundoff below zero in KL, never tensor error.
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).clamp_min(0)
        kl_sum += kl.sum().item()
        kl_max = max(kl_max, kl.max().item())
        agreements += int((ref.argmax(-1) == got.argmax(-1)).sum())
        top = ref.topk(2, dim=-1).values
        margin_min = min(margin_min, (top[:, 0] - top[:, 1]).min().item())
    report.update(
        kl_mean=kl_sum / reference.shape[0],
        kl_max=kl_max,
        top1_agreement=agreements / reference.shape[0],
        reference_margin_min=margin_min,
    )
    return report


def measured_gain(final_error: float | None, epsilons: list[float | None], noise: float) -> tuple:
    """A multi-module gain is descriptive; undefined local norms remain undefined."""
    if final_error is None or not epsilons or any(e is None for e in epsilons):
        return None, "undefined_norm_or_unobserved"
    local = math.hypot(*epsilons)
    if local == 0:
        return None, "no_resolved_local_error"
    if noise > 0 and final_error <= noise:
        return None, "baseline_noise"
    gain = final_error / local
    return (gain, "measured") if math.isfinite(gain) and math.isfinite(local) else (None, "unresolved_ratio")


def uncorrelated_prediction(errors: list[float | None]) -> float | None:
    if not errors or any(e is None for e in errors):
        return None
    norm = math.hypot(*errors)
    return norm if math.isfinite(norm) else None
