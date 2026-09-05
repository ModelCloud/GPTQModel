"""Numerical definitions shared by the twenty experiments (no model acceptance claims)."""

from __future__ import annotations

import torch


def layer_metrics(candidate: torch.Tensor, teacher: torch.Tensor) -> dict:
    """Aggregate one operator case; callers must mask padding before passing tensors."""
    if candidate.shape != teacher.shape or candidate.numel() == 0:
        raise ValueError("Candidate and teacher must have identical nonempty shapes")
    a = candidate.detach().to(dtype=torch.float64)
    b = teacher.detach().to(device=a.device, dtype=torch.float64)
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("Non-finite candidate or teacher")
    delta = a - b
    norm_a, norm_b = a.norm().item(), b.norm().item()
    mae, maximum = delta.abs().mean().item(), delta.abs().max().item()
    return {
        "elements": a.numel(),
        "mean_abs": mae,
        "max_abs": maximum,
        "relative_l2": delta.norm().item() / norm_b if norm_b else None,
        "cosine": (a.flatten() @ b.flatten()).item() / (norm_a * norm_b)
        if norm_a and norm_b
        else None,
        "reference_zero_norm": norm_b == 0,
        "equal_values": torch.equal(candidate, teacher),
        "local_tolerance_pass": mae <= 0.002 and maximum <= 0.046875,
    }


def logits_metrics(candidate: torch.Tensor, teacher: torch.Tensor) -> dict:
    """Mean teacher-to-candidate KL and set-overlap/k, over valid token rows."""
    if (
        candidate.shape != teacher.shape
        or candidate.ndim != 2
        or min(candidate.shape) == 0
    ):
        raise ValueError("Expected matching nonempty [tokens, vocabulary] logits")
    if candidate.shape[1] < 10:
        raise ValueError("Vocabulary must have at least ten entries")
    a = candidate.detach().double()
    b = teacher.detach().to(device=a.device, dtype=torch.float64)
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("Non-finite logits")
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    result = {
        "tokens": a.shape[0],
        "kl_teacher_candidate": (log_b.exp() * (log_b - log_a)).sum(-1).mean().item(),
    }
    # Stable sorting defines ties by ascending vocabulary index for both arms.
    order_a = torch.argsort(a, dim=-1, descending=True, stable=True)[:, :10]
    order_b = torch.argsort(b, dim=-1, descending=True, stable=True)[:, :10]
    for k in (1, 5, 10):
        ka, kb = order_a[:, :k], order_b[:, :k]
        overlap = (ka[:, :, None] == kb[:, None, :]).any(-1).double().mean().item()
        result[f"top{k}_agreement"] = overlap
    return result


def effective_bpw(storage_bytes: dict[str, int], logical_weights: int) -> dict:
    """Every shared object must occur exactly once in the caller's storage inventory."""
    if logical_weights <= 0 or not storage_bytes:
        raise ValueError(
            "Positive logical weight count and complete storage inventory required"
        )
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0
        for v in storage_bytes.values()
    ):
        raise ValueError("Storage counts must be nonnegative integer bytes")
    total = sum(storage_bytes.values())
    return {
        "components_bytes": storage_bytes,
        "total_bytes": total,
        "logical_weights": logical_weights,
        "effective_bpw": 8 * total / logical_weights,
    }
