# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fidelity metrics for statistically lossless quantization.

Implements the Expected Acceptance Rate (EAR) and top-K restricted KL
 divergence from Section 3.2 of arXiv:2605.02404.
"""

import torch
import torch.nn.functional as F


def _topk_mask(probs: torch.Tensor, top_k: int, dim: int = -1) -> torch.Tensor:
    """Return a boolean mask selecting the top-k entries of ``probs`` along ``dim``."""

    if top_k is None or top_k <= 0 or top_k >= probs.shape[dim]:
        return torch.ones_like(probs, dtype=torch.bool)

    _, top_k_indices = torch.topk(probs, top_k, dim=dim)
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter(dim, top_k_indices, True)
    return mask


def expected_acceptance_rate(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    top_k: int | None = None,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Expected Acceptance Rate (EAR) between two next-token distributions.

    EAR is the average total-variation overlap ``sum(min(p, q))`` over the
    requested token positions. When ``top_k`` is supplied, the sum is restricted
    to the top-K tokens under ``p``, matching the paper's usage. The return value
    is a scalar tensor in ``[0, 1]``.

    Args:
        p: Reference probability distribution. Must be non-negative and sum to 1.
        q: Quantized probability distribution. Must be non-negative and sum to 1.
        top_k: If given, restrict the metric to the top-k tokens under ``p``.
        dim: Dimension along which token probabilities are laid out.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor containing the mean EAR across all positions.
    """

    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    mask = _topk_mask(p, top_k, dim=dim)
    overlap = (torch.min(p, q) * mask).sum(dim=dim)
    p_mass = (p * mask).sum(dim=dim)

    ear = overlap / p_mass.clamp(min=eps)
    return ear.mean()


def kl_divergence_topk(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    top_k: int | None = None,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Top-K restricted KL divergence ``D_KL(p || q)``.

    The sum is restricted to the top-K tokens under ``p`` so that the metric is
    robust to the long tail of low-probability tokens, as described in the paper.

    Args:
        p: Reference probability distribution.
        q: Quantized probability distribution.
        top_k: If given, restrict the sum to the top-k tokens under ``p``.
        dim: Dimension along which token probabilities are laid out.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor containing the mean top-K KL divergence.
    """

    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    mask = _topk_mask(p, top_k, dim=dim)
    p_mass = (p * mask).sum(dim=dim)
    kl = (p * mask * ((p / q).log())).sum(dim=dim)

    return (kl / p_mass.clamp(min=eps)).mean()


def next_token_distributions(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    *,
    temperature: float = 1.0,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert unnormalized logits to next-token distributions.

    Args:
        logits_p: Logits from the reference model.
        logits_q: Logits from the quantized model.
        temperature: Sampling temperature applied before softmax.
        dim: Token dimension.

    Returns:
        Tuple ``(p, q)`` of normalized probability distributions.
    """

    if temperature != 1.0:
        logits_p = logits_p / temperature
        logits_q = logits_q / temperature

    p = F.softmax(logits_p, dim=dim)
    q = F.softmax(logits_q, dim=dim)
    return p, q
