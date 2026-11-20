# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Divergence metrics between a quantized model and its reference.

Everything here operates on tensors: next-token logits and token ids. No model,
tokenizer or dataset is involved, which is what makes the arithmetic testable on
its own - the identity case (a model compared with itself) must produce exact
zeros, and a test that cannot express that is not measuring what it claims.

The four metrics answer "did quantization change the computation", not "is the
model good". End-to-end quality is a benchmark's job (Evalution); these are the
fast numbers that catch a kernel or packing change moving the math.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


__all__ = [
    "mean_negative_log_likelihood",
    "kl_divergence",
    "topk_overlap",
    "logit_rmse",
]


def _require_logits(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() != 3:
        raise ValueError(f"{name} must be [batch, sequence, vocab], got {tuple(tensor.shape)}")


def _shifted(logits: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align next-token logits with the tokens they predict.

    ``logits[:, i]`` predicts ``input_ids[:, i + 1]``, so both sides drop one
    position. Doing the shift here means every metric compares the same pairs.
    """

    if logits.dim() != 3:
        raise ValueError(f"logits must be [batch, sequence, vocab], got {tuple(logits.shape)}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} and input_ids {tuple(input_ids.shape)} disagree"
        )
    if logits.shape[1] < 2:
        raise ValueError("need at least two positions to compare a next-token prediction")

    return logits[:, :-1, :], input_ids[:, 1:]


def mean_negative_log_likelihood(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    ignore_index: Optional[int] = None,
) -> Tuple[float, int]:
    """``(sum of -log p(token), number of scored tokens)`` over next-token pairs.

    Returned as a sum and a count rather than a mean so several batches can be
    accumulated without weighting mistakes.
    """

    shifted_logits, targets = _shifted(logits, input_ids)
    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    gather_targets = targets
    if ignore_index is not None:
        # Gather on a clamped copy: `ignore_index` is usually -100, which is a
        # valid-looking index to `gather` and would either wrap or raise.
        gather_targets = targets.masked_fill(targets == ignore_index, 0)
    picked = log_probs.gather(-1, gather_targets.unsqueeze(-1)).squeeze(-1)

    if ignore_index is not None:
        picked = picked[targets != ignore_index]
    if picked.numel() == 0:
        return 0.0, 0
    return float(-picked.sum()), int(picked.numel())


def kl_divergence(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> float:
    """Mean ``KL(reference ‖ candidate)`` over the leading positions.

    Both sides are already aligned (same token stream), so this is a pure
    distribution comparison. ``log_softmax`` on both sides keeps it finite when
    a candidate probability underflows.
    """

    _require_logits(reference_logits, "reference_logits")
    _require_logits(candidate_logits, "candidate_logits")
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"logit shapes differ: {tuple(reference_logits.shape)} vs "
            f"{tuple(candidate_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    reference = F.log_softmax(reference_logits.float() / temperature, dim=-1)
    candidate = F.log_softmax(candidate_logits.float() / temperature, dim=-1)
    per_position = (reference.exp() * (reference - candidate)).sum(dim=-1)
    return float(per_position.mean())


def topk_overlap(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    k: int,
) -> float:
    """Mean ``|top_k(reference) ∩ top_k(candidate)| / k`` over positions.

    Set overlap rather than ordered equality: near-tied logits flip ordering
    under a rounding change that does not move the model, and for ``k=1`` this
    is exactly top-1 accuracy.
    """

    _require_logits(reference_logits, "reference_logits")
    _require_logits(candidate_logits, "candidate_logits")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    vocab = reference_logits.shape[-1]
    if k > vocab:
        raise ValueError(f"k={k} exceeds the vocabulary size {vocab}")
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"logit shapes differ: {tuple(reference_logits.shape)} vs "
            f"{tuple(candidate_logits.shape)}"
        )

    reference_top = reference_logits.topk(k, dim=-1).indices
    candidate_top = candidate_logits.topk(k, dim=-1).indices
    matches = (reference_top.unsqueeze(-1) == candidate_top.unsqueeze(-2)).any(-1)
    return float(matches.float().sum(dim=-1).div(k).mean())


def logit_rmse(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    normalize: bool = False,
) -> float:
    """Root-mean-square logit difference, optionally normalised by the reference RMS."""

    _require_logits(reference_logits, "reference_logits")
    _require_logits(candidate_logits, "candidate_logits")
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"logit shapes differ: {tuple(reference_logits.shape)} vs "
            f"{tuple(candidate_logits.shape)}"
        )
    difference = (reference_logits.float() - candidate_logits.float()).pow(2).mean().sqrt()
    if not normalize:
        return float(difference)

    reference_scale = reference_logits.float().pow(2).mean().sqrt()
    if reference_scale == 0:
        raise ValueError("cannot normalise by a reference logit RMS of zero")
    return float(difference / reference_scale)
