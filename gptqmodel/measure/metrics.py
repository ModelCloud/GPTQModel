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


# Two FP32 tensors exist per NLL tile: the cast logits and their log-softmax.
# The full-vocabulary copies used to scale with batch size and sequence length,
# including padded positions, so one long row could add hundreds of MiB of
# scratch for a single scalar per target. Bound the pair by a byte budget and
# accumulate the sums on the logits' device instead.
_NLL_FP32_TILE_BYTES = 32 * 1024 * 1024


def _nll_tile_rows(vocab: int) -> int:
    """Rows per NLL tile, bounded by the two FP32 temporaries it creates."""

    return max(1, _NLL_FP32_TILE_BYTES // (2 * vocab * 4))


def mean_negative_log_likelihood(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    ignore_index: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    score_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, int]:
    """``(sum of -log p(token), number of scored tokens)`` over next-token pairs.

    Returned as a sum and a count rather than a mean so several batches can be
    accumulated without weighting mistakes.

    ``attention_mask`` marks real tokens so padding is not scored. ``score_mask``
    is the window's own coverage mask: position ``j`` is True when ``j`` is a
    target this window is responsible for. Overlapping windows pass their mask
    so a target shared as context by a later window is not counted twice.

    The FP32 work runs in row tiles, so a long sequence or a large batch cannot
    allocate a full-vocabulary copy of every logit at once. Per-tile sums and
    valid-token counts accumulate on the logits' device and are copied back once.
    """

    shifted_logits, targets = _shifted(logits, input_ids)
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"input_ids {tuple(input_ids.shape)}"
        )
    if score_mask is not None and score_mask.shape != input_ids.shape:
        raise ValueError(
            f"score_mask {tuple(score_mask.shape)} does not match "
            f"input_ids {tuple(input_ids.shape)}"
        )

    batch, sequence, vocab = shifted_logits.shape
    if batch == 0:
        return 0.0, 0
    row_budget = _nll_tile_rows(vocab)
    # Split the row budget across the batch so one tile never exceeds it.
    batch_step = min(batch, row_budget)
    sequence_step = max(1, row_budget // batch_step)

    total = torch.zeros((), dtype=torch.float32, device=shifted_logits.device)
    count = torch.zeros((), dtype=torch.long, device=shifted_logits.device)

    for batch_start in range(0, batch, batch_step):
        batch_end = min(batch_start + batch_step, batch)
        for sequence_start in range(0, sequence, sequence_step):
            sequence_end = min(sequence_start + sequence_step, sequence)
            tile_logits = shifted_logits[
                batch_start:batch_end, sequence_start:sequence_end, :
            ].float()
            tile_targets = targets[batch_start:batch_end, sequence_start:sequence_end]

            keep = torch.ones_like(tile_targets, dtype=torch.bool)
            if ignore_index is not None:
                keep &= tile_targets != ignore_index
            if attention_mask is not None:
                # A padded position predicts nothing: its target is a pad id, and
                # counting it would drag the perplexity toward whatever pad happens to
                # look like.
                keep &= attention_mask[
                    batch_start:batch_end, 1 + sequence_start:1 + sequence_end
                ].bool()
            if score_mask is not None:
                # A target belongs to exactly one window. Windows overlap for
                # context, not to score the same transition again.
                keep &= score_mask[
                    batch_start:batch_end, 1 + sequence_start:1 + sequence_end
                ].bool()

            # `cross_entropy` takes classes from dim 1, so flatten the tile's
            # batch and sequence rows. Ignored and padded positions share one
            # sentinel and contribute neither a loss nor a count.
            sentinel = ignore_index if ignore_index is not None else -1
            total += F.cross_entropy(
                tile_logits.reshape(-1, vocab),
                tile_targets.masked_fill(~keep, sentinel).reshape(-1),
                reduction="sum",
                ignore_index=sentinel,
            )
            count += keep.sum()

    if count == 0:
        return 0.0, 0
    return float(total), int(count)


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
    """Mean ``|topk(reference) ∩ topk(candidate)| / k`` over positions.

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
