"""Offline-only V4 candidate screening.

The production V4 decoder deliberately has no mapping metadata: its second
pair is selected with ``state ^ 0xA5A5``.  This module evaluates alternative
second-pair permutations for *pruning only*.  It never changes checkpoint
serialization or the runtime decoder.  A candidate must be promoted through
the normal lifecycle and held-out output gates before it can become a format
change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .qvq_codecs.pgc16 import (
    PGC16_STATE_COUNT,
    canonical_pgc16_levels,
    pgc16_mix_states,
    validate_pgc16_levels,
)


@dataclass(frozen=True)
class V4Candidate:
    """An offline V4 mapping and normalized-code scale proposal."""

    name: str
    xor_mask: int
    scale_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("V4 candidate name must not be empty")
        if not isinstance(self.xor_mask, int) or isinstance(self.xor_mask, bool):
            raise TypeError("V4 candidate xor_mask must be an integer")
        if not 0 <= self.xor_mask < PGC16_STATE_COUNT:
            raise ValueError("V4 candidate xor_mask must fit in 16 bits")
        if not isinstance(self.scale_multiplier, (int, float)) or isinstance(self.scale_multiplier, bool):
            raise TypeError("V4 candidate scale_multiplier must be a real scalar")
        if not torch.isfinite(torch.tensor(float(self.scale_multiplier))):
            raise ValueError("V4 candidate scale_multiplier must be finite")
        if self.scale_multiplier <= 0:
            raise ValueError("V4 candidate scale_multiplier must be positive")


@dataclass(frozen=True)
class V4CandidateScore:
    """Local geometry metrics; these are not an end-to-end quality gate."""

    candidate: V4Candidate
    mse: float
    p95: float
    represented_orthants: float


def decode_v4_candidate_states(
    states: torch.Tensor,
    *,
    xor_mask: int,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode a candidate mapping without changing the production decoder."""

    if states.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("V4 candidate states must use an integer dtype")
    if not isinstance(xor_mask, int) or isinstance(xor_mask, bool) or not 0 <= xor_mask < PGC16_STATE_COUNT:
        raise ValueError("V4 candidate xor_mask must fit in 16 bits")
    target_levels = canonical_pgc16_levels().to(device=states.device) if levels is None else levels
    if target_levels.device != states.device:
        raise ValueError("V4 candidate states and levels must be on the same device")
    validate_pgc16_levels(target_levels)
    first = pgc16_mix_states(states)
    second = pgc16_mix_states(states.to(torch.int64) ^ xor_mask)
    indices = torch.stack((first >> 8, first & 0xFF, second >> 8, second & 0xFF), dim=-1)
    return target_levels[indices].contiguous()


def v4_candidate_codebook(
    candidate: V4Candidate,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    levels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize one offline candidate codebook."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("V4 candidate codebook dtype must be floating point")
    target_device = torch.device("cpu" if device is None else device)
    states = torch.arange(PGC16_STATE_COUNT, dtype=torch.int64, device=target_device)
    target_levels = None if levels is None else levels.to(device=target_device)
    return decode_v4_candidate_states(states, xor_mask=candidate.xor_mask, levels=target_levels).to(dtype=dtype)


def score_v4_candidate(
    target_vectors: torch.Tensor,
    candidate: V4Candidate,
    *,
    codebook: torch.Tensor | None = None,
    chunk_size: int = 256,
) -> V4CandidateScore:
    """Score nearest-code local geometry for normalized four-weight vectors.

    ``target_vectors`` is expected to have a final dimension of four.  The
    score is intentionally only a pruning signal; it does not replace live
    prefix or final-logit validation.
    """

    if target_vectors.shape[-1:] != (4,) or not target_vectors.is_floating_point():
        raise ValueError("V4 candidate targets must have final shape (..., 4) and be floating point")
    if not torch.isfinite(target_vectors).all():
        raise ValueError("V4 candidate targets must be finite")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("V4 candidate chunk_size must be a positive integer")
    flat = target_vectors.detach().to(dtype=torch.float32).reshape(-1, 4)
    if flat.numel() == 0:
        raise ValueError("V4 candidate targets must not be empty")
    cb = v4_candidate_codebook(candidate, device=flat.device, dtype=torch.float32) if codebook is None else codebook
    if cb.shape != (PGC16_STATE_COUNT, 4) or cb.device != flat.device:
        raise ValueError("V4 candidate codebook must have shape (65536, 4) on the target device")
    cb = cb * float(candidate.scale_multiplier)
    errors = []
    nearest = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        distances = (chunk.square().sum(dim=1, keepdim=True) + cb.square().sum(dim=1).unsqueeze(0))
        distances.addmm_(chunk, cb.transpose(0, 1), beta=1.0, alpha=-2.0)
        best, indices = distances.min(dim=1)
        errors.append(best.clamp_min(0))
        nearest.append(cb[indices])
    error = torch.cat(errors)
    signs = torch.sign(torch.cat(nearest)).gt(0).to(torch.int64)
    orthant_ids = signs[:, 0] | (signs[:, 1] << 1) | (signs[:, 2] << 2) | (signs[:, 3] << 3)
    represented = float(orthant_ids.unique().numel())
    return V4CandidateScore(candidate, float(error.mean()), float(error.quantile(0.95)), float(represented))


def select_v4_candidate(
    scores: list[V4CandidateScore],
    *,
    baseline: V4Candidate,
    minimum_relative_mse_improvement: float = 0.0,
) -> V4Candidate:
    """Select only a finite local winner; caller must run held-out output gates."""

    if not scores:
        raise ValueError("V4 candidate score list must not be empty")
    if minimum_relative_mse_improvement < 0 or not torch.isfinite(torch.tensor(minimum_relative_mse_improvement)):
        raise ValueError("minimum_relative_mse_improvement must be finite and nonnegative")
    baseline_scores = [score for score in scores if score.candidate == baseline]
    if len(baseline_scores) != 1:
        raise ValueError("scores must contain exactly one baseline candidate")
    base = baseline_scores[0]
    if not torch.isfinite(torch.tensor([base.mse, base.p95, base.represented_orthants])).all():
        raise ValueError("baseline candidate score must be finite")
    eligible = [
        score
        for score in scores
        if torch.isfinite(torch.tensor([score.mse, score.p95, score.represented_orthants])).all()
        and score.mse <= base.mse * (1.0 - minimum_relative_mse_improvement)
        and score.p95 <= base.p95
    ]
    return min(eligible or [base], key=lambda score: (score.mse, score.p95)).candidate


__all__ = [
    "V4Candidate",
    "V4CandidateScore",
    "decode_v4_candidate_states",
    "score_v4_candidate",
    "select_v4_candidate",
    "v4_candidate_codebook",
]
