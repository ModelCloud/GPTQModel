# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure a quantized model against its reference on the same token stream.

Metrics declare which captures they read, so the runner can fuse the ones that
agree: PPL needs the NLL sum, KLD/TOP_N/RMSE need sampled logits, and a single
forward per batch produces both - one pass for all four. A metric that needs a
capture no other metric produces would form its own pass; that path exists so
adding a weight-space metric later does not require rewriting the loop.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..utils.logger import setup_logger
from .config import (
    CAPTURE_NLL,
    CAPTURE_SAMPLED_LOGITS,
    KLDConfig,
    MeasureConfig,
    Metric,
    MetricGrades,
    PPLConfig,
    fuse,
    metric_of,
)
from .metrics import (
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)
from .report import MeasurementReport


__all__ = ["MeasurementError", "build_token_rows", "measure_models"]

log = setup_logger()

#: Window used when PPL is not requested but rows must still be sliced.
_DEFAULT_WINDOW = 512

#: Positions kept per batch when a metric needs distributions rather than a scalar.
_DEFAULT_POSITIONS = 64


class MeasurementError(RuntimeError):
    """Raised when a metric grades FAIL and ``strict`` is set."""

    def __init__(self, report: MeasurementReport) -> None:
        self.report = report
        broken = ", ".join(f"{name}={report.values.get(name)}" for name in report.violations)
        super().__init__(f"measurement failed its grades: {broken}")


@dataclass
class _Captures:
    """What the candidate pass needs from the reference model."""

    nll_sum: float = 0.0
    tokens: int = 0
    logits: List[torch.Tensor] = field(default_factory=list)

    def perplexity(self) -> Optional[float]:
        return None if self.tokens == 0 else math.exp(self.nll_sum / self.tokens)


def build_token_rows(
    tokenizer: Any,
    dataset: Sequence[Any],
    *,
    window: int = _DEFAULT_WINDOW,
    stride: int = 0,
    max_rows: int = 4,
) -> List[torch.Tensor]:
    """Turn texts (or token-id rows) into ``[1, tokens]`` windows."""

    if dataset is None:
        raise ValueError("measurement needs `dataset` (texts or token-id rows)")
    rows: List[torch.Tensor] = []
    for sample in dataset:
        if isinstance(sample, str):
            if tokenizer is None:
                raise ValueError("measurement got text but no tokenizer is available")
            encoded = tokenizer(sample, return_tensors="pt")["input_ids"]
        elif torch.is_tensor(sample):
            encoded = sample.reshape(1, -1)
        elif isinstance(sample, (list, tuple)):
            encoded = torch.tensor([int(token) for token in sample], dtype=torch.long).reshape(1, -1)
        else:
            encoded = torch.as_tensor(sample, dtype=torch.long).reshape(1, -1)
        if encoded.shape[1] < 2:
            continue

        if encoded.shape[1] <= window:
            rows.append(encoded)
        else:
            step = window if stride <= 0 else stride
            for start in range(0, encoded.shape[1] - 1, step):
                piece = encoded[:, start:start + window]
                if piece.shape[1] < 2:
                    break
                rows.append(piece)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("measurement dataset produced no usable token rows")
    return rows[:max_rows]


def _sample_positions(length: int, count: int) -> torch.Tensor:
    usable = length - 1
    if usable <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.linspace(0, usable - 1, min(count, usable)).round().long().unique()


def _model_device(model: torch.nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    return torch.device("cpu")


def _logits_of(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None:
        logits = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(logits) or logits.dim() != 3:
        raise TypeError(
            f"model output has no [batch, sequence, vocab] logits: {type(output).__name__}"
        )
    return logits


def _capture(
    model: torch.nn.Module,
    batches: Sequence[torch.Tensor],
    *,
    device: torch.device,
    needs_nll: bool,
    positions: int,
    track_reference: bool,
) -> _Captures:
    """One pass over the batches, collecting what the fused metrics need."""

    captures = _Captures()
    for tokens in batches:
        tokens = tokens.to(device)
        with torch.no_grad():
            logits = _logits_of(model(input_ids=tokens))
        if needs_nll:
            total, count = mean_negative_log_likelihood(logits, tokens)
            captures.nll_sum += total
            captures.tokens += count
        if positions:
            picked = _sample_positions(logits.shape[1], positions)
            if picked.numel() == 0:
                raise ValueError("measurement rows are too short to sample a prediction position")
            captures.logits.append(
                logits[:, picked, :].detach().cpu().to(torch.float16)
            )
    return captures


def measure_models(
    candidate: torch.nn.Module,
    reference: torch.nn.Module,
    token_rows: Sequence[torch.Tensor],
    config: MeasureConfig,
) -> MeasurementReport:
    """Run every configured metric over the same token rows for both models."""

    if reference is None:
        raise ValueError("measure() needs a reference model or checkpoint path")
    if not token_rows:
        raise ValueError("measurement needs at least one token row")

    reference_device = config.device or _model_device(reference)
    candidate_device = config.device or _model_device(candidate)
    batches = list(token_rows)[: max(1, config.max_batches)]

    # `fuse` plans the passes; the captures those passes must produce are the
    # union of what the members read.
    fuse(config.metrics)
    needs = frozenset().union(*[member.capture_needs for member in config.metrics])
    positions_wanted = _DEFAULT_POSITIONS
    for member in config.metrics:
        if isinstance(member, KLDConfig):
            positions_wanted = max(positions_wanted, member.positions)
    if CAPTURE_SAMPLED_LOGITS not in needs:
        positions_wanted = 0

    # One forward per batched pass for each model; fused metrics share it.
    reference_captures = _capture(
        reference,
        batches,
        device=reference_device,
        needs_nll=CAPTURE_NLL in needs,
        positions=positions_wanted,
        track_reference=True,
    )
    candidate_captures = _capture(
        candidate,
        batches,
        device=candidate_device,
        needs_nll=CAPTURE_NLL in needs,
        positions=positions_wanted,
        track_reference=False,
    )

    values: Dict[str, Any] = {}
    reference_values: Dict[str, Any] = {}
    grades: Dict[str, MetricGrades] = {}

    for member in config.metrics:
        name = metric_of(member)
        value = _evaluate(member, name, reference_captures, candidate_captures)
        values[name.value] = value
        if isinstance(member, PPLConfig):
            reference_values[name.value] = reference_captures.perplexity()
        if member.threshold is not None:
            graded = value if not isinstance(value, dict) else min(value.values())
            grades[name.value] = member.threshold.grade_for(float(graded))

    report = MeasurementReport(
        values=values,
        grades=grades,
        reference=reference_values,
        batches=len(batches),
        positions=positions_wanted,
    )
    if config.strict and report.overall is MetricGrades.FAIL:
        raise MeasurementError(report)
    return report


def _evaluate(
    member: Any,
    name: Metric,
    reference_captures: _Captures,
    candidate_captures: _Captures,
) -> Any:
    """Compute one metric from the shared captures."""

    if name is Metric.PPL:
        if candidate_captures.tokens == 0:
            raise ValueError("measurement produced no scored tokens")
        return math.exp(candidate_captures.nll_sum / candidate_captures.tokens)

    pairs = list(zip(reference_captures.logits, candidate_captures.logits))
    if not pairs:
        raise ValueError(f"{name.value} needs sampled logits but none were captured")

    if name is Metric.KLD:
        return _mean(
            kl_divergence(reference.float(), candidate.float(), temperature=member.temperature)
            for reference, candidate in pairs
        )
    if name is Metric.RMSE:
        return _mean(
            logit_rmse(reference.float(), candidate.float(), normalize=member.normalize)
            for reference, candidate in pairs
        )
    if name is Metric.TOP_N:
        return {
            k: _mean(topk_overlap(reference.float(), candidate.float(), k=k) for reference, candidate in pairs)
            for k in member.top_k
        }
    raise TypeError(f"unknown metric config {type(member).__name__}")


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        raise ValueError("no measurements were collected")
    return sum(collected) / len(collected)
