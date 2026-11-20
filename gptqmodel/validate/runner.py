# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare a quantized model with its reference over the same token stream.

The reference pass keeps only what the metrics need:

- ``nll_sum`` / ``tokens`` for PPL,
- fp16 logits at a bounded number of sampled positions for KLD, TOP_N and RMSE.

A 7B model therefore costs tens of megabytes rather than a second copy of the
weights, and every metric sees identical positions in both passes - the shift
that makes these metrics lie is impossible by construction.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..utils.logger import setup_logger
from .config import KLDConfig, Metric, PPLConfig, RMSEConfig, TOP_NConfig, ValidateConfig
from .metrics import (
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)
from .report import ValidationReport


__all__ = ["ValidationError", "build_token_rows", "validate_models"]

log = setup_logger()

#: Window used when PPL is not requested but rows must still be sliced.
_DEFAULT_WINDOW = 512

#: Positions kept per batch when a metric needs distributions rather than a scalar.
_DEFAULT_POSITIONS = 64


class ValidationError(RuntimeError):
    """Raised when ``strict=True`` and a metric broke its threshold."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        broken = ", ".join(f"{name}={value:.4g}" for name, value in report.violations.items())
        super().__init__(f"validation exceeded its thresholds: {broken}")


@dataclass
class _Capture:
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
    """Turn texts (or token-id rows) into ``[1, tokens]`` windows.

    Token-id rows pass through untouched so a caller without a tokenizer - or a
    test - can drive the same code path.
    """

    if dataset is None:
        raise ValueError("validation needs `dataset` (texts or token-id rows)")
    rows: List[torch.Tensor] = []
    for sample in dataset:
        if isinstance(sample, str):
            if tokenizer is None:
                raise ValueError("validation got text but no tokenizer is available")
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
        raise ValueError("validation dataset produced no usable token rows")
    return rows[:max_rows]


def _sample_positions(length: int, count: int) -> torch.Tensor:
    """Deterministic, evenly spaced prediction positions (``0 .. length - 2``)."""

    usable = length - 1
    if usable <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long)
    picks = torch.linspace(0, usable - 1, min(count, usable)).round().long().unique()
    return picks


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


def validate_models(
    candidate: torch.nn.Module,
    reference: torch.nn.Module,
    token_rows: Sequence[torch.Tensor],
    config: ValidateConfig,
) -> ValidationReport:
    """Run every configured metric over the same token rows for both models."""

    if reference is None:
        raise ValueError("validate() needs a reference model or checkpoint path")
    if not token_rows:
        raise ValueError("validation needs at least one token row")

    reference_device = config.device or _model_device(reference)
    candidate_device = config.device or _model_device(candidate)
    ppl_config: Optional[PPLConfig] = config.config_for(Metric.PPL)
    kld_config: Optional[KLDConfig] = config.config_for(Metric.KLD)
    topn_config: Optional[TOP_NConfig] = config.config_for(Metric.TOP_N)
    rmse_config: Optional[RMSEConfig] = config.config_for(Metric.RMSE)

    needs_logits = any(c is not None for c in (kld_config, topn_config, rmse_config))
    positions_wanted = _DEFAULT_POSITIONS
    if kld_config is not None:
        positions_wanted = max(positions_wanted, kld_config.positions)

    batches = list(token_rows)[: max(1, config.max_batches)]

    capture = _Capture()
    for tokens in batches:
        tokens = tokens.to(reference_device)
        with torch.no_grad():
            reference_logits = _logits_of(reference(input_ids=tokens))
        if ppl_config is not None:
            total, count = mean_negative_log_likelihood(reference_logits, tokens)
            capture.nll_sum += total
            capture.tokens += count
        if needs_logits:
            positions = _sample_positions(reference_logits.shape[1], positions_wanted)
            if positions.numel() == 0:
                raise ValueError("validation rows are too short to sample a prediction position")
            capture.logits.append(reference_logits[:, positions, :].detach().cpu().to(torch.float16))

    kld_values: List[float] = []
    rmse_values: List[float] = []
    overlap_values: Dict[int, List[float]] = {}
    cand_nll_sum = 0.0
    cand_tokens = 0

    for index, tokens in enumerate(batches):
        tokens = tokens.to(candidate_device)
        with torch.no_grad():
            candidate_logits = _logits_of(candidate(input_ids=tokens))
        if ppl_config is not None:
            total, count = mean_negative_log_likelihood(candidate_logits, tokens)
            cand_nll_sum += total
            cand_tokens += count
        if not needs_logits:
            continue

        positions = _sample_positions(candidate_logits.shape[1], positions_wanted)
        # Compare on the host: the reference capture is already there, and this
        # keeps the metric math independent of which device each model lives on.
        picked = candidate_logits[:, positions, :].detach().cpu().float()
        reference_picked = capture.logits[index].float()
        if kld_config is not None:
            kld_values.append(
                kl_divergence(reference_picked, picked, temperature=kld_config.temperature)
            )
        if rmse_config is not None:
            rmse_values.append(logit_rmse(reference_picked, picked, normalize=rmse_config.normalize))
        if topn_config is not None:
            for k in topn_config.top_k:
                overlap_values.setdefault(k, []).append(
                    topk_overlap(reference_picked, picked, k=k)
                )

    values: Dict[str, Any] = {}
    reference_values: Dict[str, Any] = {}
    if ppl_config is not None:
        if cand_tokens == 0 or capture.tokens == 0:
            raise ValueError("validation produced no scored tokens")
        reference_values["PPL"] = capture.perplexity()
        values["PPL"] = math.exp(cand_nll_sum / cand_tokens)
    if kld_config is not None:
        values["KLD"] = sum(kld_values) / len(kld_values)
    if rmse_config is not None:
        values["RMSE"] = sum(rmse_values) / len(rmse_values)
    if topn_config is not None:
        values["TOP_N"] = {k: sum(v) / len(v) for k, v in overlap_values.items()}

    report = ValidationReport(
        values=values,
        batches=len(batches),
        positions=positions_wanted if needs_logits else 0,
        thresholds=dict(config.thresholds),
        reference=reference_values,
    )
    if config.strict and report.passed is False:
        raise ValidationError(report)
    return report
