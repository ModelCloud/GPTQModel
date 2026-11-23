# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure a quantized model against its reference on the same token stream.

Metrics declare which captures they read, so the runner can fuse the ones that
agree: PPL needs the NLL sum, KLD/TOP_K/RMSE need sampled logits, and a single
forward per batch produces both - one pass for all four. A metric that needs a
capture no other metric produces would form its own pass; that path exists so
adding a weight-space metric later does not require rewriting the loop.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

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
    kl_divergence_sum,
    logit_rmse_sums,
    mean_negative_log_likelihood,
    topk_overlap_sum,
)
from .report import MeasurementReport


__all__ = [
    "MeasurementError",
    "ReferenceCapture",
    "TokenWindow",
    "build_token_rows",
    "capture_reference",
    "measure_against",
    "measure_models",
]

log = setup_logger()

#: Window used when PPL is not requested but rows must still be sliced.
_DEFAULT_WINDOW = 512

#: Positions kept per batch when a metric needs distributions rather than a scalar.
_DEFAULT_POSITIONS = 64

#: Row forms accepted by :func:`build_token_rows`.
_ROW_FORMATS = (
    "a text string, a token-id list/tuple, a token-id tensor, or a mapping "
    "with `input_ids` (tensor/list) or `text` (string)"
)


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
    positions: int = 0
    logits: List[torch.Tensor] = field(default_factory=list)

    def perplexity(self) -> Optional[float]:
        return None if self.tokens == 0 else math.exp(self.nll_sum / self.tokens)


@dataclass(frozen=True, eq=False)
class TokenWindow:
    """One model input plus which of its tokens are newly scored targets.

    ``score_mask`` is aligned with ``input_ids``: position ``j`` is True when the
    token at ``j`` is a next-token target this window is responsible for. The
    mask is what lets overlapping windows share context without scoring the
    shared targets twice, and what keeps the seam between adjacent windows from
    being dropped.
    """

    input_ids: torch.Tensor
    score_mask: torch.Tensor

    def __post_init__(self):
        if self.input_ids.dim() != 2 or self.input_ids.shape[0] != 1:
            raise ValueError(
                f"TokenWindow.input_ids must be [1, tokens], got {tuple(self.input_ids.shape)}"
            )
        if self.score_mask.shape != self.input_ids.shape:
            raise ValueError(
                f"TokenWindow.score_mask {tuple(self.score_mask.shape)} does not match "
                f"input_ids {tuple(self.input_ids.shape)}"
            )

    @classmethod
    def covering(cls, input_ids: torch.Tensor) -> "TokenWindow":
        """A window that scores every transition inside it (its first token is context)."""

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, 1:] = True
        return cls(input_ids=input_ids, score_mask=mask)


def _windows_of_row(encoded: torch.Tensor, window: int, stride: int) -> List[TokenWindow]:
    """Slice one row into windows whose scoring masks partition its transitions.

    A window feeds ``window`` tokens to the model so the logits see as much
    context as the configured window allows. Only targets not already scored by
    an earlier window are marked in ``score_mask``, so every transition in the
    row is scored exactly once however the windows overlap.

    The step is capped at ``window - 1``: a window with ``w`` tokens can score at
    most ``w - 1`` transitions, and stepping further would leave the transition
    at each seam unscored. Capping it also retains one token of context at every
    seam instead of introducing a gap.
    """

    length = int(encoded.shape[1])
    if length < 2:
        return []
    if length <= window:
        return [TokenWindow.covering(encoded)]

    step = window - 1 if stride <= 0 else min(int(stride), window - 1)
    windows: List[TokenWindow] = []
    # Global index of the first target token (index ``t`` means the transition
    # from token ``t - 1`` to token ``t``) this row has not scored yet.
    next_target = 1
    for start in range(0, length - 1, step):
        if next_target > length - 1:
            break
        piece = encoded[:, start:start + window]
        positions = torch.arange(piece.shape[1])
        target_index = start + positions
        score_mask = ((positions >= 1) & (target_index >= next_target)).reshape(1, -1)
        if not bool(score_mask.any()):
            continue
        windows.append(TokenWindow(input_ids=piece, score_mask=score_mask))
        next_target = start + int(piece.shape[1])
    return windows


def _encoded_mapping_row(tokenizer: Any, sample: Mapping) -> torch.Tensor:
    """Encode one HF dataset or calibration-style mapping row."""

    if "input_ids" in sample:
        input_ids = sample["input_ids"]
        if torch.is_tensor(input_ids):
            return input_ids.reshape(1, -1)
        if isinstance(input_ids, (list, tuple)):
            return torch.tensor([int(token) for token in input_ids], dtype=torch.long).reshape(1, -1)
        raise ValueError(
            "measurement dataset mapping row `input_ids` must be a tensor or "
            f"list of token ids, got {type(input_ids).__name__}"
        )
    if "text" in sample:
        text = sample["text"]
        if not isinstance(text, str):
            raise ValueError(
                "measurement dataset mapping row `text` must be a string, "
                f"got {type(text).__name__}"
            )
        if tokenizer is None:
            raise ValueError("measurement got text but no tokenizer is available")
        return tokenizer(text, return_tensors="pt")["input_ids"]
    raise ValueError(
        "measurement dataset mapping rows must contain `input_ids` (tensor/list) "
        f"or `text` (string), got keys {list(sample)}"
    )


def build_token_rows(
    tokenizer: Any,
    dataset: Sequence[Any],
    *,
    window: int = _DEFAULT_WINDOW,
    stride: int = 0,
    max_rows: int = 4,
) -> List[TokenWindow]:
    """Turn text, token-id, or mapping rows into windows with scoring masks.

    Windows overlap when ``stride < window`` and always retain one token of
    context at a seam, so the masks partition every transition in a row exactly
    once. Mapping rows carry either ``input_ids`` (a tensor or token-id list) or
    ``text`` (a string), which is the shape of HF dataset and calibration rows.
    Pass the result straight to :func:`measure_models`.
    """

    if dataset is None:
        raise ValueError(
            "measurement needs `dataset` (texts, token-id rows, or mapping rows)"
        )
    if window < 2:
        raise ValueError(f"measurement window must be >= 2, got {window}")
    rows: List[TokenWindow] = []
    for sample in dataset:
        if isinstance(sample, str):
            if tokenizer is None:
                raise ValueError("measurement got text but no tokenizer is available")
            encoded = tokenizer(sample, return_tensors="pt")["input_ids"]
        elif torch.is_tensor(sample):
            encoded = sample.reshape(1, -1)
        elif isinstance(sample, Mapping):
            encoded = _encoded_mapping_row(tokenizer, sample)
        elif isinstance(sample, (list, tuple)):
            encoded = torch.tensor([int(token) for token in sample], dtype=torch.long).reshape(1, -1)
        else:
            raise ValueError(
                f"measurement dataset rows must be {_ROW_FORMATS}, got {type(sample).__name__}"
            )
        if encoded.shape[1] < 2:
            continue

        rows.extend(_windows_of_row(encoded, window, stride))
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("measurement dataset produced no usable token rows")
    return rows[:max_rows]


@dataclass
class _Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    score_mask: torch.Tensor


def _as_window(row: Any) -> TokenWindow:
    """Accept a plain ``[1, tokens]`` row as "score every transition in it"."""

    if isinstance(row, TokenWindow):
        return row
    input_ids = row if torch.is_tensor(row) else torch.as_tensor(row)
    return TokenWindow.covering(input_ids.reshape(1, -1))


def _batch_rows(
    rows: Sequence[Union[TokenWindow, torch.Tensor]], batch: int, pad_token_id: int = 0
) -> List[_Batch]:
    """Pad rows into batches of ``batch``.

    Padding is unavoidable for text - windows differ in length - and it is only
    safe because every consumer masks it: the attention mask keeps pads out of
    attention, the NLL ignores padded targets, and position sampling stays inside
    each row's real length. Padding tokens are therefore inert rather than
    counted, which is the difference between batching and corrupting the metric.
    """

    windows = [_as_window(row) for row in rows]
    batches: List[_Batch] = []
    for start in range(0, len(windows), max(1, batch)):
        chunk = windows[start:start + batch]
        widths = [int(window.input_ids.shape[1]) for window in chunk]
        width = max(widths)
        input_ids = torch.full((len(chunk), width), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
        score_mask = torch.zeros((len(chunk), width), dtype=torch.bool)
        for index, window in enumerate(chunk):
            length = int(window.input_ids.shape[1])
            input_ids[index, :length] = window.input_ids[0, :length]
            attention_mask[index, :length] = 1
            score_mask[index, :length] = window.score_mask[0, :length].bool()
        batches.append(
            _Batch(input_ids=input_ids, attention_mask=attention_mask, score_mask=score_mask)
        )
    return batches


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
    batches: Sequence[_Batch],
    *,
    device: torch.device,
    needs_nll: bool,
    positions: int,
) -> _Captures:
    """One pass over the batches, collecting what the fused metrics need."""

    captures = _Captures()
    for batch in batches:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        with torch.no_grad():
            logits = _logits_of(model(input_ids=input_ids, attention_mask=attention_mask))
        if needs_nll:
            total, count = mean_negative_log_likelihood(
                logits,
                input_ids,
                attention_mask=attention_mask,
                score_mask=batch.score_mask.to(device),
            )
            captures.nll_sum += total
            captures.tokens += count
        if positions:
            # Sample each row independently. Using the batch's shortest row as
            # the cap silently dropped positions from longer peers, so the
            # sample (and every metric weighting it) changed with `batch`.
            picked_rows = []
            for row, length in enumerate(attention_mask.sum(dim=1).tolist()):
                index = _sample_positions(int(length), positions).to(device)
                if index.numel() == 0:
                    continue
                picked_rows.append(logits[row].index_select(0, index))
            if picked_rows:
                picked = torch.cat(picked_rows, dim=0).unsqueeze(0)
                captures.positions += int(picked.shape[1])
                # Keep the dtype the model returned. Narrowing to FP16 here
                # rounded distinct FP32 logits together - hiding a top-token
                # flip - and turned finite BF16 logits above 65504 into
                # infinities, so a model compared with itself reported NaN.
                # The metrics promote to FP32 when they compute, which keeps
                # the capture lossless without widening the stored sample.
                captures.logits.append(picked.detach().cpu())
    return captures


@dataclass
class ReferenceCapture:
    """Reference-side artifacts, kept so the candidate can be measured later.

    Quantization usually has the fp16 weights in memory only until the first
    module is replaced. Capturing here means `quantize(measure=...)` never needs
    a second model, which at 7B is the difference between 14 GB and a report.
    """

    batches: List[_Batch]
    nll_sum: float
    tokens: int
    positions: int
    logits: List[torch.Tensor]
    positions_wanted: int


def _plan(config: MeasureConfig) -> tuple:
    """The captures the metrics need and the sample width they ask for."""

    fuse(config.metrics)
    needs = frozenset().union(*[member.capture_needs for member in config.metrics])
    # An explicit KLDConfig.positions wins; the default only applies when no
    # metric asked for a width (TOP_K and RMSE read the same sample).
    asked = [member.positions for member in config.metrics if isinstance(member, KLDConfig)]
    positions_wanted = max(asked) if asked else _DEFAULT_POSITIONS
    if CAPTURE_SAMPLED_LOGITS not in needs:
        positions_wanted = 0
    return needs, positions_wanted


def capture_reference(
    reference: torch.nn.Module,
    token_rows: Sequence[Union[TokenWindow, torch.Tensor]],
    config: MeasureConfig,
    *,
    pad_token_id: Optional[int] = None,
) -> ReferenceCapture:
    """Read the reference side once, before the model is modified."""

    if reference is None:
        raise ValueError("measurement needs a reference model or checkpoint path")
    if not token_rows:
        raise ValueError("measurement needs at least one token row")

    needs, positions_wanted = _plan(config)
    batches = _batch_rows(list(token_rows), config.batch, pad_token_id or 0)[
        : max(1, config.max_batches)
    ]
    captures = _capture(
        reference,
        batches,
        device=config.device or _model_device(reference),
        needs_nll=CAPTURE_NLL in needs,
        positions=positions_wanted,
    )
    return ReferenceCapture(
        batches=batches,
        nll_sum=captures.nll_sum,
        tokens=captures.tokens,
        positions=captures.positions,
        logits=captures.logits,
        positions_wanted=positions_wanted,
    )


def measure_against(
    candidate: torch.nn.Module,
    capture: ReferenceCapture,
    config: MeasureConfig,
) -> MeasurementReport:
    """Measure ``candidate`` against artifacts captured earlier."""

    needs, _positions_wanted = _plan(config)
    candidate_captures = _capture(
        candidate,
        capture.batches,
        device=config.device or _model_device(candidate),
        needs_nll=CAPTURE_NLL in needs,
        positions=capture.positions_wanted,
    )
    reference_captures = _Captures(
        nll_sum=capture.nll_sum,
        tokens=capture.tokens,
        positions=capture.positions,
        logits=capture.logits,
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
        batches=len(capture.batches),
        positions=candidate_captures.positions,
    )
    if config.strict and report.overall is MetricGrades.FAIL:
        raise MeasurementError(report)
    return report


def measure_models(
    candidate: torch.nn.Module,
    reference: torch.nn.Module,
    token_rows: Sequence[Union[TokenWindow, torch.Tensor]],
    config: MeasureConfig,
    *,
    pad_token_id: Optional[int] = None,
) -> MeasurementReport:
    """Run every configured metric over the same token rows for both models.

    ``token_rows`` accepts :class:`TokenWindow` objects from
    :func:`build_token_rows` (which carry a scoring mask) or plain ``[1, tokens]``
    tensors (every transition in the row is scored). The mask is what makes PPL
    weight each transition in the stream exactly once across overlapping windows.
    """

    capture = capture_reference(reference, token_rows, config, pad_token_id=pad_token_id)
    return measure_against(candidate, capture, config)


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

    # Pool positions and elements across batches, then divide once. Averaging
    # per-batch means would give a short final batch the same vote as a full
    # one, so `batch` would still move the number even with a correct sample.
    if name is Metric.KLD:
        total = 0.0
        count = 0
        for reference, candidate in pairs:
            batch_total, batch_count = kl_divergence_sum(
                reference, candidate, temperature=member.temperature
            )
            total += batch_total
            count += batch_count
        return _pooled_mean(total, count, name)

    if name is Metric.RMSE:
        squared_error = 0.0
        squared_reference = 0.0
        count = 0
        for reference, candidate in pairs:
            batch_error, batch_reference, batch_count = logit_rmse_sums(
                reference, candidate, normalize=member.normalize
            )
            squared_error += batch_error
            squared_reference += batch_reference
            count += batch_count
        if count == 0:
            raise ValueError(f"{name.value} needs sampled logits but none were captured")
        difference = math.sqrt(squared_error / count)
        if not member.normalize:
            return difference
        reference_scale = math.sqrt(squared_reference / count)
        if reference_scale == 0:
            raise ValueError("cannot normalise by a reference logit RMS of zero")
        return difference / reference_scale

    if name is Metric.TOP_K:
        overlaps = {}
        for k in member.k:
            total = 0.0
            count = 0
            for reference, candidate in pairs:
                batch_total, batch_count = topk_overlap_sum(reference, candidate, k=k)
                total += batch_total
                count += batch_count
            overlaps[k] = _pooled_mean(total, count, name)
        return overlaps

    raise TypeError(f"unknown metric config {type(member).__name__}")


def _pooled_mean(total: float, count: int, name: Metric) -> float:
    if count == 0:
        raise ValueError(f"{name.value} needs sampled logits but none were captured")
    return total / count
