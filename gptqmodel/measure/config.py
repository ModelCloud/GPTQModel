# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Configuration for :meth:`gptqmodel.models.base.BaseQModel.measure`.

Each metric owns its knobs *and* its acceptance grades, because a KLD budget is
a property of the KLD metric, not of the run. ``MeasureConfig`` only holds what
every metric needs: the data, the reference, the sample budget.

Metrics also declare which captures they read. Metrics with the same needs are
fused into one forward pass; a metric that needs something else gets its own.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple, Union


__all__ = [
    "Metric",
    "MetricGrades",
    "GradedThresholdConfig",
    "PPLConfig",
    "KLDConfig",
    "TOPKConfig",
    "RMSEConfig",
    "MeasureConfig",
]


class Metric(str, Enum):
    """Short names used in code and in the report; docs carry the long forms."""

    PPL = "PPL"
    KLD = "KLD"
    TOP_K = "TOP_K"
    RMSE = "RMSE"


class MetricGrades(str, Enum):
    """Ordered from best to worst; the report keeps the worst grade seen."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    @property
    def severity(self) -> int:
        return {"PASS": 0, "WARN": 1, "FAIL": 2}[self.value]


class GradedThresholdConfig:
    """Grade bounds written as comparison expressions.

    ``GradedThresholdConfig(grades={PASS: "<=0.02", WARN: "<=0.05"})`` reads as
    "at or below 0.02 passes, at or below 0.05 warns, everything else fails".
    ``FAIL`` is the fallthrough and takes no expression. Bounds are evaluated in
    grade order, so the first match wins and a value can never satisfy two
    grades.
    """

    __slots__ = ("_bounds",)

    _OPERATORS = {
        ">=": lambda value, bound: value >= bound,
        "<=": lambda value, bound: value <= bound,
        ">": lambda value, bound: value > bound,
        "<": lambda value, bound: value < bound,
    }

    def __init__(self, grades: Mapping[Union[str, MetricGrades], str]):
        bounds = []
        for grade, expression in grades.items():
            try:
                resolved = MetricGrades(grade)
            except ValueError:
                raise ValueError(
                    f"unknown grade {grade!r}; expected one of {[g.value for g in MetricGrades]}"
                ) from None
            if resolved is MetricGrades.FAIL:
                if str(expression).strip().lower() != "fallthrough":
                    raise ValueError(
                        "GradedThresholdConfig.FAIL is the fallthrough grade and takes "
                        f"'fallthrough', got {expression!r}"
                    )
                continue
            bounds.append((resolved, self._parse(expression, resolved)))
        if not bounds:
            raise ValueError("GradedThresholdConfig needs at least a PASS or WARN bound")
        bounds.sort(key=lambda item: item[0].severity)
        self._bounds = tuple(bounds)

    @classmethod
    def _parse(cls, expression: str, grade: MetricGrades):
        text = str(expression).strip()
        for operator in (">=", "<=", ">", "<"):
            if text.startswith(operator):
                try:
                    bound = float(text[len(operator):])
                except ValueError:
                    raise ValueError(
                        f"GradedThresholdConfig.{grade.value} bound {expression!r} is not a number"
                    ) from None
                return cls._OPERATORS[operator], bound
        raise ValueError(
            f"GradedThresholdConfig.{grade.value} must start with one of "
            f"{sorted(cls._OPERATORS)}, got {expression!r}"
        )

    def grade_for(self, value: float) -> MetricGrades:
        for grade, (compare, bound) in self._bounds:
            if compare(value, bound):
                return grade
        return MetricGrades.FAIL

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{grade.value}={compare.__name__}{bound}"
            for grade, (compare, bound) in self._bounds
        )
        return f"GradedThresholdConfig({parts}, FAIL=fallthrough)"


#: Captures a metric reads, and the pass that produces them.
CAPTURE_NLL = "nll"
CAPTURE_SAMPLED_LOGITS = "sampled_logits"

#: Which pass produces each capture. Metrics whose captures share a pass fuse.
_CAPTURE_PASS = {
    CAPTURE_NLL: "forward",
    CAPTURE_SAMPLED_LOGITS: "forward",
}


@dataclass(frozen=True)
class PPLConfig:
    """Perplexity over sliding windows of the token stream."""

    window: int = 2048
    stride: int = 1024
    threshold: Optional[GradedThresholdConfig] = None

    capture_needs: FrozenSet[str] = field(default=frozenset({CAPTURE_NLL}), init=False, repr=False)

    def __post_init__(self):
        if self.window <= 1:
            raise ValueError(f"PPLConfig.window must be > 1, got {self.window}")
        if self.stride < 0:
            raise ValueError(f"PPLConfig.stride must be >= 0, got {self.stride}")


@dataclass(frozen=True)
class KLDConfig:
    """Mean ``KL(reference ‖ candidate)`` over sampled token positions."""

    positions: int = 64
    temperature: float = 1.0
    threshold: Optional[GradedThresholdConfig] = None

    capture_needs: FrozenSet[str] = field(
        default=frozenset({CAPTURE_SAMPLED_LOGITS}), init=False, repr=False
    )

    def __post_init__(self):
        if self.positions <= 0:
            raise ValueError(f"KLDConfig.positions must be positive, got {self.positions}")
        if self.temperature <= 0:
            raise ValueError(f"KLDConfig.temperature must be positive, got {self.temperature}")


@dataclass(frozen=True)
class TOPKConfig:
    """Top-k set agreement; ``k`` may name several widths at once."""

    k: Tuple[int, ...] = (1,)
    threshold: Optional[GradedThresholdConfig] = None

    capture_needs: FrozenSet[str] = field(
        default=frozenset({CAPTURE_SAMPLED_LOGITS}), init=False, repr=False
    )

    def __post_init__(self):
        widths = tuple(int(k) for k in self.k)
        if not widths:
            raise ValueError("TOPKConfig.k must name at least one width")
        if any(k <= 0 for k in widths):
            raise ValueError(f"TOPKConfig.k must be positive, got {widths}")
        object.__setattr__(self, "k", widths)


@dataclass(frozen=True)
class RMSEConfig:
    """Root-mean-square logit difference at the sampled positions."""

    normalize: bool = False
    threshold: Optional[GradedThresholdConfig] = None

    capture_needs: FrozenSet[str] = field(
        default=frozenset({CAPTURE_SAMPLED_LOGITS}), init=False, repr=False
    )


_DEFAULT_CONFIGS = {
    Metric.PPL: PPLConfig,
    Metric.KLD: KLDConfig,
    Metric.TOP_K: TOPKConfig,
    Metric.RMSE: RMSEConfig,
}

MetricSpec = Union[Metric, str, PPLConfig, KLDConfig, TOPKConfig, RMSEConfig]
MetricSpecs = Union[MetricSpec, Sequence[MetricSpec]]


@dataclass(frozen=True)
class MeasureConfig:
    """What to measure, against what, and how much of it to read.

    ``metrics`` takes one config, several, or metric names (defaults are used).
    ``dataset`` defaults to the calibration set when measuring inside
    ``quantize()``; ``reference`` is only needed for an already-quantized
    checkpoint.
    """

    metrics: MetricSpecs = (
        PPLConfig(),
        KLDConfig(),
        TOPKConfig(k=(1, 5)),
        RMSEConfig(),
    )
    dataset: Any = None
    reference: Any = None
    max_batches: int = 4
    batch_size: int = 1
    device: Any = None
    strict: bool = False
    seed: int = 0

    def __post_init__(self):
        specs = self.metrics
        if isinstance(specs, (PPLConfig, KLDConfig, TOPKConfig, RMSEConfig)):
            specs = (specs,)
        elif isinstance(specs, (Metric, str)):
            specs = (specs,)

        resolved = []
        seen = set()
        for spec in specs:
            config = self._resolve(spec)
            name = metric_of(config)
            if name in seen:
                raise ValueError(f"MeasureConfig.metrics names {name.value} twice")
            seen.add(name)
            resolved.append(config)
        if not resolved:
            raise ValueError("MeasureConfig.metrics must name at least one metric")
        object.__setattr__(self, "metrics", tuple(resolved))

        if self.max_batches <= 0:
            raise ValueError(f"MeasureConfig.max_batches must be positive, got {self.max_batches}")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"MeasureConfig.batch_size must be a positive int, got {self.batch_size!r}"
            )

    @staticmethod
    def _resolve(spec: MetricSpec):
        if isinstance(spec, (PPLConfig, KLDConfig, TOPKConfig, RMSEConfig)):
            return spec
        if isinstance(spec, Metric):
            return _DEFAULT_CONFIGS[spec]()
        try:
            metric = Metric(str(spec))
        except ValueError:
            valid = [m.value for m in Metric]
            raise ValueError(f"unknown metric {spec!r}; expected one of {valid}") from None
        return _DEFAULT_CONFIGS[metric]()

    def config_for(self, metric: Metric):
        """The per-metric config in this run, or None when it was not requested."""

        for config in self.metrics:
            if metric_of(config) is metric:
                return config
        return None

    def thresholds(self) -> Dict[str, GradedThresholdConfig]:
        """Configured grades by metric name."""

        return {
            metric_of(config).value: config.threshold
            for config in self.metrics
            if config.threshold is not None
        }


def metric_of(config) -> Metric:
    if isinstance(config, PPLConfig):
        return Metric.PPL
    if isinstance(config, KLDConfig):
        return Metric.KLD
    if isinstance(config, TOPKConfig):
        return Metric.TOP_K
    if isinstance(config, RMSEConfig):
        return Metric.RMSE
    raise TypeError(f"not a metric config: {type(config).__name__}")


def fuse(configs: Sequence[Any]) -> Tuple[Tuple[Tuple[str, ...], Tuple[Any, ...]], ...]:
    """Group metrics into forward passes.

    Two metrics fuse when their captures come from the same pass: the NLL sum and
    the sampled logits are both produced by one forward over a batch, so PPL,
    KLD, TOP_K and RMSE share a single pass. A metric that needs a capture from a
    different source (a weight-space metric, say) maps to a different pass kind
    and gets its own group.
    """

    groups: Dict[Tuple[str, ...], list] = {}
    for config in configs:
        kinds = tuple(sorted({_CAPTURE_PASS.get(capture, capture) for capture in config.capture_needs}))
        groups.setdefault(kinds, []).append(config)
    return tuple((kinds, tuple(members)) for kinds, members in groups.items())
