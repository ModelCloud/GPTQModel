# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Configuration for :meth:`gptqmodel.models.base.BaseQModel.validate`.

Each metric carries its own config because they do not share knobs: a sliding
window means nothing to KLD, and a top-k list means nothing to PPL. The
container only owns what all of them need - the data, the reference, the sample
budget and the thresholds.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple, Union


__all__ = [
    "Metric",
    "PPLConfig",
    "KLDConfig",
    "TOP_NConfig",
    "RMSEConfig",
    "ValidateConfig",
]


class Metric(str, Enum):
    """Short names used in code and in the report; docs carry the long forms."""

    PPL = "PPL"
    KLD = "KLD"
    TOP_N = "TOP_N"
    RMSE = "RMSE"


@dataclass(frozen=True)
class PPLConfig:
    """Perplexity over sliding windows of the token stream."""

    window: int = 2048
    stride: int = 1024

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

    def __post_init__(self):
        if self.positions <= 0:
            raise ValueError(f"KLDConfig.positions must be positive, got {self.positions}")
        if self.temperature <= 0:
            raise ValueError(f"KLDConfig.temperature must be positive, got {self.temperature}")


@dataclass(frozen=True)
class TOP_NConfig:
    """Top-k set agreement; ``top_k`` may name several widths at once."""

    top_k: Tuple[int, ...] = (1,)

    def __post_init__(self):
        widths = tuple(int(k) for k in self.top_k)
        if not widths:
            raise ValueError("TOP_NConfig.top_k must name at least one width")
        if any(k <= 0 for k in widths):
            raise ValueError(f"TOP_NConfig.top_k must be positive, got {widths}")
        object.__setattr__(self, "top_k", widths)


@dataclass(frozen=True)
class RMSEConfig:
    """Root-mean-square logit difference at the sampled positions."""

    normalize: bool = False


_DEFAULT_CONFIGS = {
    Metric.PPL: PPLConfig,
    Metric.KLD: KLDConfig,
    Metric.TOP_N: TOP_NConfig,
    Metric.RMSE: RMSEConfig,
}

MetricSpec = Union[Metric, str, PPLConfig, KLDConfig, TOP_NConfig, RMSEConfig]


@dataclass(frozen=True)
class ValidateConfig:
    """What to compare, against what, and how much of it to read.

    ``metrics`` accepts metric names (defaults are used) and per-metric configs,
    in any mix. ``dataset`` defaults to the calibration set when validation runs
    inside ``quantize()``; ``reference`` is only needed when validating an
    already-quantized checkpoint.
    """

    metrics: Tuple[MetricSpec, ...] = (
        PPLConfig(),
        KLDConfig(),
        TOP_NConfig(top_k=(1, 5)),
        RMSEConfig(),
    )
    dataset: Any = None
    reference: Any = None
    max_batches: int = 4
    batch_size: Union[int, str] = 1
    device: Any = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    strict: bool = False
    seed: int = 0

    def __post_init__(self):
        resolved = []
        seen = set()
        for spec in self.metrics:
            config = self._resolve(spec)
            name = _metric_of(config)
            if name in seen:
                raise ValueError(f"ValidateConfig.metrics names {name.value} twice")
            seen.add(name)
            resolved.append(config)
        if not resolved:
            raise ValueError("ValidateConfig.metrics must name at least one metric")
        object.__setattr__(self, "metrics", tuple(resolved))

        if self.max_batches <= 0:
            raise ValueError(f"ValidateConfig.max_batches must be positive, got {self.max_batches}")
        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ValueError(
                f"ValidateConfig.batch_size must be a positive int or 'auto', got {self.batch_size!r}"
            )
        for key, value in dict(self.thresholds).items():
            if Metric(key) not in seen:
                raise ValueError(f"threshold for {key!r} but that metric was not requested")
            if not isinstance(value, (int, float)):
                raise TypeError(f"threshold for {key!r} must be a number, got {type(value).__name__}")
        object.__setattr__(self, "thresholds", dict(self.thresholds))

    @staticmethod
    def _resolve(spec: MetricSpec):
        if isinstance(spec, (PPLConfig, KLDConfig, TOP_NConfig, RMSEConfig)):
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
            if _metric_of(config) is metric:
                return config
        return None


def _metric_of(config) -> Metric:
    if isinstance(config, PPLConfig):
        return Metric.PPL
    if isinstance(config, KLDConfig):
        return Metric.KLD
    if isinstance(config, TOP_NConfig):
        return Metric.TOP_N
    if isinstance(config, RMSEConfig):
        return Metric.RMSE
    raise TypeError(f"not a metric config: {type(config).__name__}")
