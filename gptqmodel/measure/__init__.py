# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fast divergence checks between a quantized model and its reference.

These metrics say whether quantization moved the computation; they do not say
whether the model is good. End-to-end quality belongs to a benchmark
(Evalution); this is the seconds-long check that catches a kernel, packing or
calibration change silently rewriting the math.
"""

from .config import (
    GradedThresholdConfig,
    KLDConfig,
    MeasureConfig,
    Metric,
    MetricGrades,
    PPLConfig,
    RMSEConfig,
    TOP_NConfig,
    fuse,
)
from .metrics import (
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)
from .report import MeasurementReport
from .runner import MeasurementError, build_token_rows, measure_models


__all__ = [
    "GradedThresholdConfig",
    "KLDConfig",
    "MeasureConfig",
    "MeasurementError",
    "MeasurementReport",
    "Metric",
    "MetricGrades",
    "PPLConfig",
    "RMSEConfig",
    "TOP_NConfig",
    "build_token_rows",
    "fuse",
    "kl_divergence",
    "logit_rmse",
    "mean_negative_log_likelihood",
    "measure_models",
    "topk_overlap",
]
