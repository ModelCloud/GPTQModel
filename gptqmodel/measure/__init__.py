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
    TOPKConfig,
    fuse,
)
from .metrics import (
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)
from .report import MeasurementReport
from .runner import (
    MeasurementError,
    ReferenceCapture,
    build_token_rows,
    capture_reference,
    measure_against,
    measure_models,
)


__all__ = [
    "GradedThresholdConfig",
    "KLDConfig",
    "MeasureConfig",
    "MeasurementError",
    "MeasurementReport",
    "ReferenceCapture",
    "Metric",
    "MetricGrades",
    "PPLConfig",
    "RMSEConfig",
    "TOPKConfig",
    "build_token_rows",
    "capture_reference",
    "fuse",
    "kl_divergence",
    "logit_rmse",
    "mean_negative_log_likelihood",
    "measure_against",
    "measure_models",
    "topk_overlap",
]
