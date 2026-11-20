# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fast divergence checks between a quantized model and its reference.

These metrics say whether quantization moved the computation; they do not say
whether the model is good. End-to-end quality belongs to a benchmark
(Evalution); this is the seconds-long check that catches a kernel, packing or
calibration change that silently rewrote the math.
"""

from .config import (
    KLDConfig,
    Metric,
    PPLConfig,
    RMSEConfig,
    TOP_NConfig,
    ValidateConfig,
)
from .metrics import (
    kl_divergence,
    logit_rmse,
    mean_negative_log_likelihood,
    topk_overlap,
)
from .report import ValidationReport
from .runner import ValidationError, validate_models


__all__ = [
    "KLDConfig",
    "Metric",
    "PPLConfig",
    "RMSEConfig",
    "TOP_NConfig",
    "ValidateConfig",
    "ValidationError",
    "ValidationReport",
    "kl_divergence",
    "logit_rmse",
    "mean_negative_log_likelihood",
    "topk_overlap",
    "validate_models",
]
