# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Statistically lossless quantization (SLQ) utilities from arXiv:2605.02404.

The subpackage exposes the paper's fidelity metrics, asymmetric-vs-symmetric
variance law, non-uniform bitwidth allocation, and single-point calibration as
standalone helpers that can be composed with the existing GPT-QModel pipeline.
"""

from .allocation import (
    allocate_bitwidth_ilp,
    binary_search_budget,
    linear_sensitivity,
    reconstruction_error,
    shapley_sensitivity,
)
from .calibration import DistributionLosslessCalibrator, TaskLosslessCalibrator
from .config import build_dynamic_bits
from .gamma import (
    centering_inefficiency,
    gamma_squared_variance_law,
    quantization_noise_variance_asym,
    quantization_noise_variance_sym,
    step_size_asym,
    step_size_sym,
)
from .metrics import (
    expected_acceptance_rate,
    kl_divergence_topk,
    next_token_distributions,
)

__all__ = [
    "allocate_bitwidth_ilp",
    "binary_search_budget",
    "build_dynamic_bits",
    "centering_inefficiency",
    "DistributionLosslessCalibrator",
    "expected_acceptance_rate",
    "gamma_squared_variance_law",
    "kl_divergence_topk",
    "linear_sensitivity",
    "next_token_distributions",
    "quantization_noise_variance_asym",
    "quantization_noise_variance_sym",
    "reconstruction_error",
    "shapley_sensitivity",
    "step_size_asym",
    "step_size_sym",
    "TaskLosslessCalibrator",
]
