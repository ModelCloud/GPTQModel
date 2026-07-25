# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Single-point calibration and target-driven search from arXiv:2605.02404.

Implements the task-lossless single-point calibration procedure
(Algorithm 2) and a distribution-lossless search driven by an EAR threshold.
"""

from __future__ import annotations

import numpy as np

from .allocation import allocate_bitwidth_ilp


class TaskLosslessCalibrator:
    """Calibrate the linear KL-recovery relationship from one measurement.

    Following Section 3.3, the model ``recovery ≈ 1 - α * D_KL`` is fit using a
    single calibration point. The intercept is fixed at 1 because zero KL
    implies perfect recovery.
    """

    def __init__(
        self,
        calibration_kl: float,
        calibration_recovery: float,
        calibration_predicted_kl: float | None = None,
    ):
        """Initialize with the calibration point.

        Args:
            calibration_kl: Measured KL divergence at the calibration bitwidth.
            calibration_recovery: Measured task recovery at the calibration
                bitwidth, as a fraction of the BF16 baseline score.
            calibration_predicted_kl: Optional Shapley-predicted KL at the
                calibration point. If provided, a calibration ratio ``rho`` is
                computed and applied to all subsequent predictions.
        """

        if calibration_kl <= 0:
            raise ValueError("calibration_kl must be positive.")
        if not 0.0 <= calibration_recovery <= 1.0:
            raise ValueError("calibration_recovery must be in [0, 1].")

        self.calibration_kl = calibration_kl
        self.calibration_recovery = calibration_recovery
        # Slope of the linear KL-recovery model.
        self.alpha = (1.0 - calibration_recovery) / calibration_kl

        self.rho = 1.0
        if calibration_predicted_kl is not None and calibration_predicted_kl > 0:
            self.rho = calibration_kl / calibration_predicted_kl

    def kl_threshold(self, target_recovery: float) -> float:
        """Return the maximum KL that still yields ``target_recovery``."""

        if not 0.0 <= target_recovery <= 1.0:
            raise ValueError("target_recovery must be in [0, 1].")
        return (1.0 - target_recovery) / self.alpha

    def predicted_kl(self, shapley_predicted_kl: float) -> float:
        """Apply the calibration ratio to a Shapley-predicted KL."""

        return self.rho * shapley_predicted_kl

    def search(
        self,
        costs_kl: np.ndarray,
        bitwidths: list[int],
        target_recovery: float,
        *,
        weights: list[float] | np.ndarray | None = None,
        max_iter: int = 20,
        tolerance: float = 0.05,
    ) -> tuple[float, np.ndarray]:
        """Binary-search for the minimum average bitwidth meeting recovery.

        Args:
            costs_kl: ``(M, B)`` Shapley-predicted KL cost matrix.
            bitwidths: Candidate bitwidths.
            target_recovery: Desired recovery fraction (e.g. 0.99).
            weights: Optional per-group weights.
            max_iter: Maximum binary-search iterations.
            tolerance: Search tolerance in average bits.

        Returns:
            Tuple ``(best_budget, best_assignment)`` where ``best_assignment``
            is an array of column indices into ``bitwidths``.
        """

        threshold = self.kl_threshold(target_recovery)
        bitwidths_arr = np.asarray(bitwidths)
        low = float(bitwidths_arr.min())
        high = float(bitwidths_arr.max())

        best_assignment = allocate_bitwidth_ilp(costs_kl, bitwidths_arr, weights=weights, budget=high)
        best_budget = high

        for _ in range(max_iter):
            if high - low <= tolerance:
                break
            mid = (low + high) / 2.0
            assignment = allocate_bitwidth_ilp(costs_kl, bitwidths_arr, weights=weights, budget=mid)
            predicted = self.predicted_kl(sum(costs_kl[i, a] for i, a in enumerate(assignment)))
            if predicted <= threshold:
                best_budget = mid
                best_assignment = assignment
                high = mid
            else:
                low = mid

        return best_budget, best_assignment


class DistributionLosslessCalibrator:
    """Search for the minimum average bitwidth satisfying an EAR target."""

    def __init__(self, target_ear: float = 0.99):
        """Initialize with an EAR target.

        Args:
            target_ear: Minimum Expected Acceptance Rate (default 0.99).
        """

        if not 0.0 <= target_ear <= 1.0:
            raise ValueError("target_ear must be in [0, 1].")
        self.target_ear = target_ear

    def search(
        self,
        costs_ear: np.ndarray,
        bitwidths: list[int],
        *,
        weights: list[float] | np.ndarray | None = None,
        max_iter: int = 20,
        tolerance: float = 0.05,
    ) -> tuple[float, np.ndarray]:
        """Binary-search for the minimum average bitwidth meeting ``target_ear``.

        ``costs_ear`` is assumed to contain the predicted 1 - EAR degradation,
        so the predicted EAR is ``1 - sum(costs_ear[assignment])``.

        Args:
            costs_ear: ``(M, B)`` predicted 1 - EAR cost matrix.
            bitwidths: Candidate bitwidths.
            weights: Optional per-group weights.
            max_iter: Maximum binary-search iterations.
            tolerance: Search tolerance in average bits.

        Returns:
            Tuple ``(best_budget, best_assignment)``.
        """

        bitwidths_arr = np.asarray(bitwidths)
        low = float(bitwidths_arr.min())
        high = float(bitwidths_arr.max())

        best_assignment = allocate_bitwidth_ilp(costs_ear, bitwidths_arr, weights=weights, budget=high)
        best_budget = high

        for _ in range(max_iter):
            if high - low <= tolerance:
                break
            mid = (low + high) / 2.0
            assignment = allocate_bitwidth_ilp(costs_ear, bitwidths_arr, weights=weights, budget=mid)
            predicted_ear = 1.0 - sum(costs_ear[i, a] for i, a in enumerate(assignment))
            if predicted_ear >= self.target_ear:
                best_budget = mid
                best_assignment = assignment
                high = mid
            else:
                low = mid

        return best_budget, best_assignment
