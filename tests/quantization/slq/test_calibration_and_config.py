# GPU=-1
import numpy as np
import pytest
import torch

from gptqmodel.quantization.slq import (
    DistributionLosslessCalibrator,
    TaskLosslessCalibrator,
    build_dynamic_bits,
    linear_sensitivity,
)


def test_task_lossless_alpha():
    cal = TaskLosslessCalibrator(calibration_kl=0.02, calibration_recovery=0.95)
    # 1 - 0.95 = 0.05; alpha = 0.05 / 0.02 = 2.5
    assert abs(cal.alpha - 2.5) < 1e-6
    threshold = cal.kl_threshold(0.99)
    assert abs(threshold - 0.01 / 2.5) < 1e-6


def test_task_lossless_rho():
    cal = TaskLosslessCalibrator(
        calibration_kl=0.04,
        calibration_recovery=0.90,
        calibration_predicted_kl=0.02,
    )
    assert abs(cal.rho - 2.0) < 1e-6
    assert abs(cal.predicted_kl(0.01) - 0.02) < 1e-6


def test_task_lossless_search():
    torch.manual_seed(0)
    weights = [torch.randn(128) for _ in range(4)]
    bitwidths = [2, 3, 4, 5, 6, 7, 8]
    costs = linear_sensitivity(weights, bitwidths, symmetric=False)
    cal = TaskLosslessCalibrator(calibration_kl=0.1, calibration_recovery=0.90)
    budget, assignment = cal.search(costs, bitwidths, target_recovery=0.99)
    assert 2.0 <= budget <= 8.0
    assert len(assignment) == costs.shape[0]


def test_distribution_lossless_search():
    np.random.seed(0)
    bitwidths = [4, 5, 6, 7, 8]
    groups = 6
    # Synthetic 1 - EAR costs that decrease to near-zero at the top bitwidth.
    costs = np.array([[max(0.0, 0.05 - 0.0063 * b) * (i + 1) for b in bitwidths] for i in range(groups)])
    cal = DistributionLosslessCalibrator(target_ear=0.99)
    budget, assignment = cal.search(costs, bitwidths, tolerance=0.2)
    assert 4.0 <= budget <= 8.0
    predicted_ear = 1.0 - sum(costs[i, a] for i, a in enumerate(assignment))
    assert predicted_ear >= 0.99 - 1e-6


def test_build_dynamic_bits():
    groups = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.gate_proj"]
    bitwidths = [2, 4, 8]
    assignment = [0, 2]
    dynamic = build_dynamic_bits(groups, bitwidths, assignment)
    assert len(dynamic) == 2
    for key, entry in dynamic.items():
        assert "bits" in entry
        assert entry["bits"] in bitwidths


def test_build_dynamic_bits_with_overrides():
    groups = ["a", "b"]
    bitwidths = [4, 8]
    assignment = [0, 1]
    overrides = {"a": {"sym": False}, "b": {"group_size": 64}}
    dynamic = build_dynamic_bits(groups, bitwidths, assignment, additional_overrides=overrides)
    assert dynamic["^a$"]["sym"] is False
    assert dynamic["^b$"]["group_size"] == 64


def test_invalid_recovery_raises():
    with pytest.raises(ValueError):
        TaskLosslessCalibrator(calibration_kl=0.01, calibration_recovery=1.5)


def test_invalid_target_ear_raises():
    with pytest.raises(ValueError):
        DistributionLosslessCalibrator(target_ear=1.5)
