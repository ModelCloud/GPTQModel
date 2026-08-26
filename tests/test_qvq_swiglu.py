# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.quantization import (
    QVQConfig,
    SmoothSwiGLUConfig,
    apply_swiglu_reparameterization,
    choose_swiglu_scales,
    select_swiglu_candidate_triplet,
    swiglu_error_diagnostics,
    swiglu_jacobian_salience,
)


def _weights():
    torch.manual_seed(20260826)
    return (
        torch.randn(12, 8),
        torch.randn(12, 8),
        torch.randn(8, 12),
        torch.randn(32, 8),
    )


def test_swiglu_reparameterization_preserves_dense_mlp_output():
    gate, up, down, inputs = _weights()
    scales = torch.tensor(
        [0.5, 2.0, 1.25, 0.75, 1.0, 1.5, 0.8, 1.2, 0.6, 1.8, 0.9, 1.1]
    )
    _, transformed_up, transformed_down = apply_swiglu_reparameterization(
        gate, up, down, scales
    )

    dense = (F.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T
    transformed = (
        F.silu(inputs @ gate.T) * (inputs @ transformed_up.T)
    ) @ transformed_down.T
    torch.testing.assert_close(dense, transformed, rtol=1e-5, atol=1e-5)


def test_swiglu_scale_search_is_grouped_and_serializable():
    gate, up, down, inputs = _weights()
    scales, stats = choose_swiglu_scales(
        inputs,
        gate,
        up,
        down,
        group_size=4,
        candidate_exponents=(-1.0, 0.0, 1.0),
    )

    assert scales.shape == (12,)
    assert stats["groups"] == 3
    for start in range(0, 12, 4):
        assert torch.all(scales[start : start + 4] == scales[start])
    config = QVQConfig(smooth_swiglu=SmoothSwiGLUConfig(enabled=True, group_size=4))
    assert QVQConfig.from_quant_config(config.to_dict()).to_dict() == config.to_dict()


def test_swiglu_grouped_scale_uses_aggregate_optimum():
    gate, up, down, inputs = _weights()
    scales, _ = choose_swiglu_scales(
        inputs,
        gate,
        up,
        down,
        group_size=4,
        candidate_exponents=(0.0,),
    )
    gate_activation = inputs @ gate.T
    up_activation = inputs @ up.T
    hidden = F.silu(gate_activation) * up_activation
    group_a = (F.silu(gate_activation).square().mean(0) * up.square().mean(1))[:4].sum()
    group_b = (hidden.square().mean(0) * down.square().sum(0))[:4].sum()
    expected = (group_b / group_a).pow(0.25).clamp(0.5, 2.0)
    torch.testing.assert_close(scales[:4], torch.full((4,), expected))


def test_swiglu_proxy_objective_counts_each_group_once():
    gate, up, down, inputs = _weights()
    _, stats = choose_swiglu_scales(
        inputs,
        gate,
        up,
        down,
        group_size=4,
        candidate_exponents=(0.0,),
    )
    gate_activation = inputs @ gate.T
    up_activation = inputs @ up.T
    silu_gate = F.silu(gate_activation)
    hidden = silu_gate * up_activation
    up_error_weight = silu_gate.square().mean(0) * up.square().mean(1)
    down_error_weight = hidden.square().mean(0) * down.square().sum(0)
    expected = 0.0
    for start in range(0, 12, 4):
        stop = start + 4
        group_a = up_error_weight[start:stop].sum()
        group_b = down_error_weight[start:stop].sum()
        scale = (group_b / group_a).pow(0.25).clamp(0.5, 2.0)
        expected += float(
            (
                up_error_weight[start:stop] * scale.square()
                + down_error_weight[start:stop] / scale.square()
            )
            .sum()
            .item()
        )
    assert stats["proxy_objective"] == pytest.approx(expected)


def test_swiglu_scale_search_supports_non_divisible_group_width():
    gate, up, down, inputs = _weights()
    gate, up, down = gate[:11], up[:11], down[:, :11]
    scales, stats = choose_swiglu_scales(
        inputs, gate, up, down, group_size=4, candidate_exponents=(0.0,)
    )
    assert scales.shape == (11,)
    assert stats["groups"] == 3
    assert torch.isfinite(scales).all()


def test_swiglu_jacobian_salience_matches_definition():
    gate, up, down, inputs = _weights()
    gate_activation = inputs @ gate.T
    up_activation = inputs @ up.T
    salience = swiglu_jacobian_salience(gate_activation, up_activation, down)
    down_energy = down.square().sum(dim=0)
    derivative = torch.sigmoid(gate_activation) * (
        1 + gate_activation * (1 - torch.sigmoid(gate_activation))
    )
    expected_gate = (up_activation * derivative).square().mean(0)
    expected_up = F.silu(gate_activation).square().mean(0)
    torch.testing.assert_close(salience["gate"], expected_gate * down_energy)
    torch.testing.assert_close(salience["up"], expected_up * down_energy)


def test_swiglu_triplet_selection_scores_nonlinear_output():
    gate, up, down, inputs = _weights()
    result = select_swiglu_candidate_triplet(
        inputs,
        gate,
        up,
        down,
        [gate * 1.05, gate],
        [up, up * 0.95],
        [down * 1.05, down],
        beam_size=2,
    )

    assert result["gate_index"] == 1
    assert result["up_index"] == 0
    assert result["down_index"] == 1
    assert result["loss"] == 0.0
    assert result["evaluated_triplets"] == 8
    assert len(result["beam"]) == 2


def test_swiglu_error_diagnostics_exposes_amplification_distributions():
    gate, up, down, inputs = _weights()
    metrics = swiglu_error_diagnostics(
        inputs @ gate.T,
        inputs @ up.T,
        inputs @ (gate * 1.01).T,
        inputs @ (up * 0.99).T,
        down,
        down * 1.01,
    )

    assert metrics["post_swiglu_relative_error"]["max"] > 0
    assert metrics["A_swiglu"]["p95"] > 0
    assert metrics["A_down"]["p50"] > 0
    assert set(metrics["cross_error"]) == {
        "mean",
        "mean_abs",
        "min",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "max",
        "negative_fraction",
        "positive_fraction",
    }
