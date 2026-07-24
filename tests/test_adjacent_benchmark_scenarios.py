# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from scripts.quantum_quantization.benchmark_adjacent_exact import (
    SYMMETRIC_LARGE_SCENARIOS,
    make_activations,
)
from scripts.quantum_quantization.benchmark_adjacent_native import (
    SCENARIOS as NATIVE_SCENARIOS,
)


def test_symmetric_large_sweep_covers_group64_and_group128():
    assert len(SYMMETRIC_LARGE_SCENARIOS) == 8
    assert {scenario.size for scenario in SYMMETRIC_LARGE_SCENARIOS} == {64, 128}
    assert all(scenario.sym for scenario in SYMMETRIC_LARGE_SCENARIOS)
    assert len({scenario.name for scenario in SYMMETRIC_LARGE_SCENARIOS}) == 8

    for size in (64, 128):
        scenarios = [
            scenario
            for scenario in SYMMETRIC_LARGE_SCENARIOS
            if scenario.size == size
        ]
        assert {scenario.coupled_block_size for scenario in scenarios} == {
            None,
            8,
            16,
            32,
        }


def test_symmetric_large_block_activations_have_exact_zero_cross_couplings():
    for index, scenario in enumerate(SYMMETRIC_LARGE_SCENARIOS):
        calibration, heldout = make_activations(scenario, seed=9100 + index)
        assert calibration.shape[1] == heldout.shape[1] == scenario.size

        if scenario.coupled_block_size is None:
            assert bool((torch.count_nonzero(calibration, dim=1) <= 1).all())
            continue

        block_size = scenario.coupled_block_size
        block_ids = torch.arange(scenario.size) // block_size
        cross_block = block_ids.unsqueeze(0) != block_ids.unsqueeze(1)
        calibration_gram = calibration.mT @ calibration
        heldout_gram = heldout.mT @ heldout
        assert not bool(torch.count_nonzero(calibration_gram[cross_block]))
        assert not bool(torch.count_nonzero(heldout_gram[cross_block]))


def test_native_sweep_uses_full_dense_symmetric_group64_and_group128_hessians():
    assert len(NATIVE_SCENARIOS) == 6
    assert {scenario.size for scenario in NATIVE_SCENARIOS} == {64, 128}
    assert {scenario.activation for scenario in NATIVE_SCENARIOS} == {
        "dense",
        "ill_conditioned",
        "signed_correlated",
    }
    assert all(scenario.sym for scenario in NATIVE_SCENARIOS)
    assert all(scenario.coupled_block_size is None for scenario in NATIVE_SCENARIOS)

    for index, scenario in enumerate(NATIVE_SCENARIOS):
        calibration, _ = make_activations(scenario, seed=9300 + index)
        hessian = calibration.mT @ calibration
        off_diagonal = ~torch.eye(scenario.size, dtype=torch.bool)
        assert bool(hessian[off_diagonal].ne(0).all())
