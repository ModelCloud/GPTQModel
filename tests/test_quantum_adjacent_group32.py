# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from scripts.quantum_quantization.adjacent_group32 import (
    build_adjacent_problem,
    enumerate_binary,
    exact_block_optimum,
    hessian_cost,
    ising_cost,
    make_reference_group,
    quantized_values,
    qubo_cost,
    qubo_to_ising,
    round_to_nearest_state,
)


@pytest.mark.parametrize("bits", [2, 3])
def test_group32_adjacent_codebook_matches_affine_round_to_nearest(bits: int):
    weights, hessian = make_reference_group()
    problem = build_adjacent_problem(weights, hessian, bits)
    state = round_to_nearest_state(problem)

    maxq = (1 << bits) - 1
    expected_codes = np.clip(np.rint(weights / problem.scale) + problem.zero, 0, maxq)
    expected_values = problem.scale * (expected_codes - problem.zero)

    assert problem.size == 32
    assert problem.active_decisions == 30
    assert np.all(problem.lower_codes >= 0)
    assert np.all(problem.upper_codes <= maxq)
    assert np.all(np.isin(problem.upper_codes - problem.lower_codes, [0, 1]))
    np.testing.assert_allclose(
        quantized_values(problem, state), expected_values, rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_direct_qubo_and_ising_costs_match_for_group32(bits: int):
    weights, hessian = make_reference_group()
    problem = build_adjacent_problem(weights, hessian, bits)
    ising = qubo_to_ising(problem)
    rng = np.random.default_rng(1000 + bits)
    states = rng.integers(0, 2, size=(1024, problem.size)).astype(np.float64)

    direct = hessian_cost(problem, states)
    np.testing.assert_allclose(
        qubo_cost(problem, states), direct, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        ising_cost(ising, states), direct, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_exact_block_solver_matches_independent_exhaustive_blocks(bits: int):
    weights, hessian = make_reference_group()
    problem = build_adjacent_problem(weights, hessian, bits)
    result = exact_block_optimum(problem)
    local_states = enumerate_binary(8)

    assert result.states_checked == 4 * 256
    assert result.cost < float(
        hessian_cost(problem, round_to_nearest_state(problem))[0]
    )
    for start in range(0, problem.size, 8):
        candidates = np.repeat(result.state[None, :], local_states.shape[0], axis=0)
        candidates[:, start : start + 8] = local_states
        assert result.cost <= float(hessian_cost(problem, candidates).min()) + 1e-12
