# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.adjacent import (
    AdjacentExactIncompleteError,
    adjacent_branch_bound_cuda,
    adjacent_exact,
    adjacent_exact_blocks,
    adjacent_exact_cuda,
    build_adjacent_rounding_qubo,
)
from gptqmodel.utils.adjacent_exact import (
    _MAX_WORKER_WARPS,
    adjacent_branch_bound_candidates,
    adjacent_exact_candidates,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _problem(
    bits: int,
    size: int,
    *,
    block_size: int | None = None,
    sym: bool = False,
    device: str = "cuda",
):
    generator = torch.Generator().manual_seed(8100 + 100 * bits + size)
    weight = 0.45 * torch.tanh(
        torch.randn(size, generator=generator, dtype=torch.float64)
    )
    activations = torch.randn(3 * size, size, generator=generator, dtype=torch.float64)
    hessian = activations.mT @ activations / activations.shape[0]
    if block_size is not None:
        block_ids = torch.arange(size) // block_size
        hessian[block_ids.unsqueeze(0) != block_ids.unsqueeze(1)] = 0
    hessian += 0.1 * torch.eye(size, dtype=torch.float64)
    maxq = (1 << bits) - 1
    scale = max(0.02, 2.4 / maxq)
    return build_adjacent_rounding_qubo(
        weight.to(device),
        hessian.to(device),
        scale=scale,
        zero=(maxq + 1) / 2 if sym else maxq // 2,
        bits=bits,
    )


def _dense_planted_problem(size: int, *, device: str = "cuda"):
    weight = torch.full((size,), 0.9, dtype=torch.float64)
    hessian = torch.eye(size, dtype=torch.float64)
    hessian += 1e-3 * torch.ones(size, size, dtype=torch.float64)
    return build_adjacent_rounding_qubo(
        weight.to(device),
        hessian.to(device),
        scale=1.0,
        zero=4.0,
        bits=3,
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_adjacent_exact_cuda_matches_torch_exhaustive(bits: int):
    cpu_problem = _problem(bits, 9, device="cpu")
    cuda_problem = _problem(bits, 9)

    expected = adjacent_exact(cpu_problem)
    # Seven worker warps force each warp to traverse many Gray-code states and
    # cross the periodic FP64 rebase boundary.
    actual = adjacent_exact_cuda(cuda_problem, decompose=False, warps=7)

    assert actual.states_checked == 1 << cuda_problem.active_decisions
    assert actual.state.shape == (cuda_problem.size,)
    assert actual.state.dtype == torch.float64
    assert actual.state.device.type == "cuda"
    assert torch.isfinite(actual.state).all()
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_adjacent_branch_bound_cuda_matches_torch_exhaustive(bits: int):
    cpu_problem = _problem(bits, 9, device="cpu")
    cuda_problem = _problem(bits, 9)
    expected = adjacent_exact(cpu_problem)

    actual = adjacent_branch_bound_cuda(cuda_problem, split_depth=3)

    assert actual.optimal
    assert actual.lower_bound == actual.cost
    assert actual.nodes_visited > 0
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("size", [5, 12, 16])
def test_adjacent_branch_bound_cuda_matches_dense_exhaustive_sizes(size: int):
    cpu_problem = _problem(3, size, device="cpu")
    cuda_problem = _problem(3, size)
    expected = adjacent_exact(cpu_problem)

    actual = adjacent_branch_bound_cuda(cuda_problem, split_depth=min(4, size))

    assert actual.optimal
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


def test_adjacent_exact_cuda_factorizes_exact_blocks_on_non_default_stream():
    cpu_problem = _problem(3, 12, block_size=6, device="cpu")
    cuda_problem = _problem(3, 12, block_size=6)
    expected = adjacent_exact(cpu_problem)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = adjacent_exact_cuda(cuda_problem)
    stream.synchronize()

    assert actual.states_checked == 2 * (1 << 6)
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("size", [64, 128])
def test_adjacent_exact_cuda_factorizes_large_symmetric_group_shapes(size: int):
    cpu_problem = _problem(3, size, block_size=8, sym=True, device="cpu")
    cuda_problem = _problem(3, size, block_size=8, sym=True)
    expected = adjacent_exact_blocks(cpu_problem, block_size=8)

    actual = adjacent_exact_cuda(cuda_problem)

    assert actual.states_checked == (size // 8) * (1 << 8)
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("size", [64, 128])
def test_adjacent_exact_cuda_natively_certifies_dense_large_component(size: int):
    problem = _dense_planted_problem(size)

    actual = adjacent_exact_cuda(problem, split_depth=6)

    assert actual.optimal
    assert actual.lower_bound == actual.cost
    assert actual.nodes_visited > 0
    assert torch.count_nonzero(problem.pair + problem.pair.mT) == size * (size - 1)
    torch.testing.assert_close(
        actual.state,
        torch.ones_like(actual.state),
        rtol=0.0,
        atol=0.0,
    )


def test_adjacent_branch_bound_cuda_reports_uncertified_node_budget():
    cpu_problem = _problem(3, 16, sym=True, device="cpu")
    problem = _problem(3, 16, sym=True)
    expected = adjacent_exact(cpu_problem)

    partial = adjacent_branch_bound_cuda(
        problem,
        split_depth=0,
        max_nodes_per_worker=1,
        require_optimal=False,
    )

    assert not partial.optimal
    assert partial.nodes_visited == 1
    assert partial.lower_bound is not None
    assert partial.lower_bound <= expected.cost + 1e-12
    assert expected.cost <= partial.cost + 1e-12
    with pytest.raises(AdjacentExactIncompleteError) as raised:
        adjacent_branch_bound_cuda(
            problem,
            split_depth=0,
            max_nodes_per_worker=1,
        )
    assert raised.value.result.cost == partial.cost


def test_adjacent_branch_bound_cuda_rejects_more_than_128_active_decisions():
    problem = _problem(3, 129, sym=True)

    with pytest.raises(ValueError, match="at most 128 active decisions"):
        adjacent_branch_bound_cuda(problem)


def test_adjacent_exact_cuda_rejects_cpu_problem():
    with pytest.raises(ValueError, match="CUDA-resident"):
        adjacent_exact_cuda(_problem(2, 4, device="cpu"))


def test_adjacent_exact_cuda_rejects_invalid_native_coefficients():
    constant = torch.zeros((), device="cuda", dtype=torch.float64)
    linear = torch.zeros(3, device="cuda", dtype=torch.float64)
    interaction = torch.zeros(3, 3, device="cuda", dtype=torch.float64)
    candidate_states, candidate_costs = adjacent_exact_candidates(
        constant, linear, interaction, warps=3
    )
    assert candidate_states.shape == candidate_costs.shape == (3,)
    assert candidate_states.dtype == torch.int64
    assert candidate_costs.dtype == torch.float64
    assert candidate_states.device == candidate_costs.device == linear.device
    assert torch.isfinite(candidate_costs).all()

    native = adjacent_branch_bound_candidates(
        constant,
        linear,
        interaction,
        split_depth=2,
    )
    assert native.states.shape == (4, 2)
    assert native.costs.shape == native.root_lower_bounds.shape == (4,)
    assert native.nodes_visited.shape == native.completed.shape == (4,)
    assert native.completed.bool().all()
    assert native.global_best.shape == ()

    interaction[0, 1] = 1.0

    with pytest.raises(ValueError, match="symmetric"):
        adjacent_exact_candidates(constant, linear, interaction)
    interaction[1, 0] = 1.0
    interaction[0, 0] = 1.0
    with pytest.raises(ValueError, match="diagonal"):
        adjacent_exact_candidates(constant, linear, interaction)
    interaction.zero_()
    linear[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        adjacent_exact_candidates(constant, linear, interaction)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
def test_adjacent_exact_cuda_fp32_matches_cpu_exhaustive(bits: int, sym: bool):
    """FP32 Gray-code exact solver must match the FP64 CPU exhaustive reference."""
    cpu_problem = _problem(bits, 20, sym=sym, device="cpu")
    cuda_problem = _problem(bits, 20, sym=sym, device="cuda")
    expected = adjacent_exact(cpu_problem)

    actual = adjacent_exact_cuda(cuda_problem, decompose=False, warps=0)

    assert actual.states_checked == 1 << cuda_problem.active_decisions
    torch.testing.assert_close(actual.state.cpu(), expected.state, rtol=0.0, atol=0.0)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("size", [28, 30, 32])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sym", [False, True])
def test_adjacent_exact_cuda_fp32_stable_across_warp_granularity(
    size: int, bits: int, sym: bool
):
    """Different warp granularities must agree; this catches FP32 rounding drift."""
    problem = _problem(bits, size, sym=sym, device="cuda")
    default = adjacent_exact_cuda(problem, decompose=False, warps=0)
    fine = adjacent_exact_cuda(problem, decompose=False, warps=_MAX_WORKER_WARPS)

    total_states = 1 << problem.active_decisions
    assert default.states_checked == total_states
    assert fine.states_checked == total_states
    assert default.optimal
    assert fine.optimal
    torch.testing.assert_close(default.state, fine.state, rtol=0.0, atol=0.0)
    assert default.cost == pytest.approx(fine.cost, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("size", [28, 30, 32])
def test_adjacent_exact_cuda_fp32_finds_dense_planted_optimum(size: int):
    """The FP32 accumulator must still recover the known all-ones optimum."""
    problem = _dense_planted_problem(size)
    expected_state = torch.ones_like(problem.weight)

    for warps in (0, _MAX_WORKER_WARPS):
        actual = adjacent_exact_cuda(problem, decompose=False, warps=warps)
        assert actual.optimal
        torch.testing.assert_close(
            actual.state, expected_state, rtol=0.0, atol=0.0
        )
