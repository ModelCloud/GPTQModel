#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference math for group-size-32 Hessian-weighted adjacent rounding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUP_SIZE = 32
REFERENCE_BLOCK_SIZE = 8
SUPPORTED_BITS = (2, 3, 4, 8)


@dataclass(frozen=True)
class AdjacentRoundingProblem:
    bits: int
    weights: np.ndarray
    hessian: np.ndarray
    scale: float
    zero: int
    lower_codes: np.ndarray
    upper_codes: np.ndarray
    lower_values: np.ndarray
    steps: np.ndarray
    qubo_constant: float
    qubo_linear: np.ndarray
    qubo_pair: np.ndarray

    @property
    def size(self) -> int:
        return int(self.weights.size)

    @property
    def active_decisions(self) -> int:
        return int(np.count_nonzero(self.steps))


@dataclass(frozen=True)
class IsingProblem:
    constant: float
    linear_z: np.ndarray
    pair_zz: np.ndarray


@dataclass(frozen=True)
class ClassicalResult:
    state: np.ndarray
    cost: float
    states_checked: int


def enumerate_binary(size: int) -> np.ndarray:
    """Return binary states in CUDA-Q bitstring order, with qubit zero at the left."""

    if not 1 <= size <= 20:
        raise ValueError("Exhaustive enumeration size must be in [1, 20].")
    integers = np.arange(1 << size, dtype=np.uint64)
    shifts = np.arange(size - 1, -1, -1, dtype=np.uint64)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def _validate_inputs(
    weights: np.ndarray, hessian: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    hessian = np.asarray(hessian, dtype=np.float64)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, received {bits}.")
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("weights must be a non-empty rank-one array.")
    if hessian.shape != (weights.size, weights.size):
        raise ValueError("hessian shape must match the weight group.")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(hessian)):
        raise ValueError("weights and hessian must contain only finite values.")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("hessian must be symmetric.")
    if float(np.linalg.eigvalsh(hessian).min()) < -1e-10:
        raise ValueError("hessian must be positive semidefinite.")
    return weights, hessian


def _qubo_terms(
    weights: np.ndarray,
    hessian: np.ndarray,
    lower_values: np.ndarray,
    steps: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    residual = weights - lower_values
    hessian_times_residual = hessian @ residual
    qubo_constant = float(residual @ hessian_times_residual)
    qubo_linear = (
        steps * steps * np.diag(hessian) - 2.0 * steps * hessian_times_residual
    )
    qubo_pair = np.triu(2.0 * np.outer(steps, steps) * hessian, k=1)
    return qubo_constant, qubo_linear, qubo_pair


def build_adjacent_problem(
    weights: np.ndarray,
    hessian: np.ndarray,
    bits: int,
) -> AdjacentRoundingProblem:
    """Build a fixed-codebook lower/upper QUBO using GPTQ's asymmetric affine convention."""

    weights, hessian = _validate_inputs(weights, hessian, bits)
    maxq = (1 << bits) - 1
    minimum = min(float(weights.min()), 0.0)
    maximum = max(float(weights.max()), 0.0)
    if maximum <= minimum:
        raise ValueError("The affine range must have non-zero width.")

    scale = (maximum - minimum) / maxq
    zero = int(np.clip(np.rint(-minimum / scale), 0, maxq))
    return build_adjacent_problem_fixed_codebook(
        weights,
        hessian,
        bits,
        scale=scale,
        zero=zero,
    )


def build_adjacent_problem_fixed_codebook(
    weights: np.ndarray,
    hessian: np.ndarray,
    bits: int,
    *,
    scale: float,
    zero: int,
) -> AdjacentRoundingProblem:
    """Build a QUBO from the exact scale and zero emitted by GPTQ."""

    weights, hessian = _validate_inputs(weights, hessian, bits)
    maxq = (1 << bits) - 1
    scale = float(scale)
    zero = int(zero)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    if not 0 <= zero <= maxq:
        raise ValueError(f"zero must be in [0, {maxq}].")

    real_codes = weights / scale + zero
    lower_codes = np.clip(np.floor(real_codes), 0, maxq).astype(np.int64)
    upper_codes = np.clip(np.ceil(real_codes), 0, maxq).astype(np.int64)
    if np.any((upper_codes - lower_codes < 0) | (upper_codes - lower_codes > 1)):
        raise AssertionError("Adjacent code endpoints must differ by zero or one.")

    lower_values = scale * (lower_codes.astype(np.float64) - zero)
    steps = scale * (upper_codes - lower_codes).astype(np.float64)
    qubo_constant, qubo_linear, qubo_pair = _qubo_terms(
        weights, hessian, lower_values, steps
    )
    return AdjacentRoundingProblem(
        bits=bits,
        weights=weights,
        hessian=hessian,
        scale=float(scale),
        zero=zero,
        lower_codes=lower_codes,
        upper_codes=upper_codes,
        lower_values=lower_values,
        steps=steps,
        qubo_constant=qubo_constant,
        qubo_linear=qubo_linear,
        qubo_pair=qubo_pair,
    )


def problem_from_payload(payload: dict) -> AdjacentRoundingProblem:
    """Load and validate a Torch-exported GPTQ adjacent-rounding problem."""

    if payload.get("schema") != "gptqmodel-adjacent-qubo-v1":
        raise ValueError("Unsupported adjacent-rounding problem schema.")
    problem = build_adjacent_problem_fixed_codebook(
        np.asarray(payload["weight"], dtype=np.float64),
        np.asarray(payload["hessian"], dtype=np.float64),
        int(payload["bits"]),
        scale=float(payload["scale"]),
        zero=int(round(float(payload["zero"]))),
    )
    if int(payload["size"]) != problem.size:
        raise ValueError("Payload size does not match its weight vector.")
    if int(payload["active_decisions"]) != problem.active_decisions:
        raise ValueError("Payload active-decision count does not match its codebook.")

    exact_arrays = {
        "lower_codes": problem.lower_codes,
        "upper_codes": problem.upper_codes,
    }
    for name, expected in exact_arrays.items():
        actual = np.asarray(payload[name], dtype=expected.dtype)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Payload {name} does not match the reconstructed codebook."
            )

    coefficient_arrays = {
        "qubo_linear": problem.qubo_linear,
        "qubo_pair": problem.qubo_pair,
    }
    for name, expected in coefficient_arrays.items():
        actual = np.asarray(payload[name], dtype=np.float64)
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Payload {name} does not match the reconstructed QUBO.")
    if not np.isclose(
        float(payload["qubo_constant"]), problem.qubo_constant, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Payload qubo_constant does not match the reconstructed QUBO.")
    return problem


def slice_problem(
    problem: AdjacentRoundingProblem, start: int, stop: int
) -> AdjacentRoundingProblem:
    """Slice a problem without recomputing the group-level scale or zero point."""

    if not 0 <= start < stop <= problem.size:
        raise ValueError("Invalid problem slice.")
    weights = problem.weights[start:stop]
    hessian = problem.hessian[start:stop, start:stop]
    lower_values = problem.lower_values[start:stop]
    steps = problem.steps[start:stop]
    qubo_constant, qubo_linear, qubo_pair = _qubo_terms(
        weights, hessian, lower_values, steps
    )
    return AdjacentRoundingProblem(
        bits=problem.bits,
        weights=weights,
        hessian=hessian,
        scale=problem.scale,
        zero=problem.zero,
        lower_codes=problem.lower_codes[start:stop],
        upper_codes=problem.upper_codes[start:stop],
        lower_values=lower_values,
        steps=steps,
        qubo_constant=qubo_constant,
        qubo_linear=qubo_linear,
        qubo_pair=qubo_pair,
    )


def quantized_values(problem: AdjacentRoundingProblem, state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (problem.size,) or np.any((state != 0.0) & (state != 1.0)):
        raise ValueError("state must be a binary vector matching the problem size.")
    return problem.lower_values + problem.steps * state


def hessian_cost(problem: AdjacentRoundingProblem, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != problem.size:
        raise ValueError("states must have shape [samples, problem.size].")
    error = (
        problem.weights[None, :]
        - problem.lower_values[None, :]
        - states * problem.steps[None, :]
    )
    return np.einsum("bi,ij,bj->b", error, problem.hessian, error)


def qubo_cost(problem: AdjacentRoundingProblem, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    linear = states @ problem.qubo_linear
    pair = np.einsum("bi,ij,bj->b", states, problem.qubo_pair, states)
    return problem.qubo_constant + linear + pair


def qubo_to_ising(problem: AdjacentRoundingProblem) -> IsingProblem:
    """Map z=(1-Z)/2 to an Ising Hamiltonian with upper-triangular ZZ terms."""

    incident_pairs = problem.qubo_pair.sum(axis=0) + problem.qubo_pair.sum(axis=1)
    constant = (
        problem.qubo_constant
        + 0.5 * float(problem.qubo_linear.sum())
        + 0.25 * float(problem.qubo_pair.sum())
    )
    linear_z = -0.5 * problem.qubo_linear - 0.25 * incident_pairs
    pair_zz = 0.25 * problem.qubo_pair
    return IsingProblem(constant=constant, linear_z=linear_z, pair_zz=pair_zz)


def ising_cost(problem: IsingProblem, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    spins = 1.0 - 2.0 * states
    linear = spins @ problem.linear_z
    pair = np.einsum("bi,ij,bj->b", spins, problem.pair_zz, spins)
    return problem.constant + linear + pair


def round_to_nearest_state(problem: AdjacentRoundingProblem) -> np.ndarray:
    real_codes = problem.weights / problem.scale + problem.zero
    rounded_codes = np.clip(np.rint(real_codes), 0, (1 << problem.bits) - 1).astype(
        np.int64
    )
    state = np.zeros(problem.size, dtype=np.float64)
    active = problem.upper_codes != problem.lower_codes
    state[active] = rounded_codes[active] - problem.lower_codes[active]
    return state


def coordinate_descent(
    problem: AdjacentRoundingProblem, initial_state: np.ndarray
) -> ClassicalResult:
    """Run deterministic best-improvement bit flips to a one-flip local optimum."""

    state = np.asarray(initial_state, dtype=np.float64).copy()
    current_cost = float(hessian_cost(problem, state)[0])
    states_checked = 1
    while True:
        best_cost = current_cost
        best_index: int | None = None
        for index in np.flatnonzero(problem.steps):
            candidate = state.copy()
            candidate[index] = 1.0 - candidate[index]
            candidate_cost = float(hessian_cost(problem, candidate)[0])
            states_checked += 1
            if candidate_cost < best_cost - 1e-14:
                best_cost = candidate_cost
                best_index = int(index)
        if best_index is None:
            return ClassicalResult(
                state=state, cost=current_cost, states_checked=states_checked
            )
        state[best_index] = 1.0 - state[best_index]
        current_cost = best_cost


def exact_block_optimum(
    problem: AdjacentRoundingProblem,
    block_size: int = REFERENCE_BLOCK_SIZE,
) -> ClassicalResult:
    """Prove the optimum when the Hessian is block diagonal by exhaustive block factorization."""

    if problem.size % block_size:
        raise ValueError("problem size must be divisible by block_size.")
    block_ids = np.arange(problem.size) // block_size
    cross_block = block_ids[:, None] != block_ids[None, :]
    if np.max(np.abs(problem.hessian[cross_block]), initial=0.0) > 1e-14:
        raise ValueError("exact_block_optimum requires a block-diagonal Hessian.")

    local_states = enumerate_binary(block_size)
    optimum = np.zeros(problem.size, dtype=np.float64)
    states_checked = 0
    for start in range(0, problem.size, block_size):
        stop = start + block_size
        block = slice_problem(problem, start, stop)
        costs = hessian_cost(block, local_states)
        optimum[start:stop] = local_states[int(np.argmin(costs))]
        states_checked += int(local_states.shape[0])
    return ClassicalResult(
        state=optimum,
        cost=float(hessian_cost(problem, optimum)[0]),
        states_checked=states_checked,
    )


def make_reference_group(
    seed: int = 20260724,
    *,
    group_size: int = GROUP_SIZE,
    block_size: int = REFERENCE_BLOCK_SIZE,
    calibration_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic block-correlated weight group and activation Hessian."""

    if group_size != GROUP_SIZE:
        raise ValueError(f"The reference experiment requires group_size={GROUP_SIZE}.")
    if block_size != REFERENCE_BLOCK_SIZE:
        raise ValueError(
            f"The reference experiment requires block_size={REFERENCE_BLOCK_SIZE}."
        )

    rng = np.random.default_rng(seed)
    hessian = np.zeros((group_size, group_size), dtype=np.float64)
    latent_width = 3
    for start in range(0, group_size, block_size):
        latent = rng.normal(size=(calibration_samples, latent_width))
        mixing = rng.normal(size=(latent_width, block_size))
        activations = latent @ mixing + 0.25 * rng.normal(
            size=(calibration_samples, block_size)
        )
        block = activations.T @ activations / float(calibration_samples)
        block += 0.05 * np.eye(block_size)
        hessian[start : start + block_size, start : start + block_size] = block
    hessian /= float(np.diag(hessian).mean())

    weights = rng.standard_t(df=5, size=group_size) * 0.55
    weights += rng.normal(loc=0.0, scale=0.1, size=group_size)
    return weights, hessian


def format_bits(state: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in np.asarray(state))
