#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Validate a tiny Hessian-weighted adaptive-rounding QUBO with CUDA-Q QAOA."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_TOY_QUBITS = 16


@dataclass(frozen=True)
class RoundingProblem:
    weights: np.ndarray
    lower: np.ndarray
    scale: float
    hessian: np.ndarray
    qubo_constant: float
    qubo_linear: np.ndarray
    qubo_pair: np.ndarray


@dataclass(frozen=True)
class IsingProblem:
    constant: float
    linear_z: np.ndarray
    pair_zz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a toy GPTQ-like Hessian QUBO through CUDA-Q on one visible NVIDIA GPU."
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=8,
        help=f"Toy rounding decisions, at most {MAX_TOY_QUBITS}.",
    )
    parser.add_argument("--layers", type=int, default=3, help="QAOA layer count.")
    parser.add_argument(
        "--restarts", type=int, default=6, help="Classical QAOA optimizer restarts."
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=250,
        help="Maximum COBYLA evaluations per restart.",
    )
    parser.add_argument(
        "--shots", type=int, default=10000, help="Final optimized-circuit samples."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260723,
        help="Deterministic problem and optimizer seed.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="Optional JSON result path."
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.qubits <= MAX_TOY_QUBITS:
        raise ValueError(f"--qubits must be in [2, {MAX_TOY_QUBITS}].")
    if args.layers < 1:
        raise ValueError("--layers must be positive.")
    if args.restarts < 1:
        raise ValueError("--restarts must be positive.")
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive.")
    if args.shots < 1:
        raise ValueError("--shots must be positive.")


def enumerate_binary(qubits: int) -> np.ndarray:
    integers = np.arange(1 << qubits, dtype=np.uint64)
    shifts = np.arange(qubits - 1, -1, -1, dtype=np.uint64)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.float64)


def exact_costs(problem: RoundingProblem, states: np.ndarray) -> np.ndarray:
    quantized = problem.lower[None, :] + problem.scale * states
    error = problem.weights[None, :] - quantized
    return np.einsum("bi,ij,bj->b", error, problem.hessian, error)


def qubo_costs(problem: RoundingProblem, states: np.ndarray) -> np.ndarray:
    linear = states @ problem.qubo_linear
    pair = np.einsum("bi,ij,bj->b", states, problem.qubo_pair, states)
    return problem.qubo_constant + linear + pair


def build_rounding_problem(
    weights: np.ndarray, scale: float, hessian: np.ndarray
) -> RoundingProblem:
    lower = np.floor(weights / scale) * scale
    remainder = weights - lower
    hessian_times_remainder = hessian @ remainder

    qubo_linear = (
        scale * scale * np.diag(hessian) - 2.0 * scale * hessian_times_remainder
    )
    qubo_pair = np.triu(2.0 * scale * scale * hessian, k=1)
    qubo_constant = float(remainder @ hessian @ remainder)
    return RoundingProblem(
        weights=weights,
        lower=lower,
        scale=scale,
        hessian=hessian,
        qubo_constant=qubo_constant,
        qubo_linear=qubo_linear,
        qubo_pair=qubo_pair,
    )


def qubo_to_ising(problem: RoundingProblem) -> IsingProblem:
    pair_sums = problem.qubo_pair.sum(axis=0) + problem.qubo_pair.sum(axis=1)
    constant = (
        problem.qubo_constant
        + 0.5 * float(problem.qubo_linear.sum())
        + 0.25 * float(problem.qubo_pair.sum())
    )
    linear_z = -0.5 * problem.qubo_linear - 0.25 * pair_sums
    pair_zz = 0.25 * problem.qubo_pair
    return IsingProblem(constant=constant, linear_z=linear_z, pair_zz=pair_zz)


def ising_costs(problem: IsingProblem, states: np.ndarray) -> np.ndarray:
    spins = 1.0 - 2.0 * states
    linear = spins @ problem.linear_z
    pair = np.einsum("bi,ij,bj->b", spins, problem.pair_zz, spins)
    return problem.constant + linear + pair


def make_correlated_problem(
    qubits: int, seed: int
) -> tuple[RoundingProblem, np.ndarray, np.ndarray]:
    """Find one deterministic covariance case where correlated rounding improves on RTN."""

    states = enumerate_binary(qubits)
    for offset in range(256):
        rng = np.random.default_rng(seed + offset)
        sample_count = max(4 * qubits, 32)
        latent_width = max(2, qubits // 3)
        latent = rng.normal(size=(sample_count, latent_width))
        mixing = rng.normal(size=(latent_width, qubits))
        activations = latent @ mixing + 0.15 * rng.normal(size=(sample_count, qubits))
        hessian = activations.T @ activations / float(sample_count)
        hessian += 0.05 * np.eye(qubits)
        hessian /= float(np.diag(hessian).mean())

        scale = 0.25
        base = rng.integers(-5, 5, size=qubits)
        fractions = rng.uniform(0.15, 0.85, size=qubits)
        weights = scale * (base + fractions)
        problem = build_rounding_problem(weights, scale, hessian)

        costs = exact_costs(problem, states)
        nearest = (fractions >= 0.5).astype(np.float64)
        nearest_cost = float(exact_costs(problem, nearest[None, :])[0])
        optimum = float(costs.min())
        improvement = nearest_cost - optimum
        if improvement > max(1e-8, 0.02 * nearest_cost):
            return problem, states, costs

    raise RuntimeError(
        "Could not construct a correlated toy problem with a non-trivial rounding improvement."
    )


def validate_mapping(
    problem: RoundingProblem, ising: IsingProblem, states: np.ndarray
) -> float:
    direct = exact_costs(problem, states)
    qubo = qubo_costs(problem, states)
    ising_values = ising_costs(ising, states)
    max_error = float(
        max(np.max(np.abs(direct - qubo)), np.max(np.abs(direct - ising_values)))
    )
    if max_error > 1e-10:
        raise AssertionError(
            f"QUBO/Ising mapping error {max_error:.3e} exceeds tolerance."
        )
    return max_error


def build_cudaq_hamiltonian(
    cudaq: Any, ising: IsingProblem, normalization: float
) -> Any:
    from cudaq import spin

    hamiltonian = (ising.constant / normalization) * spin.i(0)
    for index, coefficient in enumerate(ising.linear_z):
        hamiltonian += (float(coefficient) / normalization) * spin.z(index)
    for left in range(ising.pair_zz.shape[0]):
        for right in range(left + 1, ising.pair_zz.shape[1]):
            coefficient = float(ising.pair_zz[left, right])
            if coefficient:
                hamiltonian += (
                    (coefficient / normalization) * spin.z(left) * spin.z(right)
                )
    return hamiltonian


def build_qaoa_kernel(
    cudaq: Any, ising: IsingProblem, normalization: float, layers: int
) -> Any:
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(ising.linear_z.size)
    kernel.h(qubits)

    for layer in range(layers):
        gamma = parameters[layer]
        beta = parameters[layers + layer]
        for index, coefficient in enumerate(ising.linear_z):
            angle = 2.0 * float(coefficient) / normalization * gamma
            kernel.rz(angle, qubits[index])
        for left in range(ising.pair_zz.shape[0]):
            for right in range(left + 1, ising.pair_zz.shape[1]):
                coefficient = float(ising.pair_zz[left, right])
                if coefficient:
                    kernel.cx(qubits[left], qubits[right])
                    angle = 2.0 * coefficient / normalization * gamma
                    kernel.rz(angle, qubits[right])
                    kernel.cx(qubits[left], qubits[right])
        for index in range(ising.linear_z.size):
            kernel.rx(2.0 * beta, qubits[index])

    return kernel


def optimize_qaoa(
    cudaq: Any,
    kernel: Any,
    hamiltonian: Any,
    *,
    layers: int,
    restarts: int,
    maxiter: int,
    seed: int,
) -> tuple[float, np.ndarray, int]:
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_energy = math.inf
    best_parameters: np.ndarray | None = None
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(
            cudaq.observe(kernel, hamiltonian, parameters.tolist()).expectation()
        )

    bounds = [(-math.pi, math.pi)] * layers + [(-0.5 * math.pi, 0.5 * math.pi)] * layers
    for _ in range(restarts):
        initial = np.concatenate(
            [
                rng.uniform(-math.pi, math.pi, size=layers),
                rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=layers),
            ]
        )
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            bounds=bounds,
            options={"maxiter": maxiter, "catol": 1e-8, "tol": 1e-7},
        )
        if float(result.fun) < best_energy:
            best_energy = float(result.fun)
            best_parameters = np.asarray(result.x, dtype=np.float64)

    if best_parameters is None:
        raise RuntimeError("QAOA optimization did not return parameters.")
    return best_energy, best_parameters, evaluations


def bitstring_to_state(bitstring: str, qubits: int) -> np.ndarray:
    bits = bitstring.replace(" ", "")
    if len(bits) != qubits or any(bit not in "01" for bit in bits):
        raise ValueError(f"Unexpected CUDA-Q bitstring {bitstring!r}.")
    return np.fromiter((int(bit) for bit in bits), dtype=np.float64)


def format_bits(state: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in state)


def main() -> None:
    args = parse_args()
    validate_args(args)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices or "," in visible_devices:
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one GPU before importing CUDA-Q."
        )

    import cudaq

    cudaq.set_target("nvidia")
    cudaq.set_random_seed(args.seed)
    if cudaq.num_available_gpus() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA-Q GPU, found {cudaq.num_available_gpus()}."
        )

    problem, states, costs = make_correlated_problem(args.qubits, args.seed)
    ising = qubo_to_ising(problem)
    mapping_error = validate_mapping(problem, ising, states)

    optimum = float(costs.min())
    optimal_mask = np.isclose(costs, optimum, rtol=0.0, atol=1e-11)
    optimal_states = states[optimal_mask]
    fractions = (problem.weights - problem.lower) / problem.scale
    nearest_state = (fractions >= 0.5).astype(np.float64)
    nearest_cost = float(exact_costs(problem, nearest_state[None, :])[0])

    energy_span = float(costs.max() - costs.min())
    normalization = max(energy_span, np.finfo(np.float64).eps)
    hamiltonian = build_cudaq_hamiltonian(cudaq, ising, normalization)
    kernel = build_qaoa_kernel(cudaq, ising, normalization, args.layers)
    normalized_energy, parameters, evaluations = optimize_qaoa(
        cudaq,
        kernel,
        hamiltonian,
        layers=args.layers,
        restarts=args.restarts,
        maxiter=args.maxiter,
        seed=args.seed + 10000,
    )

    counts = cudaq.sample(kernel, parameters.tolist(), shots_count=args.shots)
    measured = [(str(bitstring), int(count)) for bitstring, count in counts.items()]
    measured.sort(key=lambda item: item[1], reverse=True)

    sampled_records = []
    for bitstring, count in measured:
        state = bitstring_to_state(bitstring, args.qubits)
        cost = float(exact_costs(problem, state[None, :])[0])
        sampled_records.append((cost, bitstring, count))
    sampled_records.sort()
    best_sampled_cost, best_sampled_bits, best_sampled_count = sampled_records[0]

    optimal_bitstrings = {format_bits(state) for state in optimal_states}
    optimal_shots = sum(
        count for bitstring, count in measured if bitstring in optimal_bitstrings
    )
    optimal_probability = optimal_shots / float(args.shots)
    uniform_optimal_probability = len(optimal_bitstrings) / float(states.shape[0])
    optimal_probability_enrichment = optimal_probability / uniform_optimal_probability
    sampled_mean_cost = sum(cost * count for cost, _, count in sampled_records) / float(
        args.shots
    )
    uniform_mean_cost = float(costs.mean())
    qaoa_expectation = normalized_energy * normalization

    condition_number = float(np.linalg.cond(problem.hessian))
    result = {
        "cuda_visible_devices": visible_devices,
        "cudaq_version": str(cudaq.__version__),
        "target": cudaq.get_target().name,
        "available_gpus": cudaq.num_available_gpus(),
        "qubits": args.qubits,
        "qaoa_layers": args.layers,
        "optimizer_restarts": args.restarts,
        "optimizer_evaluations": evaluations,
        "shots": args.shots,
        "hessian_condition_number": condition_number,
        "mapping_max_abs_error": mapping_error,
        "classical_states_checked": int(states.shape[0]),
        "nearest_bits": format_bits(nearest_state),
        "nearest_cost": nearest_cost,
        "optimal_bits": sorted(optimal_bitstrings),
        "optimal_cost": optimum,
        "uniform_mean_cost": uniform_mean_cost,
        "uniform_optimal_probability": uniform_optimal_probability,
        "qaoa_expectation": qaoa_expectation,
        "qaoa_sampled_mean_cost": sampled_mean_cost,
        "qaoa_best_sampled_bits": best_sampled_bits,
        "qaoa_best_sampled_count": best_sampled_count,
        "qaoa_best_sampled_cost": best_sampled_cost,
        "qaoa_optimal_sample_probability": optimal_probability,
        "qaoa_optimal_probability_enrichment": optimal_probability_enrichment,
        "qaoa_parameters": parameters.tolist(),
    }

    print("CUDA-Q GPTQ-like adaptive-rounding probe")
    print(f"visible GPU token : {visible_devices}")
    print(f"CUDA-Q / target   : {cudaq.__version__} / {cudaq.get_target().name}")
    print(f"qubits / QAOA p   : {args.qubits} / {args.layers}")
    print(f"Hessian condition : {condition_number:.6f}")
    print(f"mapping max error : {mapping_error:.3e}")
    print()
    print("method             bits              Hessian-weighted cost")
    print("-----------------  ----------------  ---------------------")
    print(f"round-to-nearest   {format_bits(nearest_state):<16}  {nearest_cost:.12f}")
    print(f"classical optimum  {sorted(optimal_bitstrings)[0]:<16}  {optimum:.12f}")
    print(f"QAOA best sample   {best_sampled_bits:<16}  {best_sampled_cost:.12f}")
    print()
    print(f"QAOA expectation               : {qaoa_expectation:.12f}")
    print(f"QAOA sampled mean              : {sampled_mean_cost:.12f}")
    print(f"uniform mean                   : {uniform_mean_cost:.12f}")
    print(f"QAOA optimum sampling probability: {optimal_probability:.6f}")
    print(f"uniform optimum probability    : {uniform_optimal_probability:.6f}")
    print(f"QAOA probability enrichment    : {optimal_probability_enrichment:.3f}x")
    print(f"optimizer evaluations          : {evaluations}")
    print(f"classical states checked       : {states.shape[0]}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"JSON result                    : {args.json_out}")


if __name__ == "__main__":
    main()
