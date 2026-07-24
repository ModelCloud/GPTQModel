#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare FP64 CUDA-Q QAOA with classical 2/3-bit group-32 adjacent rounding."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from adjacent_group32 import (
    GROUP_SIZE,
    REFERENCE_BLOCK_SIZE,
    AdjacentRoundingProblem,
    IsingProblem,
    build_adjacent_problem,
    coordinate_descent,
    enumerate_binary,
    exact_block_optimum,
    format_bits,
    hessian_cost,
    ising_cost,
    make_reference_group,
    problem_from_payload,
    quantized_values,
    qubo_cost,
    qubo_to_ising,
    round_to_nearest_state,
    slice_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the accuracy-first 2/3-bit group-size-32 adjacent-rounding probe on one CUDA-Q GPU."
    )
    parser.add_argument("--bits", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    parser.add_argument(
        "--layers",
        type=int,
        default=5,
        help="QAOA layers per correlated 8-variable block.",
    )
    parser.add_argument(
        "--restarts", type=int, default=6, help="COBYLA restarts per block."
    )
    parser.add_argument(
        "--maxiter", type=int, default=300, help="COBYLA evaluations per restart."
    )
    parser.add_argument(
        "--block-shots",
        type=int,
        default=20000,
        help="Samples used to postselect each block.",
    )
    parser.add_argument(
        "--full-shots",
        type=int,
        default=20000,
        help="Samples from the full 32-qubit circuit.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--problem-json",
        type=Path,
        default=None,
        help="Optional problem payload exported from GPTQ's real Hessian and quantizer.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--skip-full-state",
        action="store_true",
        help="Debug block optimization without allocating the final 32-qubit state vector.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.layers < 1:
        raise ValueError("--layers must be positive.")
    if args.restarts < 1:
        raise ValueError("--restarts must be positive.")
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive.")
    if args.block_shots < 1 or args.full_shots < 1:
        raise ValueError("shot counts must be positive.")
    if len(set(args.bits)) != len(args.bits):
        raise ValueError("--bits values must be unique.")


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
                    coefficient / normalization * spin.z(left) * spin.z(right)
                )
    return hamiltonian


def apply_cost_layer(
    kernel: Any,
    qubits: Any,
    ising: IsingProblem,
    normalization: float,
    gamma: Any,
    *,
    offset: int = 0,
) -> None:
    for index, coefficient in enumerate(ising.linear_z):
        if coefficient:
            kernel.rz(
                2.0 * float(coefficient) / normalization * gamma, qubits[offset + index]
            )
    for left in range(ising.pair_zz.shape[0]):
        for right in range(left + 1, ising.pair_zz.shape[1]):
            coefficient = float(ising.pair_zz[left, right])
            if coefficient:
                kernel.cx(qubits[offset + left], qubits[offset + right])
                kernel.rz(
                    2.0 * coefficient / normalization * gamma, qubits[offset + right]
                )
                kernel.cx(qubits[offset + left], qubits[offset + right])


def build_block_qaoa_kernel(
    cudaq: Any, ising: IsingProblem, normalization: float, layers: int
) -> Any:
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(ising.linear_z.size)
    kernel.h(qubits)
    for layer in range(layers):
        apply_cost_layer(kernel, qubits, ising, normalization, parameters[layer])
        beta = parameters[layers + layer]
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
    initial_points = [np.zeros(2 * layers, dtype=np.float64)]
    for _ in range(restarts - 1):
        initial_points.append(
            np.concatenate(
                (
                    rng.uniform(-math.pi, math.pi, size=layers),
                    rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=layers),
                )
            )
        )

    for initial in initial_points:
        result = minimize(
            objective,
            initial,
            method="COBYLA",
            bounds=bounds,
            options={"maxiter": maxiter, "catol": 1e-9, "tol": 1e-8},
        )
        if float(result.fun) < best_energy:
            best_energy = float(result.fun)
            best_parameters = np.asarray(result.x, dtype=np.float64)
    if best_parameters is None:
        raise RuntimeError("QAOA optimizer returned no parameters.")
    return best_energy, best_parameters, evaluations


def bitstring_to_state(bitstring: str, size: int) -> np.ndarray:
    bits = bitstring.replace(" ", "")
    if len(bits) != size or any(bit not in "01" for bit in bits):
        raise ValueError(f"Unexpected CUDA-Q bitstring {bitstring!r}.")
    return np.fromiter((int(bit) for bit in bits), dtype=np.float64)


def score_counts(
    problem: AdjacentRoundingProblem,
    counts: Any,
    optimum_cost: float,
) -> dict[str, Any]:
    records: list[tuple[float, str, int]] = []
    total_shots = 0
    weighted_cost = 0.0
    optimal_shots = 0
    for bitstring, count_value in counts.items():
        count = int(count_value)
        state = bitstring_to_state(str(bitstring), problem.size)
        cost = float(hessian_cost(problem, state)[0])
        records.append((cost, str(bitstring), count))
        total_shots += count
        weighted_cost += cost * count
        if math.isclose(cost, optimum_cost, rel_tol=0.0, abs_tol=1e-10):
            optimal_shots += count
    if not records or total_shots < 1:
        raise RuntimeError("CUDA-Q returned no samples.")
    records.sort()
    best_cost, best_bits, best_count = records[0]
    return {
        "best_bits": best_bits,
        "best_state": bitstring_to_state(best_bits, problem.size),
        "best_cost": best_cost,
        "best_count": best_count,
        "mean_cost": weighted_cost / total_shots,
        "optimal_probability": optimal_shots / total_shots,
        "unique_states": len(records),
        "shots": total_shots,
    }


def build_full_qaoa_kernel(
    cudaq: Any,
    block_isings: list[IsingProblem],
    normalizations: list[float],
    layers: int,
) -> Any:
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(GROUP_SIZE)
    kernel.h(qubits)
    parameters_per_block = 2 * layers
    for layer in range(layers):
        for block_index, (ising, normalization) in enumerate(
            zip(block_isings, normalizations, strict=True)
        ):
            parameter_offset = block_index * parameters_per_block
            qubit_offset = block_index * REFERENCE_BLOCK_SIZE
            apply_cost_layer(
                kernel,
                qubits,
                ising,
                normalization,
                parameters[parameter_offset + layer],
                offset=qubit_offset,
            )
        for block_index in range(len(block_isings)):
            parameter_offset = block_index * parameters_per_block
            qubit_offset = block_index * REFERENCE_BLOCK_SIZE
            beta = parameters[parameter_offset + layers + layer]
            for local_index in range(REFERENCE_BLOCK_SIZE):
                kernel.rx(2.0 * beta, qubits[qubit_offset + local_index])
    return kernel


def gpu_metadata(visible_device: str) -> dict[str, Any]:
    fields = "name,uuid,pci.bus_id,memory.total,driver_version"
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible_device,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "name": values[0],
        "uuid": values[1],
        "pci_bus_id": values[2],
        "memory_total_mib": int(values[3]),
        "driver_version": values[4],
    }


def validate_mapping(problem: AdjacentRoundingProblem, seed: int) -> float:
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 2, size=(4096, problem.size)).astype(np.float64)
    direct = hessian_cost(problem, states)
    qubo = qubo_cost(problem, states)
    ising = ising_cost(qubo_to_ising(problem), states)
    error = float(max(np.max(np.abs(direct - qubo)), np.max(np.abs(direct - ising))))
    if error > 1e-10:
        raise AssertionError(f"QUBO/Ising mapping error {error:.3e} exceeds tolerance.")
    return error


def run_bitwidth(
    cudaq: Any,
    args: argparse.Namespace,
    problem: AdjacentRoundingProblem,
    source_record: dict | None = None,
) -> dict:
    bits = problem.bits
    weights = problem.weights
    mapping_error = validate_mapping(problem, args.seed + bits)

    started = time.perf_counter()
    rtn_state = round_to_nearest_state(problem)
    rtn_cost = float(hessian_cost(problem, rtn_state)[0])
    rtn_seconds = time.perf_counter() - started

    started = time.perf_counter()
    greedy = coordinate_descent(problem, rtn_state)
    greedy_incremental_seconds = time.perf_counter() - started

    started = time.perf_counter()
    exact = exact_block_optimum(problem)
    exact_seconds = time.perf_counter() - started
    if source_record is not None:
        exported_rtn = float(source_record["round_to_nearest_cost"])
        exported_exact = float(source_record["classical_exact_cost"])
        if not math.isclose(rtn_cost, exported_rtn, rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError(
                "CUDA-Q bridge changed the exported round-to-nearest cost."
            )
        if not math.isclose(exact.cost, exported_exact, rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError(
                "CUDA-Q bridge changed the exported exact adjacent optimum."
            )

    block_isings: list[IsingProblem] = []
    normalizations: list[float] = []
    optimized_parameters: list[np.ndarray] = []
    block_results: list[dict[str, Any]] = []
    blockwise_state = np.zeros(problem.size, dtype=np.float64)
    total_evaluations = 0
    total_expected_cost = 0.0
    blockwise_started = time.perf_counter()

    for block_index, start in enumerate(range(0, problem.size, REFERENCE_BLOCK_SIZE)):
        stop = start + REFERENCE_BLOCK_SIZE
        block = slice_problem(problem, start, stop)
        states = enumerate_binary(block.size)
        costs = hessian_cost(block, states)
        optimum_cost = float(costs.min())
        energy_span = float(costs.max() - optimum_cost)
        normalization = max(energy_span, np.finfo(np.float64).eps)
        ising = qubo_to_ising(block)
        hamiltonian = build_cudaq_hamiltonian(cudaq, ising, normalization)
        kernel = build_block_qaoa_kernel(cudaq, ising, normalization, args.layers)

        normalized_energy, parameters, evaluations = optimize_qaoa(
            cudaq,
            kernel,
            hamiltonian,
            layers=args.layers,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed=args.seed + bits * 1000 + block_index,
        )
        counts = cudaq.sample(kernel, parameters.tolist(), shots_count=args.block_shots)
        sampled = score_counts(block, counts, optimum_cost)
        blockwise_state[start:stop] = sampled["best_state"]
        total_evaluations += evaluations
        total_expected_cost += normalized_energy * normalization
        block_isings.append(ising)
        normalizations.append(normalization)
        optimized_parameters.append(parameters)
        block_results.append(
            {
                "block": block_index,
                "active_decisions": block.active_decisions,
                "exact_cost": optimum_cost,
                "qaoa_expectation": normalized_energy * normalization,
                "qaoa_best_bits": sampled["best_bits"],
                "qaoa_best_cost": sampled["best_cost"],
                "qaoa_sampled_mean_cost": sampled["mean_cost"],
                "qaoa_optimal_probability": sampled["optimal_probability"],
                "qaoa_unique_states": sampled["unique_states"],
                "optimizer_evaluations": evaluations,
                "parameters": parameters.tolist(),
            }
        )
        print(
            f"bits={bits} block={block_index} exact={optimum_cost:.12f} "
            f"QAOA-best={sampled['best_cost']:.12f} p(opt)={sampled['optimal_probability']:.6f} "
            f"evals={evaluations}",
            flush=True,
        )

    blockwise_cost = float(hessian_cost(problem, blockwise_state)[0])
    blockwise_seconds = time.perf_counter() - blockwise_started
    full_sampled: dict[str, Any] | None = None
    full_seconds: float | None = None
    if not args.skip_full_state:
        gc.collect()
        full_kernel = build_full_qaoa_kernel(
            cudaq, block_isings, normalizations, args.layers
        )
        flattened_parameters = np.concatenate(optimized_parameters).tolist()
        started = time.perf_counter()
        counts = cudaq.sample(
            full_kernel, flattened_parameters, shots_count=args.full_shots
        )
        full_seconds = time.perf_counter() - started
        full_sampled = score_counts(problem, counts, exact.cost)
        print(
            f"bits={bits} full32 best={full_sampled['best_cost']:.12f} "
            f"p(opt)={full_sampled['optimal_probability']:.6f} seconds={full_seconds:.3f}",
            flush=True,
        )

    exact_reduction = (rtn_cost - exact.cost) / rtn_cost
    blockwise_reduction = (rtn_cost - blockwise_cost) / rtn_cost
    rtn_mse = float(np.mean((weights - quantized_values(problem, rtn_state)) ** 2))
    exact_mse = float(np.mean((weights - quantized_values(problem, exact.state)) ** 2))
    return {
        "bits": bits,
        "group_size": problem.size,
        "active_decisions": problem.active_decisions,
        "scale": problem.scale,
        "zero": problem.zero,
        "mapping_max_abs_error": mapping_error,
        "round_to_nearest_bits": format_bits(rtn_state),
        "round_to_nearest_cost": rtn_cost,
        "round_to_nearest_seconds": rtn_seconds,
        "round_to_nearest_unweighted_mse": rtn_mse,
        "greedy_bits": format_bits(greedy.state),
        "greedy_cost": greedy.cost,
        "greedy_states_checked": greedy.states_checked,
        "greedy_seconds": rtn_seconds + greedy_incremental_seconds,
        "exact_bits": format_bits(exact.state),
        "exact_cost": exact.cost,
        "exact_unweighted_mse": exact_mse,
        "exact_states_checked": exact.states_checked,
        "exact_seconds": exact_seconds,
        "exact_hessian_cost_reduction_vs_rtn": exact_reduction,
        "classic_gptq_cost": (
            None if source_record is None else float(source_record["classic_gptq_cost"])
        ),
        "classic_gptq_seconds": (
            None
            if source_record is None
            else float(source_record["classic_gptq_seconds"])
        ),
        "qaoa_layers": args.layers,
        "qaoa_optimizer_evaluations": total_evaluations,
        "qaoa_expected_cost": total_expected_cost,
        "qaoa_blockwise_bits": format_bits(blockwise_state),
        "qaoa_blockwise_cost": blockwise_cost,
        "qaoa_blockwise_seconds": blockwise_seconds,
        "qaoa_blockwise_cost_reduction_vs_rtn": blockwise_reduction,
        "qaoa_blockwise_matches_exact": math.isclose(
            blockwise_cost, exact.cost, rel_tol=0.0, abs_tol=1e-10
        ),
        "qaoa_full32": (
            None
            if full_sampled is None
            else {
                "best_bits": full_sampled["best_bits"],
                "best_cost": full_sampled["best_cost"],
                "sampled_mean_cost": full_sampled["mean_cost"],
                "optimal_probability": full_sampled["optimal_probability"],
                "unique_states": full_sampled["unique_states"],
                "shots": full_sampled["shots"],
                "sampling_seconds": full_seconds,
                "time_to_candidate_seconds": blockwise_seconds + full_seconds,
            }
        ),
        "blocks": block_results,
    }


def print_table(results: list[dict]) -> None:
    print()
    print(
        "bits  method                  Hessian cost    reduction vs RTN    time to candidate"
    )
    print(
        "----  ----------------------  --------------  ----------------    -----------------"
    )
    for result in results:
        bits = result["bits"]
        rtn = result["round_to_nearest_cost"]
        rows = [
            (
                "round-to-nearest",
                rtn,
                result["round_to_nearest_seconds"],
            ),
            (
                "classical greedy",
                result["greedy_cost"],
                result["greedy_seconds"],
            ),
        ]
        if result["classic_gptq_cost"] is not None:
            rows.append(
                (
                    "classic GPTQ",
                    result["classic_gptq_cost"],
                    result["classic_gptq_seconds"],
                )
            )
        rows.extend(
            [
                (
                    "classical exact",
                    result["exact_cost"],
                    result["exact_seconds"],
                ),
                (
                    "CUDA-Q blockwise",
                    result["qaoa_blockwise_cost"],
                    result["qaoa_blockwise_seconds"],
                ),
            ]
        )
        full = result["qaoa_full32"]
        if full is not None:
            rows.append(
                (
                    "CUDA-Q full32 sample",
                    full["best_cost"],
                    full["time_to_candidate_seconds"],
                )
            )
        for method, cost, seconds in rows:
            reduction = (rtn - cost) / rtn
            print(
                f"{bits:>4}  {method:<22}  {cost:>14.9f}  "
                f"{reduction:>15.3%}    {seconds:>14.6f} s"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_device or "," in visible_device:
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID before importing CUDA-Q."
        )

    import cudaq

    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(args.seed)
    if cudaq.num_available_gpus() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA-Q GPU, found {cudaq.num_available_gpus()}."
        )

    metadata = gpu_metadata(visible_device)
    source_records: dict[int, dict] = {}
    if args.problem_json is None:
        weights, hessian = make_reference_group(seed=args.seed)
        problems = [
            build_adjacent_problem(weights, hessian, bits) for bits in args.bits
        ]
        reference_structure = "four independent dense 8x8 covariance blocks"
        problem_source = "NumPy reference generator"
    else:
        payload = json.loads(args.problem_json.read_text())
        if payload.get("schema") != "gptqmodel-adjacent-experiment-v1":
            raise ValueError("Unsupported GPTQ adjacent experiment schema.")
        source_records = {
            int(record["bits"]): record for record in payload.get("problems", [])
        }
        missing = sorted(set(args.bits) - source_records.keys())
        if missing:
            raise ValueError(
                f"Problem payload is missing requested bit widths: {missing}."
            )
        problems = [problem_from_payload(source_records[bits]) for bits in args.bits]
        weights = problems[0].weights
        hessian = problems[0].hessian
        for problem in problems[1:]:
            if not np.array_equal(problem.weights, weights) or not np.array_equal(
                problem.hessian, hessian
            ):
                raise ValueError(
                    "All payload bit widths must share one weight group and Hessian."
                )
        reference_structure = (
            "GPTQ-captured Hessian with four independent dense 8x8 covariance blocks"
        )
        problem_source = str(payload.get("source", args.problem_json))

    condition_number = float(np.linalg.cond(hessian))
    print("CUDA-Q group-size-32 adjacent-rounding probe")
    print(f"visible GPU       : {visible_device}")
    print(
        f"device            : {metadata['name']} ({metadata['memory_total_mib']} MiB)"
    )
    print(f"CUDA-Q / target   : {cudaq.__version__} / {cudaq.get_target().name} FP64")
    print(f"group / H / dtype : {GROUP_SIZE} / {hessian.shape} / {hessian.dtype}")
    print(f"Hessian condition : {condition_number:.6f}")
    print(f"problem source     : {problem_source}")
    print(
        f"QAOA configuration: p={args.layers}, restarts={args.restarts}, maxiter={args.maxiter}"
    )
    print(f"shots block/full  : {args.block_shots} / {args.full_shots}")
    print(flush=True)

    results = [
        run_bitwidth(cudaq, args, problem, source_records.get(problem.bits))
        for problem in problems
    ]
    print_table(results)
    output = {
        "experiment": "group32_adjacent_rounding",
        "seed": args.seed,
        "gpu": metadata,
        "cuda_visible_devices": visible_device,
        "cudaq_version": str(cudaq.__version__),
        "cudaq_target": cudaq.get_target().name,
        "statevector_precision": "complex-fp64",
        "statevector_raw_gib": 64.0,
        "group_size": GROUP_SIZE,
        "hessian_shape": list(hessian.shape),
        "hessian_dtype": str(hessian.dtype),
        "hessian_condition_number": condition_number,
        "reference_structure": reference_structure,
        "problem_source": problem_source,
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"\nJSON result: {args.json_out}")


if __name__ == "__main__":
    main()
