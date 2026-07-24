#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native dense 64/128-variable AdjacentExact branch-and-bound."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import (  # noqa: E402
    adjacent_branch_bound_cuda,
    adjacent_dequantize,
    adjacent_hessian_error,
    adjacent_round_to_nearest_state,
    build_adjacent_rounding_qubo,
)
from gptqmodel.utils.adjacent_exact import prewarm_adjacent_exact_cuda  # noqa: E402
from scripts.quantum_quantization.benchmark_adjacent_exact import (  # noqa: E402
    BITS,
    Scenario,
    hardware_metadata,
    heldout_mse,
    make_activations,
    make_gptq_task,
    make_weight,
    timed_cuda_call,
    timing_summary,
    weighted_error,
)


SCENARIOS = (
    Scenario("dense_sym_g64", 64, True, "dense", "outlier"),
    Scenario("signed_corr_sym_g64", 64, True, "signed_correlated", "skewed"),
    Scenario("ill_conditioned_sym_g64", 64, True, "ill_conditioned", "normal"),
    Scenario("dense_sym_g128", 128, True, "dense", "outlier"),
    Scenario("signed_corr_sym_g128", 128, True, "signed_correlated", "skewed"),
    Scenario("ill_conditioned_sym_g128", 128, True, "ill_conditioned", "normal"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bits", type=int, nargs="+", default=list(BITS), choices=BITS)
    parser.add_argument("--split-depth", type=int, default=8)
    parser.add_argument("--max-nodes-per-worker", type=int, default=2000)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--certificate-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def build_result(
    scenario: Scenario,
    bits: int,
    seed: int,
    *,
    split_depth: int,
    max_nodes_per_worker: int,
    timing_repeats: int,
    certificate_tolerance: float,
) -> dict:
    device = torch.device("cuda", 0)
    calibration_cpu, heldout_cpu = make_activations(
        scenario, seed + 17 * scenario.size
    )
    weight_cpu = make_weight(scenario, seed + 31 * scenario.size)
    calibration = calibration_cpu.to(device)
    heldout = heldout_cpu.to(device)
    source_weight = weight_cpu.to(device)
    task, hessian = make_gptq_task(
        scenario, bits, source_weight, calibration
    )
    weight = task.clone_module(device=device)[0]
    task.quantizer.find_params(weight.unsqueeze(0), weight=True, hessian=hessian)
    problem = build_adjacent_rounding_qubo(
        weight,
        hessian,
        scale=task.quantizer.scale[0],
        zero=task.quantizer.zero[0],
        bits=bits,
    )

    def rtn():
        state = adjacent_round_to_nearest_state(problem)
        cost = float(adjacent_hessian_error(problem, state)[0].item())
        return state, cost

    (rtn_state, rtn_cost), rtn_seconds, rtn_cuda_ms = timed_cuda_call(rtn)
    native_runs = [
        timed_cuda_call(
            lambda: adjacent_branch_bound_cuda(
                problem,
                split_depth=split_depth,
                max_nodes_per_worker=max_nodes_per_worker,
                certificate_tolerance=certificate_tolerance,
                require_optimal=False,
            )
        )
        for _ in range(timing_repeats)
    ]
    best_native = min(
        (run[0] for run in native_runs),
        key=lambda result: result.cost,
    )
    native_timing = timing_summary(
        [run[1] for run in native_runs],
        [run[2] for run in native_runs],
    )
    native_quantized = adjacent_dequantize(
        problem, best_native.state, dtype=weight.dtype
    )
    rtn_quantized = adjacent_dequantize(problem, rtn_state, dtype=weight.dtype)

    classic_tasks = [task]
    for _ in range(timing_repeats - 1):
        repeated_task, repeated_hessian = make_gptq_task(
            scenario, bits, source_weight, calibration
        )
        torch.testing.assert_close(repeated_hessian, hessian, rtol=0.0, atol=0.0)
        classic_tasks.append(repeated_task)
    classic_runs = [
        timed_cuda_call(
            lambda classic_task=classic_task: classic_task.quantize(
                blocksize=scenario.size
            )
        )
        for classic_task in classic_tasks
    ]
    classic_result = classic_runs[0][0]
    classic_timing = timing_summary(
        [run[1] for run in classic_runs],
        [run[2] for run in classic_runs],
    )
    classic_quantized, classic_scales, classic_zeros, *_ = classic_result
    classic_quantized = classic_quantized[0]
    classic_cost = weighted_error(weight, classic_quantized, hessian)

    return {
        "scenario": {
            "name": scenario.name,
            "size": scenario.size,
            "sym": scenario.sym,
            "activation": scenario.activation,
            "weight": scenario.weight,
        },
        "bits": bits,
        "dtype": "float32 weights/calibration, float64 QUBO and objective",
        "active_decisions": problem.active_decisions,
        "scale": float(problem.scale.item()),
        "zero": float(problem.zero.item()),
        "classic_scale_matches": bool(
            torch.allclose(
                classic_scales.reshape(-1)[0].to(torch.float64),
                problem.scale,
                rtol=1e-6,
                atol=1e-12,
            )
        ),
        "classic_zero_matches": bool(
            torch.allclose(
                classic_zeros.reshape(-1)[0].to(torch.float64),
                problem.zero,
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "methods": {
            "rtn": {
                "hessian_error": rtn_cost,
                "heldout_mse": heldout_mse(weight, rtn_quantized, heldout),
                "weight_mse": float((weight - rtn_quantized).square().mean().item()),
                "wall_seconds": rtn_seconds,
                "cuda_milliseconds": rtn_cuda_ms,
            },
            "adjacent_native": {
                "hessian_error": best_native.cost,
                "heldout_mse": heldout_mse(weight, native_quantized, heldout),
                "weight_mse": float(
                    (weight - native_quantized).square().mean().item()
                ),
                "optimal": best_native.optimal,
                "lower_bound": best_native.lower_bound,
                "optimality_gap": (
                    best_native.cost - best_native.lower_bound
                    if best_native.lower_bound is not None
                    else None
                ),
                "nodes_visited": best_native.nodes_visited,
                "cost_samples": [run[0].cost for run in native_runs],
                "optimal_samples": [run[0].optimal for run in native_runs],
                "lower_bound_samples": [
                    run[0].lower_bound for run in native_runs
                ],
                "nodes_visited_samples": [
                    run[0].nodes_visited for run in native_runs
                ],
                **native_timing,
            },
            "classic_gptq": {
                "hessian_error": classic_cost,
                "heldout_mse": heldout_mse(weight, classic_quantized, heldout),
                "weight_mse": float(
                    (weight - classic_quantized).square().mean().item()
                ),
                **classic_timing,
            },
        },
    }


def print_table(results: list[dict]) -> None:
    print(
        "scenario                    b   g act | RTN error   Native B&B  GPTQ        "
        "| N/GPTQ cert nodes       native ms GPTQ ms"
    )
    print(
        "--------------------------- - --- --- | ----------- ----------- ----------- "
        "| ------ ---- ----------- --------- -------"
    )
    for result in results:
        scenario = result["scenario"]
        methods = result["methods"]
        native = methods["adjacent_native"]
        classic = methods["classic_gptq"]
        ratio = native["hessian_error"] / classic["hessian_error"]
        certificate = "yes" if native["optimal"] else "no"
        print(
            f"{scenario['name']:<27} {result['bits']:>1} {scenario['size']:>3} "
            f"{result['active_decisions']:>3} | "
            f"{methods['rtn']['hessian_error']:>11.4e} "
            f"{native['hessian_error']:>11.4e} "
            f"{classic['hessian_error']:>11.4e} | "
            f"{ratio:>6.3f} {certificate:>4} {native['nodes_visited']:>11} "
            f"{native['wall_seconds'] * 1e3:>9.3f} "
            f"{classic['wall_seconds'] * 1e3:>7.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive.")
    if args.max_nodes_per_worker < 1:
        raise ValueError("--max-nodes-per-worker must be positive.")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.startswith("GPU-") or "," in visible_device:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU.")

    metadata = hardware_metadata()
    prewarm_adjacent_exact_cuda()
    results = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for bits in args.bits:
            print(f"running scenario={scenario.name} bits={bits}", flush=True)
            results.append(
                build_result(
                    scenario,
                    bits,
                    args.seed + 1009 * scenario_index,
                    split_depth=args.split_depth,
                    max_nodes_per_worker=args.max_nodes_per_worker,
                    timing_repeats=args.timing_repeats,
                    certificate_tolerance=args.certificate_tolerance,
                )
            )

    output = {
        "schema": "gptqmodel-adjacent-native-benchmark-v1",
        "seed": args.seed,
        "bits": args.bits,
        "hardware": metadata,
        "solver": {
            "backend": "custom CUDA FP64 128-bit branch-and-bound QUBO",
            "split_depth": args.split_depth,
            "max_nodes_per_worker": args.max_nodes_per_worker,
            "timing_repeats": args.timing_repeats,
            "certificate_tolerance": args.certificate_tolerance,
            "selection_rule": "best native candidate across timing repeats",
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print_table(results)
    print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
