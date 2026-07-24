#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare the FP64 AdjacentExact CUDA solver with RTN and production GPTQ."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import (  # noqa: E402
    adjacent_coordinate_descent,
    adjacent_dequantize,
    adjacent_exact_cuda,
    adjacent_hessian_error,
    adjacent_round_to_nearest_state,
    build_adjacent_rounding_qubo,
)
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402
from gptqmodel.utils.adjacent_exact import (  # noqa: E402
    adjacent_exact_candidates,
    prewarm_adjacent_exact_cuda,
)
from gptqmodel.utils.cpp import resolved_cuda_arch_flags  # noqa: E402


BITS = (2, 3, 4, 8)
REFERENCE_WEIGHT = (
    -0.116977,
    0.299869,
    -0.687072,
    -0.486045,
    -1.062862,
    -0.089342,
    0.077250,
    0.590128,
    -0.202627,
    0.168837,
    -0.347382,
    -0.472668,
    -0.015111,
    0.742941,
    0.232265,
    0.108646,
    -2.111298,
    1.521405,
    -2.144239,
    2.162887,
    -1.105445,
    0.087462,
    -0.085819,
    0.007499,
    0.477485,
    -0.037212,
    1.018842,
    -0.823000,
    0.138349,
    -0.048995,
    0.071816,
    -0.439631,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    size: int
    sym: bool
    activation: str
    weight: str
    coupled_block_size: int | None = None


BASELINE_SCENARIOS = (
    Scenario("diagonal_g8", 8, False, "diagonal", "normal"),
    Scenario("dense_g12", 12, False, "dense", "normal"),
    Scenario("ill_conditioned_g16", 16, True, "ill_conditioned", "normal"),
    Scenario("signed_corr_g20", 20, True, "signed_correlated", "skewed"),
    Scenario("block_corr_g32", 32, False, "block", "outlier", 8),
    Scenario("dense_corr_g32", 32, False, "dense", "outlier"),
    Scenario("gptq_win_ref_g32", 32, False, "reference_block", "reference", 8),
)

SYMMETRIC_LARGE_SCENARIOS = (
    Scenario("diagonal_sym_g64", 64, True, "diagonal", "normal"),
    Scenario("block8_sym_g64", 64, True, "block", "outlier", 8),
    Scenario("block16_sym_g64", 64, True, "block", "outlier", 16),
    Scenario("block32_sym_g64", 64, True, "block", "outlier", 32),
    Scenario("diagonal_sym_g128", 128, True, "diagonal", "normal"),
    Scenario("block8_sym_g128", 128, True, "block", "outlier", 8),
    Scenario("block16_sym_g128", 128, True, "block", "outlier", 16),
    Scenario("block32_sym_g128", 128, True, "block", "outlier", 32),
)

SCENARIO_SUITES = {
    "baseline": BASELINE_SCENARIOS,
    "sym-large": SYMMETRIC_LARGE_SCENARIOS,
    "all": BASELINE_SCENARIOS + SYMMETRIC_LARGE_SCENARIOS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--warps", type=int, default=0)
    parser.add_argument("--bits", type=int, nargs="+", default=list(BITS), choices=BITS)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=tuple(SCENARIO_SUITES),
        default="baseline",
        help="Scenario suite: the reproducible baseline, the symmetric group-64/128 sweep, or both.",
    )
    return parser.parse_args()


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _standardize_columns(values: torch.Tensor) -> torch.Tensor:
    scale = values.square().mean(dim=0).sqrt().clamp_min_(1e-5)
    return values / scale


def make_activations(
    scenario: Scenario, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    size = scenario.size
    generator = _generator(seed)
    if scenario.activation == "reference_block":
        generator = _generator(20260724)
        calibration = torch.zeros(256, size)
        heldout = torch.zeros(512, size)
        mixing_by_block = []
        for block in range(4):
            columns = slice(8 * block, 8 * (block + 1))
            rows = slice(64 * block, 64 * (block + 1))
            latent = torch.randn(64, 3, generator=generator)
            mixing = torch.randn(3, 8, generator=generator)
            mixing_by_block.append(mixing)
            calibration[rows, columns] = latent @ mixing + 0.25 * torch.randn(
                64, 8, generator=generator
            )
        for block, mixing in enumerate(mixing_by_block):
            columns = slice(8 * block, 8 * (block + 1))
            rows = slice(128 * block, 128 * (block + 1))
            heldout[rows, columns] = torch.randn(
                128, 3, generator=generator
            ) @ mixing + 0.25 * torch.randn(128, 8, generator=generator)
    elif scenario.activation == "diagonal":
        column_scale = torch.linspace(0.35, 1.8, size).sqrt()
        calibration = torch.eye(size).repeat(32, 1) * column_scale
        heldout = torch.randn(512, size, generator=generator) * column_scale
    elif scenario.activation == "block":
        block_size = scenario.coupled_block_size
        if block_size is None or size % block_size:
            raise ValueError(
                "Block scenarios require coupled_block_size to divide the group size."
            )
        blocks = size // block_size
        samples = 512
        if samples % blocks:
            raise ValueError("Calibration rows must divide evenly across blocks.")
        rows_per_block = samples // blocks
        calibration = torch.zeros(512, size)
        heldout = torch.zeros(512, size)
        for block in range(blocks):
            columns = slice(block_size * block, block_size * (block + 1))
            rows = slice(rows_per_block * block, rows_per_block * (block + 1))
            latent_size = min(3, block_size)
            mixing = torch.randn(latent_size, block_size, generator=generator)
            calibration[rows, columns] = torch.randn(
                rows_per_block, latent_size, generator=generator
            ) @ mixing + 0.2 * torch.randn(
                rows_per_block, block_size, generator=generator
            )
            heldout[rows, columns] = torch.randn(
                rows_per_block, latent_size, generator=generator
            ) @ mixing + 0.2 * torch.randn(
                rows_per_block, block_size, generator=generator
            )
    elif scenario.activation == "ill_conditioned":
        left, _ = torch.linalg.qr(torch.randn(size, size, generator=generator))
        singular = torch.logspace(0, -3, size)
        transform = left @ torch.diag(singular)
        calibration = torch.randn(512, size, generator=generator) @ transform
        heldout = torch.randn(512, size, generator=generator) @ transform
    elif scenario.activation == "signed_correlated":
        indices = torch.arange(size)
        covariance = 0.88 ** (indices[:, None] - indices[None, :]).abs()
        signs = torch.where(indices % 2 == 0, 1.0, -1.0)
        covariance *= signs[:, None] * signs[None, :]
        cholesky = torch.linalg.cholesky(covariance + 1e-4 * torch.eye(size))
        calibration = torch.randn(512, size, generator=generator) @ cholesky.mT
        heldout = torch.randn(512, size, generator=generator) @ cholesky.mT
    elif scenario.activation == "dense":
        rank = max(3, size // 4)
        mixing = torch.randn(rank, size, generator=generator)
        calibration = torch.randn(
            512, rank, generator=generator
        ) @ mixing + 0.3 * torch.randn(512, size, generator=generator)
        heldout = torch.randn(
            512, rank, generator=generator
        ) @ mixing + 0.3 * torch.randn(512, size, generator=generator)
        calibration = _standardize_columns(calibration)
        heldout = _standardize_columns(heldout)
    else:
        raise AssertionError(f"Unknown activation scenario: {scenario.activation}")
    return calibration.to(torch.float32), heldout.to(torch.float32)


def make_weight(scenario: Scenario, seed: int) -> torch.Tensor:
    if scenario.weight == "reference":
        return torch.tensor(REFERENCE_WEIGHT, dtype=torch.float32)
    generator = _generator(seed)
    weight = 0.65 * torch.randn(scenario.size, generator=generator)
    if scenario.weight == "outlier":
        weight[0] *= 7.0
        weight[scenario.size // 2] *= -5.0
    elif scenario.weight == "skewed":
        weight = torch.sign(weight) * weight.abs().square()
        weight += 0.15
    elif scenario.weight != "normal":
        raise AssertionError(f"Unknown weight scenario: {scenario.weight}")
    return weight.to(torch.float32)


def weighted_error(
    weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor
) -> float:
    error = (weight - quantized).to(torch.float64)
    return float((error @ hessian.to(torch.float64) @ error).item())


def heldout_mse(
    weight: torch.Tensor, quantized: torch.Tensor, heldout: torch.Tensor
) -> float:
    output_error = heldout.to(torch.float64) @ (weight - quantized).to(torch.float64)
    return float(output_error.square().mean().item())


def timed_cuda_call(function):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    result = function()
    end_event.record()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started, start_event.elapsed_time(end_event)


def timing_summary(wall_seconds: list[float], cuda_milliseconds: list[float]) -> dict:
    return {
        "timing_repeats": len(wall_seconds),
        "wall_seconds_samples": wall_seconds,
        "wall_seconds": statistics.median(wall_seconds),
        "wall_seconds_min": min(wall_seconds),
        "wall_seconds_max": max(wall_seconds),
        "cuda_milliseconds_samples": cuda_milliseconds,
        "cuda_milliseconds": statistics.median(cuda_milliseconds),
        "cuda_milliseconds_min": min(cuda_milliseconds),
        "cuda_milliseconds_max": max(cuda_milliseconds),
    }


def _component_sizes(problem) -> list[int]:
    active = torch.nonzero(problem.steps, as_tuple=False).flatten()
    if active.numel() == 0:
        return []
    pair = problem.pair.index_select(0, active).index_select(1, active)
    adjacency = (pair + pair.mT).ne(0).cpu()
    remaining = set(range(active.numel()))
    component_sizes = []
    while remaining:
        component_size = 0
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            component_size += 1
            for neighbor in (
                torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
            ):
                if int(neighbor) in remaining:
                    remaining.remove(int(neighbor))
                    stack.append(int(neighbor))
        component_sizes.append(component_size)
    return sorted(component_sizes, reverse=True)


def make_gptq_task(
    scenario: Scenario,
    bits: int,
    weight: torch.Tensor,
    calibration: torch.Tensor,
) -> tuple[GPTQ, torch.Tensor]:
    module = torch.nn.Linear(
        scenario.size,
        1,
        bias=False,
        dtype=torch.float32,
        device=weight.device,
    ).eval()
    module.weight.data.copy_(weight.unsqueeze(0))
    config = QuantizeConfig(
        bits=bits,
        group_size=scenario.size,
        sym=scenario.sym,
        desc_act=False,
        mse=0.0,
        scale_search=None,
    )
    task = GPTQ(module=module, qcfg=config)
    task.quantizer.configure(perchannel=True)
    task.add_batch(calibration, module(calibration))
    hessian = task.finalize_hessian(target_device=weight.device).clone()
    return task, hessian


def build_result(
    scenario: Scenario,
    bits: int,
    seed: int,
    warps: int,
    timing_repeats: int,
) -> dict:
    device = torch.device("cuda", 0)
    calibration_cpu, heldout_cpu = make_activations(scenario, seed + 17 * scenario.size)
    weight_cpu = make_weight(scenario, seed + 31 * scenario.size)
    calibration = calibration_cpu.to(device)
    heldout = heldout_cpu.to(device)
    source_weight = weight_cpu.to(device)
    task, hessian = make_gptq_task(scenario, bits, source_weight, calibration)
    weight = task.clone_module(device=device)[0]
    task.quantizer.find_params(weight.unsqueeze(0), weight=True, hessian=hessian)
    problem = build_adjacent_rounding_qubo(
        weight,
        hessian,
        scale=task.quantizer.scale[0],
        zero=task.quantizer.zero[0],
        bits=bits,
    )
    component_sizes = _component_sizes(problem)

    def rtn():
        state = adjacent_round_to_nearest_state(problem)
        cost = float(adjacent_hessian_error(problem, state)[0].item())
        return state, cost

    (rtn_state, rtn_cost), rtn_seconds, _ = timed_cuda_call(rtn)
    greedy, greedy_seconds, _ = timed_cuda_call(
        lambda: adjacent_coordinate_descent(problem, rtn_state)
    )
    exact_runs = [
        timed_cuda_call(lambda: adjacent_exact_cuda(problem, warps=warps))
        for _ in range(timing_repeats)
    ]
    exact = exact_runs[0][0]
    for repeated_exact, _, _ in exact_runs[1:]:
        if repeated_exact.cost != exact.cost or not torch.equal(
            repeated_exact.state, exact.state
        ):
            raise AssertionError(
                "AdjacentExact CUDA returned a non-deterministic optimum."
            )
    exact_timing = timing_summary(
        [run[1] for run in exact_runs],
        [run[2] for run in exact_runs],
    )
    exact_quantized = adjacent_dequantize(problem, exact.state, dtype=weight.dtype)
    rtn_quantized = adjacent_dequantize(problem, rtn_state, dtype=weight.dtype)
    greedy_quantized = adjacent_dequantize(problem, greedy.state, dtype=weight.dtype)

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
    winner = "adjacent_exact" if exact.cost <= classic_cost else "classic_gptq"
    eigenvalues = torch.linalg.eigvalsh(hessian.to(torch.float64))
    positive = eigenvalues[eigenvalues > 1e-14]
    condition = (
        float((positive[-1] / positive[0]).item()) if positive.numel() else float("inf")
    )

    methods = {
        "rtn": {
            "hessian_error": rtn_cost,
            "heldout_mse": heldout_mse(weight, rtn_quantized, heldout),
            "weight_mse": float((weight - rtn_quantized).square().mean().item()),
            "wall_seconds": rtn_seconds,
        },
        "adjacent_greedy": {
            "hessian_error": greedy.cost,
            "heldout_mse": heldout_mse(weight, greedy_quantized, heldout),
            "weight_mse": float((weight - greedy_quantized).square().mean().item()),
            "wall_seconds": greedy_seconds,
            "states_checked": greedy.states_checked,
        },
        "adjacent_exact_cuda": {
            "hessian_error": exact.cost,
            "heldout_mse": heldout_mse(weight, exact_quantized, heldout),
            "weight_mse": float((weight - exact_quantized).square().mean().item()),
            "states_checked": exact.states_checked,
            **exact_timing,
        },
        "classic_gptq": {
            "hessian_error": classic_cost,
            "heldout_mse": heldout_mse(weight, classic_quantized, heldout),
            "weight_mse": float((weight - classic_quantized).square().mean().item()),
            **classic_timing,
        },
        "hybrid_best": {
            "hessian_error": min(exact.cost, classic_cost),
            "winner": winner,
        },
    }
    return {
        "scenario": asdict(scenario),
        "bits": bits,
        "dtype": "float32 weights/calibration, float64 QUBO and objective",
        "active_decisions": problem.active_decisions,
        "interaction_components": len(component_sizes),
        "interaction_component_sizes": component_sizes,
        "largest_interaction_component": max(component_sizes, default=0),
        "logical_state_space": 1 << problem.active_decisions,
        "hessian_condition": condition,
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
        "methods": methods,
    }


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return ((result.stdout or "") + (result.stderr or "")).strip()


def hardware_metadata() -> dict:
    properties = torch.cuda.get_device_properties(0)
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes_at_start": free_memory,
        "cuda_mem_get_info_total_bytes": total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[0],
        "nvcc": _command_output(["nvcc", "--version"]),
        "resolved_cuda_arch_flags": resolved_cuda_arch_flags(),
        "jit_cuda_flags": [
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-Xptxas=-O3,-dlcm=ca",
            "no --use_fast_math",
        ],
    }


def warmup() -> None:
    prewarm_adjacent_exact_cuda()
    size = 16
    constant = torch.zeros((), device="cuda", dtype=torch.float64)
    linear = torch.linspace(-0.2, 0.2, size, device="cuda", dtype=torch.float64)
    interaction = torch.zeros(size, size, device="cuda", dtype=torch.float64)
    for _ in range(10):
        adjacent_exact_candidates(constant, linear, interaction)
    torch.cuda.synchronize()


def print_table(results: list[dict]) -> None:
    print(
        "scenario                 b sym grp act comp max | RTN error   greedy      AdjExact    GPTQ        "
        "| winner         Adj/GPTQ exact ms GPTQ ms"
    )
    print(
        "------------------------ - --- --- --- ---- --- | ----------- ----------- ----------- ----------- "
        "| -------------- -------- -------- -------"
    )
    for result in results:
        scenario = result["scenario"]
        methods = result["methods"]
        adjacent = methods["adjacent_exact_cuda"]
        classic = methods["classic_gptq"]
        ratio = (
            adjacent["hessian_error"] / classic["hessian_error"]
            if classic["hessian_error"]
            else 1.0
        )
        print(
            f"{scenario['name']:<24} {result['bits']:>1} {str(scenario['sym']):>3} "
            f"{scenario['size']:>3} {result['active_decisions']:>3} {result['interaction_components']:>4} "
            f"{result['largest_interaction_component']:>3} | "
            f"{methods['rtn']['hessian_error']:>11.4e} "
            f"{methods['adjacent_greedy']['hessian_error']:>11.4e} "
            f"{adjacent['hessian_error']:>11.4e} {classic['hessian_error']:>11.4e} | "
            f"{methods['hybrid_best']['winner']:<14} {ratio:>8.4f} "
            f"{adjacent['wall_seconds'] * 1e3:>8.3f} {classic['wall_seconds'] * 1e3:>7.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be positive.")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device.startswith("GPU-") or "," in visible_device:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU.")

    scenarios = SCENARIO_SUITES[args.suite]
    metadata = hardware_metadata()
    warmup()
    # Warm both high-level AdjacentExact orchestration and the production GPTQ
    # loop before collecting the one-shot time-to-solution rows.
    build_result(scenarios[0], args.bits[0], args.seed - 1, args.warps, 1)
    results = []
    for scenario_index, scenario in enumerate(scenarios):
        for bits in args.bits:
            print(f"running scenario={scenario.name} bits={bits}", flush=True)
            results.append(
                build_result(
                    scenario,
                    bits,
                    args.seed + 1009 * scenario_index,
                    args.warps,
                    args.timing_repeats,
                )
            )

    output = {
        "schema": "gptqmodel-adjacent-exact-benchmark-v2",
        "seed": args.seed,
        "bits": args.bits,
        "suite": args.suite,
        "hardware": metadata,
        "solver": {
            "backend": "custom CUDA FP64 exhaustive Gray-code QUBO",
            "warps": args.warps,
            "timing_repeats": args.timing_repeats,
            "decompose_exact_zero_couplings": True,
            "selection_rule": "hybrid_best chooses lower raw Hessian error per row-group",
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print_table(results)
    print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()
