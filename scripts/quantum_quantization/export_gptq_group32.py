#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Export real GPTQ row-group QUBOs for the isolated CUDA-Q environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.adjacent import (  # noqa: E402
    adjacent_coordinate_descent,
    adjacent_exact_blocks,
    adjacent_hessian_error,
    adjacent_problem_payload,
    adjacent_round_to_nearest_state,
    build_adjacent_rounding_qubo,
)
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def make_calibration(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    calibration = torch.zeros(256, 32, dtype=torch.float32)
    for block in range(4):
        rows = slice(64 * block, 64 * (block + 1))
        columns = slice(8 * block, 8 * (block + 1))
        latent = torch.randn(64, 3, generator=generator)
        mixing = torch.randn(3, 8, generator=generator)
        calibration[rows, columns] = latent @ mixing + 0.25 * torch.randn(
            64, 8, generator=generator
        )
    return calibration


def weighted_error(
    weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor
) -> float:
    error = (weight - quantized).to(torch.float64)
    return float((error @ hessian.to(torch.float64) @ error).item())


def build_problem(bits: int, calibration: torch.Tensor, device: torch.device) -> dict:
    config = QuantizeConfig(
        bits=bits,
        group_size=32,
        sym=False,
        desc_act=False,
        mse=0.0,
        scale_search=None,
    )
    module = torch.nn.Linear(
        32, 1, bias=False, dtype=torch.float32, device=device
    ).eval()
    module.weight.data.copy_(
        torch.tensor(REFERENCE_WEIGHT, dtype=torch.float32, device=device).unsqueeze(0)
    )
    task = GPTQ(module=module, qcfg=config)
    task.quantizer.configure(perchannel=True)

    activations = calibration.to(device=device)
    task.add_batch(activations, module(activations))
    hessian = task.finalize_hessian(target_device=device).clone()
    weight = task.clone_module(device=device)
    task.quantizer.find_params(weight, weight=True, hessian=hessian)
    problem = build_adjacent_rounding_qubo(
        weight[0],
        hessian,
        scale=task.quantizer.scale[0],
        zero=task.quantizer.zero[0],
        bits=bits,
    )

    rtn_state = adjacent_round_to_nearest_state(problem)
    rtn_cost = float(adjacent_hessian_error(problem, rtn_state)[0].item())
    greedy = adjacent_coordinate_descent(problem, rtn_state)
    exact = adjacent_exact_blocks(problem)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    classic_quantized, classic_scales, classic_zeros, classic_g_idx, *_ = task.quantize(
        blocksize=32
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    classic_seconds = time.perf_counter() - started
    classic_cost = weighted_error(weight[0], classic_quantized[0], hessian)
    payload = adjacent_problem_payload(problem)
    payload.update(
        {
            "round_to_nearest_state": rtn_state.to(torch.int64).cpu().tolist(),
            "round_to_nearest_cost": rtn_cost,
            "classical_greedy_state": greedy.state.to(torch.int64).cpu().tolist(),
            "classical_greedy_cost": greedy.cost,
            "classical_exact_state": exact.state.to(torch.int64).cpu().tolist(),
            "classical_exact_cost": exact.cost,
            "classical_exact_states_checked": exact.states_checked,
            "classic_gptq_cost": classic_cost,
            "classic_gptq_seconds": classic_seconds,
            "classic_gptq_scale": classic_scales.detach()
            .to(torch.float64)
            .cpu()
            .reshape(-1)
            .tolist(),
            "classic_gptq_zero": classic_zeros.detach()
            .to(torch.float64)
            .cpu()
            .reshape(-1)
            .tolist(),
            "classic_gptq_g_idx": classic_g_idx.detach().cpu().tolist(),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible_device or "," in visible_device:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID.")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA GPU.")
        device = torch.device("cuda", 0)
    else:
        visible_device = None
        device = torch.device("cpu")

    calibration = make_calibration(args.seed)
    problems = [build_problem(bits, calibration, device) for bits in (2, 3)]
    output = {
        "schema": "gptqmodel-adjacent-experiment-v1",
        "seed": args.seed,
        "source": "GPTQ.add_batch float32 Hessian and Quantizer.find_params",
        "device": str(device),
        "cuda_visible_devices": visible_device,
        "group_size": 32,
        "sym": False,
        "problems": problems,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print("bits  RTN error       greedy error    adjacent exact  classic GPTQ")
    print("----  --------------  --------------  --------------  --------------")
    for problem in problems:
        print(
            f"{problem['bits']:>4}  {problem['round_to_nearest_cost']:>14.9f}  "
            f"{problem['classical_greedy_cost']:>14.9f}  {problem['classical_exact_cost']:>14.9f}  "
            f"{problem['classic_gptq_cost']:>14.9f}"
        )
    print(f"JSON problem: {args.json_out}")


if __name__ == "__main__":
    main()
