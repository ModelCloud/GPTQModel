# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reproduce small CPU-only solver/format feasibility measurements.

Run from the repository root: python -m scripts.prove_format_solvers
"""

import argparse
import itertools
import json
import platform
from pathlib import Path

import torch

from gptqmodel.quantization.format_reference import FORMATS, quantize_rtn
from gptqmodel.quantization.format_solver_reference import (
    SolverResult,
    allocate_rco,
    reconstruction_error,
    solve_awq,
    solve_gptq,
    solve_gsq,
)


def problem(seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(8, 64, generator=generator, dtype=torch.float64)
    mixing = torch.randn(12, 64, generator=generator, dtype=torch.float64)
    amplitudes = torch.logspace(-1, 1, 64, dtype=torch.float64)

    def sample(count):
        latent = torch.randn(count, 12, generator=generator, dtype=torch.float64)
        noise = torch.randn(count, 64, generator=generator, dtype=torch.float64)
        return (latent @ mixing + .3 * noise) * amplitudes

    return weight, sample(256), sample(512)


def run(seeds, steps):
    rows, candidates, errors = [], [], []
    for seed in seeds:
        weight, calibration, heldout = problem(seed)
        legacy = solve_gptq(weight, calibration, "int4")
        choices, losses = [], []
        for spec in FORMATS:
            results = {
                "rtn": SolverResult(quantize_rtn(weight, spec.name)),
                "gptq": solve_gptq(weight, calibration, spec.name),
                "awq": solve_awq(weight, calibration, spec.name),
                "gsq": solve_gsq(weight, calibration, spec.name, seed=seed, steps=steps),
                "int4_gptq_then_convert": SolverResult(quantize_rtn(legacy.weight.dequantize(), spec.name)),
            }
            for method, result in results.items():
                rows.append({"seed": seed, "format": spec.name, "method": method,
                             "calibration_nmse": reconstruction_error(result, weight, calibration),
                             "heldout_nmse": reconstruction_error(result, weight, heldout),
                             "weight_bits": result.weight.storage_bits,
                             "total_bits": result.storage_bits})
            choices.append(results["gptq"])
            losses.append(reconstruction_error(results["gptq"], weight, calibration))
        candidates.append(choices)
        errors.append(losses)
        del weight, calibration, heldout, legacy, results
    cheapest = sum(min(c.storage_bits for c in row) for row in candidates)
    most_expensive = sum(max(c.storage_bits for c in row) for row in candidates)
    budget = (cheapest + most_expensive) // 2
    loss_tensor = torch.tensor(errors, dtype=torch.float64)
    assignment = allocate_rco(candidates, loss_tensor, budget_bits=budget)
    used = sum(row[int(i)].storage_bits for row, i in zip(candidates, assignment))
    score = sum(errors[i][int(j)] for i, j in enumerate(assignment))
    exhaustive = min(
        sum(errors[i][j] for i, j in enumerate(choice))
        for choice in itertools.product(range(len(FORMATS)), repeat=len(seeds))
        if sum(candidates[i][j].storage_bits for i, j in enumerate(choice)) <= budget
    )
    assert abs(score - exhaustive) <= 1e-14
    return {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
            "shape": {"weight": [8, 64], "calibration": [256, 64], "heldout": [512, 64]},
            "seeds": seeds, "gsq_steps": steps, "measurements": rows,
            "rco": {"budget_bits": budget, "used_bits": used,
                    "formats": [FORMATS[int(i)].name for i in assignment],
                    "sum_calibration_nmse": score, "exhaustive_optimum": exhaustive}}


def markdown(report):
    lines = ["# CPU solver / format feasibility", "",
             "Synthetic correlated linear layers; arithmetic feasibility, not LLM quality or native-kernel qualification.",
             "Errors below are mean held-out normalized MSE across all recorded seeds (lower is better).", "",
             "| Format | RTN | GPTQ | AWQ | GSQ | INT4 GPTQ → format |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for spec in FORMATS:
        values = []
        for method in ("rtn", "gptq", "awq", "gsq", "int4_gptq_then_convert"):
            scores = [r["heldout_nmse"] for r in report["measurements"] if r["format"] == spec.name and r["method"] == method]
            values.append(f"{sum(scores) / len(scores):.6g}")
        lines.append(f"| {spec.name} | " + " | ".join(values) + " |")
    rco = report["rco"]
    lines.extend(["", f"RCO uses {rco['used_bits']} of {rco['budget_bits']} bits including format scales; "
                  f"selected formats: {', '.join(rco['formats'])}. Its additive reconstruction objective matches exhaustive enumeration.",
                  "", "AWQ includes explicit FP32 input scaling; its storage overhead is recorded in JSON. "
                  "No NVFP8 encoding is assumed. This reference does not qualify native kernels or model graphs.",
                  "", "## Run configuration", "", "```json", json.dumps({k: v for k, v in report.items() if k != "measurements"}, indent=2), "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if not 1 <= len(args.seeds) <= 5:
        parser.error("Use 1-5 seeds to bound the exhaustive RCO oracle")
    torch.set_num_threads(1)
    report = run(args.seeds, args.steps)
    output = markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
