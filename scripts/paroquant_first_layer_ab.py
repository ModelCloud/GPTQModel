#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

from tabulate import tabulate

from gptqmodel.utils.paroquant_benchmark import (
    comparison_rows,
    render_case_tables,
    run_fp16_eval,
    run_paroquant_first_layer_case,
    write_case_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark first-N-layer ParoQuant on Llama-3.2-1B-Instruct and evaluate GSM8K Platinum."
    )
    parser.add_argument("--model", default="/monster/data/model/Llama-3.2-1B-Instruct")
    parser.add_argument("--quant-layers", type=int, default=1, help="Quantize the first N decoder layers.")
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--calibration-concat-size", type=int, default=2048)
    parser.add_argument("--quant-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-max-rows", type=int, default=None)
    parser.add_argument(
        "--sym",
        action="store_true",
        default=True,
        help="ParoQuant is sym-only; this flag is kept for compatibility and has no effect.",
    )
    parser.add_argument(
        "--no-fused-opt-rotation",
        action="store_true",
        help="Disable the fused CUDA rotation autograd path during ParoQuant optimization.",
    )
    parser.add_argument(
        "--opt-scope",
        choices=("module", "compute_block", "layer"),
        default="module",
        help="ParoQuant optimization scope for the selected decoder layers.",
    )
    parser.add_argument("--opt-rotation-epochs", type=int, default=10)
    parser.add_argument("--opt-finetune-epochs", type=int, default=10)
    parser.add_argument("--opt-train-samples", type=int, default=2048)
    parser.add_argument("--opt-validation-samples", type=int, default=64)
    parser.add_argument("--opt-batch-size", type=int, default=64)
    parser.add_argument("--skip-fp16-baseline", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the quantized-case result payload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = None
    if not args.skip_fp16_baseline:
        baseline = run_fp16_eval(
            model_path=args.model,
            eval_batch_size=args.eval_batch_size,
            eval_max_rows=args.eval_max_rows,
        )
        baseline["label"] = "fp16_baseline"

    quant_case = run_paroquant_first_layer_case(
        model_path=args.model,
        num_quant_layers=args.quant_layers,
        calibration_rows=args.calibration_rows,
        calibration_concat_size=args.calibration_concat_size,
        quant_batch_size=args.quant_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_max_rows=args.eval_max_rows,
        sym=args.sym,
        fused_opt_rotation=not args.no_fused_opt_rotation,
        opt_scope=args.opt_scope,
        opt_rotation_epochs=args.opt_rotation_epochs,
        opt_finetune_epochs=args.opt_finetune_epochs,
        opt_train_samples=args.opt_train_samples,
        opt_validation_samples=args.opt_validation_samples,
        opt_batch_size=args.opt_batch_size,
    )
    quant_case["label"] = (
        "paroquant_first_layer" if args.quant_layers == 1 else f"paroquant_first_{args.quant_layers}_layers"
    )

    if args.output_json is not None:
        write_case_json(quant_case, args.output_json)

    cases = [case for case in (baseline, quant_case) if case is not None]
    print("Summary")
    print(
        tabulate(
            comparison_rows(*cases),
            headers=["case", "opt_scope", "sym", "fused_opt", "gsm8k_platinum_cot", "quant_wall_s", "eval_wall_s"],
            tablefmt="grid",
        )
    )

    tables = render_case_tables(quant_case)
    print()
    print("Quant Module Times")
    print(tables["module_times"])
    print()
    print("Quant Region Times")
    print(tables["regions"])
    print()
    print("Kernel Parity And Speed")
    print(tables["kernels"])
    print()
    print("Eval")
    print(tables["eval"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
