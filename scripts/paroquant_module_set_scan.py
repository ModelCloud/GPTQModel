#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate

from gptqmodel.utils.paroquant_benchmark import run_paroquant_selected_modules_case


DEFAULT_CASES = [
    ("attn_all", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]),
    ("mlp_all", ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]),
    ("mlp_gate_up", ["mlp.gate_proj", "mlp.up_proj"]),
    ("mlp_gate_down", ["mlp.gate_proj", "mlp.down_proj"]),
    ("mlp_up_down", ["mlp.up_proj", "mlp.down_proj"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize selected module sets inside one decoder layer and evaluate GSM8K Platinum."
    )
    parser.add_argument("--model", default="/monster/data/model/Llama-3.2-1B-Instruct")
    parser.add_argument("--layer-index", type=int, required=True, help="Zero-based decoder layer index to probe.")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=None,
        help="Case in the form label=module_a,module_b . Can be passed multiple times.",
    )
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--calibration-concat-size", type=int, default=2048)
    parser.add_argument("--quant-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-max-rows", type=int, default=None)
    parser.add_argument("--opt-rotation-epochs", type=int, default=10)
    parser.add_argument("--opt-finetune-epochs", type=int, default=10)
    parser.add_argument("--opt-train-samples", type=int, default=2048)
    parser.add_argument("--opt-validation-samples", type=int, default=64)
    parser.add_argument("--opt-batch-size", type=int, default=64)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _parse_cases(raw_cases: list[str] | None) -> list[tuple[str, list[str]]]:
    if not raw_cases:
        return [(label, list(modules)) for label, modules in DEFAULT_CASES]

    parsed: list[tuple[str, list[str]]] = []
    for item in raw_cases:
        label, sep, module_csv = str(item).partition("=")
        if not sep:
            raise ValueError(f"Invalid --case `{item}`. Expected label=module_a,module_b")
        modules = [module.strip() for module in module_csv.split(",") if module.strip()]
        if not modules:
            raise ValueError(f"Invalid --case `{item}`. At least one module is required.")
        parsed.append((label.strip(), modules))
    return parsed


def _score(case: dict) -> float | None:
    metric = case.get("eval_metrics") or case.get("metrics") or {}
    gsm = metric.get("gsm8k_platinum_cot", {})
    if not isinstance(gsm, dict):
        return None
    return float(gsm["acc,num"]) if "acc,num" in gsm else None


def main() -> int:
    args = parse_args()
    parsed_cases = _parse_cases(args.cases)

    results: list[dict] = []
    for label, module_names in parsed_cases:
        case = run_paroquant_selected_modules_case(
            model_path=args.model,
            layer_idx=args.layer_index,
            module_names=module_names,
            calibration_rows=args.calibration_rows,
            calibration_concat_size=args.calibration_concat_size,
            quant_batch_size=args.quant_batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_max_rows=args.eval_max_rows,
            sym=True,
            fused_opt_rotation=True,
            opt_rotation_epochs=args.opt_rotation_epochs,
            opt_finetune_epochs=args.opt_finetune_epochs,
            opt_train_samples=args.opt_train_samples,
            opt_validation_samples=args.opt_validation_samples,
            opt_batch_size=args.opt_batch_size,
        )
        case["label"] = label
        results.append(case)

    rows = []
    for case in results:
        rows.append(
            [
                case["label"],
                ",".join(case.get("module_names", [])),
                "" if _score(case) is None else f"{_score(case):.6f}",
                f"{float(case['quant_wall_s']):.3f}",
                f"{float(case['eval_wall_s']):.3f}",
            ]
        )
    rows.sort(key=lambda item: float(item[2] or 0.0))

    print("Summary")
    print(
        tabulate(
            rows,
            headers=["case", "module_names", "gsm8k_platinum_cot", "quant_wall_s", "eval_wall_s"],
            tablefmt="grid",
        )
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
