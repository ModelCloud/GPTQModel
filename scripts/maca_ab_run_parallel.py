#!/usr/bin/env python3
"""Run MaCa A/B variants in parallel, each on its own GPU lease.

Uses `python -m gpu_allocator.cli run -n 1` to acquire one GPU per variant,
so up to 6 variants can execute concurrently when 6 GPUs are free.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = REPO_ROOT / "scripts"

SAMPLE_WEIGHTS = {
    "arc": 1172,
    "gsm8k": 1209,
    "mmlu_stem": 3153,
    "mmlu_history": 930,
    "mmlu_chemistry": 303,
}

# Summary modifiers requested by Qubitium:
# - arc-norm instead of arc-raw in summary
# - arc weight -25%, gsm8k weight +25%, each mmlu weight -15%
WEIGHT_MODIFIERS = {
    "arc": 0.75,
    "gsm8k": 1.25,
    "mmlu_stem": 0.85,
    "mmlu_history": 0.85,
    "mmlu_chemistry": 0.85,
}


def _extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    for task, vals in results.items():
        if not isinstance(vals, dict):
            continue
        if "acc,num" in vals:
            metrics[f"{task}.acc,num"] = float(vals["acc,num"])
        if "accuracy,loglikelihood" in vals:
            metrics[f"{task}.accuracy,loglikelihood"] = float(vals["accuracy,loglikelihood"])
        if "accuracy,loglikelihood_norm" in vals:
            metrics[f"{task}.accuracy,loglikelihood_norm"] = float(vals["accuracy,loglikelihood_norm"])
        if "acc,ll" in vals:
            metrics[f"{task}.acc,ll"] = float(vals["acc,ll"])
        if "acc,ll_avg" in vals:
            metrics[f"{task}.acc,ll_avg"] = float(vals["acc,ll_avg"])
    return metrics


def _result_json_path(variant: str) -> Path:
    return RESULTS_DIR / f"maca_ab_full_results_{variant}.json"


def _variant_log_path(variant: str) -> Path:
    return RESULTS_DIR / f"maca_ab_run_parallel_{variant}.log"


async def _run_variant(
    variant: str,
    reason: str,
    log_path: Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "gpu_allocator.cli",
        "run",
        "-n",
        "1",
        "--style",
        "uuid",
        "--reason",
        reason,
        "--",
        sys.executable,
        "scripts/maca_ab_test_full.py",
        "--variant",
        variant,
    ]
    with open(log_path, "w") as log_file:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=log_file,
        )
        return await proc.wait()


def _load_variant_result(variant: str) -> Optional[Dict[str, Any]]:
    path = _result_json_path(variant)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _summary_scores(metrics: Dict[str, float]) -> Dict[str, float]:
    # Per-metric columns keep raw arc for display; summary uses arc-norm.
    arc_raw = metrics.get("arc_challenge.accuracy,loglikelihood", float("nan"))
    arc_norm = metrics.get("arc_challenge.accuracy,loglikelihood_norm", float("nan"))
    gsm = metrics.get("gsm8k_platinum_cot.acc,num", float("nan"))
    stem = metrics.get("mmlu_stem.acc,ll", float("nan"))
    hist = metrics.get("mmlu_history.acc,ll", float("nan"))
    chem = metrics.get("mmlu_chemistry.acc,ll", float("nan"))

    # Equal-weight summary now uses arc-norm instead of arc-raw.
    equal = sum([arc_norm, gsm, stem, hist, chem]) / 5.0

    adjusted = {k: SAMPLE_WEIGHTS[k] * WEIGHT_MODIFIERS[k] for k in SAMPLE_WEIGHTS}
    total_w = sum(adjusted.values())
    sample_w = (
        arc_norm * adjusted["arc"]
        + gsm * adjusted["gsm8k"]
        + stem * adjusted["mmlu_stem"]
        + hist * adjusted["mmlu_history"]
        + chem * adjusted["mmlu_chemistry"]
    ) / total_w
    return {
        "arc": arc_raw,
        "arc_norm": arc_norm,
        "gsm8k": gsm,
        "mmlu_stem": stem,
        "mmlu_history": hist,
        "mmlu_chemistry": chem,
        "equal": equal,
        "sample_wtd": sample_w,
    }


def _print_table(results: List[Dict[str, Any]]) -> None:
    header = (
        "| variant | concat | length_aware | total(s) | "
        "arc acc | arc acc_norm | gsm8k plat | "
        "mmlu_stem | mmlu_history | mmlu_chemistry | equal | sample-wtd |"
    )
    print(header)
    sep = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|") if h]) + "|"
    print(sep)
    for r in results:
        metrics = r.get("results", {})
        s = _summary_scores(metrics)
        print(
            f"| {r.get('variant','?'):7} | {str(r.get('concat_size','')):6} | "
            f"{str(r.get('length_aware','')):30} | {r.get('total_time',0.0):8.1f} | "
            f"{s['arc']:.4f} | {s['arc_norm']:.4f} | "
            f"{s['gsm8k']:.4f} | {s['mmlu_stem']:.4f} | {s['mmlu_history']:.4f} | "
            f"{s['mmlu_chemistry']:.4f} | {s['equal']:.4f} | {s['sample_wtd']:.4f} |"
        )


async def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run MaCa A/B variants in parallel on separate GPUs.")
    parser.add_argument(
        "--variants",
        type=str,
        required=True,
        help="Comma-separated list of variants to run (e.g. O,P,Q).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=6,
        help="Maximum number of variants to run concurrently (default: 6).",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="MaCa parallel A/B sweep",
        help="Reason string for GPU allocator leases.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="maca_ab_full_results_parallel.json",
        help="Combined JSON output filename in scripts/.",
    )
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        print("No variants specified.", file=sys.stderr)
        return 1

    log_paths = {v: _variant_log_path(v) for v in variants}
    sem = asyncio.Semaphore(args.max_jobs)

    async def run_with_limit(variant: str) -> int:
        async with sem:
            print(f"[{variant}] acquiring GPU lease and starting...")
            rc = await _run_variant(variant, args.reason, log_paths[variant])
            status = "finished" if rc == 0 else f"failed (exit {rc})"
            print(f"[{variant}] {status}; log: {log_paths[variant]}")
            return rc

    results = await asyncio.gather(*[run_with_limit(v) for v in variants])
    all_ok = all(rc == 0 for rc in results)

    print("\n=== Collecting results ===")
    combined: List[Dict[str, Any]] = []
    for variant in variants:
        r = _load_variant_result(variant)
        if r is None:
            print(f"[{variant}] result JSON not found: {_result_json_path(variant)}")
        else:
            combined.append(r)

    if combined:
        print("\n=== A/B Summary ===")
        _print_table(combined)
        out_path = RESULTS_DIR / args.output
        out_path.write_text(json.dumps(combined, indent=2))
        print(f"\nCombined results written to {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
