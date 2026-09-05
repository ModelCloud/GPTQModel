#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Run the gfx950 P32 benchmark over Qwen3.8-27B linear projection shapes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTED_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
P32_RATES = (2.0, 2.5, 3.0, 3.5)
QWEN38_27B_SHAPES = (
    ("full_q_gate", 5120, 12288),
    ("full_kv", 5120, 1024),
    ("attn_out", 6144, 5120),
    ("linear_qkv", 5120, 10240),
    ("linear_z", 5120, 6144),
    ("mlp_gate_up", 5120, 17408),
    ("mlp_down", 17408, 5120),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository revision whose benchmark and kernel should be measured.",
    )
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=P32_RATES)
    parser.add_argument("--m-values", type=int, nargs="+", default=REQUESTED_M)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--fallback-warmup", type=int, default=1)
    parser.add_argument("--fallback-iterations", type=int, default=2)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument(
        "--shape-cooldown",
        type=float,
        default=3.0,
        help="Seconds to wait for ROCm to release the previous shape's allocations.",
    )
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild the aggregate from existing per-shape JSON files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-shape files only when they are marked benchmark-valid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_gfx950.json"),
    )
    args = parser.parse_args()
    if any(rate not in P32_RATES for rate in args.rates):
        parser.error("--rates supports only W2, W2.5, W3, and W3.5")
    if any(m not in REQUESTED_M for m in args.m_values):
        parser.error(f"--m-values must be drawn from {REQUESTED_M}")
    if args.shape_cooldown < 0:
        parser.error("--shape-cooldown must be nonnegative")
    return args


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    benchmark = repo_root / "scripts" / "benchmark_qvq_p32_amd.py"
    if not benchmark.is_file():
        raise FileNotFoundError(f"benchmark script not found: {benchmark}")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    component_dir = args.output.parent / f"{args.output.stem}_shapes"
    component_dir.mkdir(parents=True, exist_ok=True)
    components = []
    rows = []

    for shape_index, (name, k, n) in enumerate(QWEN38_27B_SHAPES):
        output = component_dir / f"{name}.json"
        command = [
            sys.executable,
            str(benchmark),
            "--physical-gpu",
            str(args.physical_gpu),
            "--rates",
            *(str(rate) for rate in args.rates),
            "--m-values",
            *(str(m) for m in args.m_values),
            "--k",
            str(k),
            "--n",
            str(n),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--fallback-warmup",
            str(args.fallback_warmup),
            "--fallback-iterations",
            str(args.fallback_iterations),
            "--idle-samples",
            str(args.idle_samples),
            "--idle-interval",
            str(args.idle_interval),
            "--idle-memory-tolerance-mib",
            str(args.idle_memory_tolerance_mib),
            "--output",
            str(output),
        ]
        if args.allow_busy:
            command.append("--allow-busy")
        reusable = False
        if args.resume and output.is_file():
            reusable = bool(json.loads(output.read_text()).get("benchmark_valid"))
        if not args.aggregate_only and not reusable:
            if shape_index and args.shape_cooldown:
                time.sleep(args.shape_cooldown)
            print(f"running Qwen3.8-27B {name}: K={k} N={n}", flush=True)
            subprocess.run(
                command,
                cwd=repo_root,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                check=True,
            )
        elif reusable:
            print(f"reusing valid Qwen3.8-27B {name}: K={k} N={n}", flush=True)
        elif not output.is_file():
            raise FileNotFoundError(f"component benchmark not found: {output}")
        component = json.loads(output.read_text())
        components.append(
            {
                "name": name,
                "k": k,
                "n": n,
                "path": str(output.relative_to(args.output.parent)),
                "benchmark_valid": component["benchmark_valid"],
            }
        )
        for row in component["rows"]:
            row["shape"] = name
            rows.append(row)

    result = {
        "schema": "qvq_p32_qwen38_27b_gfx950_benchmark_v1",
        "model": "Qwen/Qwen3.8-27B",
        "benchmark_valid": all(
            component["benchmark_valid"] for component in components
        ),
        "accuracy_gate": 2e-3,
        "requested_m": list(args.m_values),
        "rates": list(args.rates),
        "components": components,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
