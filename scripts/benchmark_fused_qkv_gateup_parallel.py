#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Run `benchmark_fused_qkv_gateup.py` across GPUs 4/5/6 in parallel.

One model shape is pinned to each GPU so iterations are not serialized:
    GPU 4 -> Laguna-S-2.1
    GPU 5 -> Qwen3.5-27B
    GPU 6 -> Llama-like

Example:
    python scripts/benchmark_fused_qkv_gateup_parallel.py --iters 50 --warmup 10
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


# Map each model shape to a physical GPU. Each subprocess sees its assigned
# GPU as `cuda:0` thanks to CUDA_VISIBLE_DEVICES.
MODEL_GPU_ASSIGNMENTS = [
    ("Laguna-S-2.1", "4"),
    ("Qwen3.5-27B", "5"),
    ("Llama-like", "6"),
]


def _print_table(rows: List[dict]) -> None:
    header = f"{'op':>8} {'hidden':>6} {'out':>18} {'batch':>6} {'seq':>4} {'ms_sep':>10} {'ms_fused':>10} {'speedup':>8} {'tok/s_sep':>12} {'tok/s_fused':>14} {'max_diff':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        out_str = str(r["out"])
        print(
            f"{r['op']:>8} {r['hidden']:>6} {out_str:>18} {r['batch']:>6} {r['seq_len']:>4} "
            f"{r['ms_sep']:>10.3f} {r['ms_fused']:>10.3f} {r['speedup']:>8.3f} "
            f"{r['throughput_sep']:>12.1f} {r['throughput_fused']:>14.1f} {r['max_diff']:>10.4f}"
        )


def _run_worker(
    model: str,
    gpu: str,
    warmup: int,
    iters: int,
    output_path: Path,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:1024"

    cmd = [
        sys.executable,
        "scripts/benchmark_fused_qkv_gateup.py",
        "--models",
        model,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--output",
        str(output_path),
    ]
    print(f"[GPU {gpu}] starting {model} -> {output_path}")
    result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
    return result.returncode


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=str, default=None, help="Optional combined JSON output file")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_paths = [Path(tmpdir) / f"{model.replace(' ', '_')}.json" for model, _ in MODEL_GPU_ASSIGNMENTS]
        procs = []
        for (model, gpu), out_path in zip(MODEL_GPU_ASSIGNMENTS, output_paths):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/benchmark_fused_qkv_gateup.py",
                    "--models",
                    model,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--output",
                    str(out_path),
                ],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:1024",
                },
                cwd=Path(__file__).parent.parent,
            )
            procs.append(proc)

        all_rows: List[dict] = []
        failed = False
        for proc, out_path, (model, _) in zip(procs, output_paths, MODEL_GPU_ASSIGNMENTS):
            rc = proc.wait()
            if rc != 0:
                print(f"Benchmark failed for {model} (rc={rc})", file=sys.stderr)
                failed = True
                continue
            rows = json.loads(out_path.read_text())
            all_rows.extend(rows)

        if failed:
            return 1

        # Sort rows for stable presentation (shape order, then op, then batch/seq).
        all_rows.sort(key=lambda r: (str(r.get("hidden")), r.get("op"), r.get("batch"), r.get("seq_len")))

        print()
        _print_table(all_rows)

        if args.output:
            Path(args.output).write_text(json.dumps(all_rows, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
