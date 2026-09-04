#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare separate down-recovery cast with a direct FP16 store on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

M_VALUES = (1, 2, 4, 8, 16)
N = 2048
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("scripts/benchmark_qvq_phase19_fp16_recovery_store.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/a41_phase19_h100/fp16_recovery_store_experiment.json"
        ),
    )
    args = parser.parse_args()
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.utils.qvq_cuda import qvq_cuda_hadamard

    device_info = common._assert_h100(torch)
    fingerprint = _source_fingerprint()
    rows = []
    for m in args.m_values:
        generator = torch.Generator(device="cuda").manual_seed(19100 + m)
        inner = torch.randn(
            (m, N), generator=generator, device="cuda", dtype=torch.float32
        )
        post_scale = torch.randn(
            (N,), generator=generator, device="cuda", dtype=torch.float32
        )
        bias = torch.randn(
            (N,), generator=generator, device="cuda", dtype=torch.float32
        )

        def separate_cast():
            return qvq_cuda_hadamard(
                inner,
                post_scale=post_scale,
                bias=bias,
                scale_mode=3,
            ).to(torch.float16)

        def direct_store():
            return qvq_cuda_hadamard(
                inner,
                post_scale=post_scale,
                bias=bias,
                scale_mode=3,
                output_fp16=True,
            )

        baseline_timing, baseline_output = common._graph_timing(
            torch,
            separate_cast,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        direct_timing, direct_output = common._graph_timing(
            torch,
            direct_store,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        bit_exact = all(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            for actual, expected in zip(
                direct_output, baseline_output, strict=True
            )
        )
        if not bit_exact:
            raise RuntimeError(f"direct FP16 store changed M{m} output bits")
        speedup = baseline_timing["median_ms"] / direct_timing["median_ms"]
        row = {
            "m": m,
            "n": N,
            "separate_cast": baseline_timing,
            "direct_fp16_store": direct_timing,
            "speedup_vs_separate_cast": speedup,
            "better_than_separate_cast": speedup > 1.0,
            "bit_exact": bit_exact,
            "intermediate_fp32_bytes_eliminated": m * N * 4,
        }
        rows.append(row)
        print(
            f"M{m}: separate={baseline_timing['median_ms'] * 1000:.3f}us "
            f"direct={direct_timing['median_ms'] * 1000:.3f}us "
            f"speedup={speedup:.3f}x better={row['better_than_separate_cast']}",
            flush=True,
        )

    if fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark sources changed while the H100 matrix was running")
    payload = {
        "git_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "host_launch_gaps_included": False,
        },
        "workload": "exact FP32 down recovery followed by final FP16 store",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}")
    return payload


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
