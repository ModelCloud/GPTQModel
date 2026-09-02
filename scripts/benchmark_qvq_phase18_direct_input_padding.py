#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare separate M16 input padding with direct padded Hadamard on H100."""

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
K = 2048
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("scripts/benchmark_qvq_phase18_direct_input_padding.py"),
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
            "artifacts/a41_phase18_h100/direct_input_padding_experiment.json"
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
        generator = torch.Generator(device="cuda").manual_seed(18000 + m)
        x = torch.randn(
            (m, K), generator=generator, device="cuda", dtype=torch.float16
        )
        pre_scale = torch.randn(
            (K,), generator=generator, device="cuda", dtype=torch.float16
        )

        def separate_padding():
            transformed = qvq_cuda_hadamard(
                x, pre_scale=pre_scale, scale_mode=2
            )
            if m == 16:
                return transformed
            padded = torch.zeros((16, K), dtype=transformed.dtype, device="cuda")
            padded[:m].copy_(transformed)
            return padded

        def direct_padding():
            return qvq_cuda_hadamard(
                x,
                pre_scale=pre_scale,
                scale_mode=2,
                pad_to_16=m < 16,
            )

        baseline_timing, baseline_output = common._graph_timing(
            torch,
            separate_padding,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        direct_timing, direct_output = common._graph_timing(
            torch,
            direct_padding,
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
            raise RuntimeError(f"direct input padding changed M{m} output bits")
        speedup = baseline_timing["median_ms"] / direct_timing["median_ms"]
        row = {
            "m": m,
            "k": K,
            "separate_padding": baseline_timing,
            "direct_padding": direct_timing,
            "speedup_vs_separate_padding": speedup,
            "better_than_separate_padding": speedup > 1.0,
            "bit_exact": bit_exact,
            "eliminated_transient_bytes": 0 if m == 16 else m * K * 2,
        }
        rows.append(row)
        print(
            f"M{m}: separate={baseline_timing['median_ms'] * 1000:.3f}us "
            f"direct={direct_timing['median_ms'] * 1000:.3f}us "
            f"speedup={speedup:.3f}x better={row['better_than_separate_padding']}",
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
        "workload": "exact K=2048 input Hadamard direct-padding experiment",
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
