#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare standalone SiLU + precondition with exact fused SiLU on H100."""

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
N = 8192
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("scripts/benchmark_qvq_phase10_silu_precondition.py"),
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
        default=Path("artifacts/a41_phase10_h100/fused_silu_precondition_experiment.json"),
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
    import torch.nn.functional as F

    from gptqmodel.utils.qvq_cuda import qvq_cuda_swiglu_precondition_multiblock

    device_info = common._assert_h100(torch)
    fingerprint = _source_fingerprint()
    rows = []
    for m in args.m_values:
        generator = torch.Generator(device="cuda").manual_seed(85000 + m)
        gate = torch.randn(
            (m, N), generator=generator, device="cuda", dtype=torch.float16
        )
        up = torch.randn(
            (m, N), generator=generator, device="cuda", dtype=torch.float16
        )
        pre_scale = torch.randn(
            (N,), generator=generator, device="cuda", dtype=torch.float16
        )

        def unfused(gate=gate, up=up, pre_scale=pre_scale):
            activated_gate = F.silu(gate)
            return (
                qvq_cuda_swiglu_precondition_multiblock(
                    activated_gate,
                    up,
                    pre_scale,
                    half2_high=True,
                ),
            )

        def fused(gate=gate, up=up, pre_scale=pre_scale):
            return (
                qvq_cuda_swiglu_precondition_multiblock(
                    gate,
                    up,
                    pre_scale,
                    half2_high=True,
                    fuse_silu=True,
                ),
            )

        unfused_timing, unfused_output = common._graph_timing(
            torch,
            unfused,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        fused_timing, fused_output = common._graph_timing(
            torch,
            fused,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        bit_exact = torch.equal(
            fused_output[0].view(torch.int16), unfused_output[0].view(torch.int16)
        )
        if not bit_exact:
            raise RuntimeError(f"fused SiLU precondition changed M{m} output bits")
        speedup = unfused_timing["median_ms"] / fused_timing["median_ms"]
        row = {
            "m": m,
            "n": N,
            "unfused_silu_precondition": unfused_timing,
            "fused_silu_precondition": fused_timing,
            "speedup_vs_unfused": speedup,
            "better_than_unfused": speedup > 1.0,
            "bit_exact": bit_exact,
            "workspace_bytes": m * N * 2,
            "activated_gate_bytes_eliminated": m * N * 2,
            "gpu_launches": {"unfused": 3, "fused": 2},
        }
        rows.append(row)
        print(
            f"M{m}: unfused={unfused_timing['median_ms'] * 1000:.3f}us "
            f"fused={fused_timing['median_ms'] * 1000:.3f}us "
            f"speedup={speedup:.3f}x better={row['better_than_unfused']}",
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
        "workload": "exact FP16 SiLU plus N=8192 SwiGLU/down precondition",
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
