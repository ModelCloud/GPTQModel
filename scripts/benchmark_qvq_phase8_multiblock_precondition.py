#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare exact single-CTA and multiblock N=8192 MLP precondition on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
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
    Path("scripts/benchmark_qvq_phase8_multiblock_precondition.py"),
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
        default=Path("artifacts/a41_phase8_h100/multiblock_precondition_experiment.json"),
    )
    args = parser.parse_args()
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_swiglu_precondition,
        qvq_cuda_swiglu_precondition_multiblock,
    )

    device_info = common._assert_h100(torch)
    fingerprint = _source_fingerprint()
    rows = []
    for m in args.m_values:
        generator = torch.Generator(device="cuda").manual_seed(83000 + m)
        gate = torch.randn(
            (m, N), generator=generator, device="cuda", dtype=torch.float16
        )
        up = torch.randn(
            (m, N), generator=generator, device="cuda", dtype=torch.float16
        )
        pre_scale = torch.randn(
            (N,), generator=generator, device="cuda", dtype=torch.float16
        )
        single_cta = partial(qvq_cuda_swiglu_precondition, gate, up, pre_scale)
        multiblock = partial(
            qvq_cuda_swiglu_precondition_multiblock, gate, up, pre_scale
        )
        control_timing, control_output = common._graph_timing(
            torch,
            single_cta,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        candidate_timing, candidate_output = common._graph_timing(
            torch,
            multiblock,
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        bit_exact = torch.equal(candidate_output[0], control_output[0])
        if not bit_exact:
            raise RuntimeError(f"multiblock precondition changed M{m} output")
        row = {
            "m": m,
            "n": N,
            "single_cta": control_timing,
            "multiblock": candidate_timing,
            "speedup": control_timing["median_ms"] / candidate_timing["median_ms"],
            "better_than_single_cta": (
                candidate_timing["median_ms"] < control_timing["median_ms"]
            ),
            "bit_exact": bit_exact,
            "workspace_bytes": m * N * 2,
        }
        rows.append(row)
        print(
            f"M{m}: single={control_timing['median_ms'] * 1000:.3f}us "
            f"multiblock={candidate_timing['median_ms'] * 1000:.3f}us "
            f"speedup={row['speedup']:.3f}x better={row['better_than_single_cta']}",
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
        "workload": "exact N=8192 SwiGLU product and QVQ down-input precondition",
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
