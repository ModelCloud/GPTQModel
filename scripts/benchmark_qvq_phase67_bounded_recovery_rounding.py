#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare Phase-64 recovery with conservatively bounded rounding on H100."""

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
    Path("scripts/benchmark_qvq_phase67_bounded_recovery_rounding.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a41_phase67_h100/bounded_recovery_rounding.json"),
    )
    args = parser.parse_args()
    if any(m not in M_VALUES for m in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        digest.update(str(relative).encode())
        digest.update((REPO_ROOT / relative).read_bytes())
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
    )

    fingerprint = _fingerprint()
    device = common._assert_h100(torch)
    rows = []
    for m in args.m_values:
        generator = torch.Generator(device="cuda").manual_seed(167000 + m)
        input0 = torch.randn((m, N), generator=generator, device="cuda") * 20
        input1 = torch.randn((m, N), generator=generator, device="cuda") * 20
        scale0 = torch.randn((N,), generator=generator, device="cuda")
        scale1 = torch.randn((N,), generator=generator, device="cuda")
        bias0 = torch.randn((N,), generator=generator, device="cuda")
        bias1 = torch.randn((N,), generator=generator, device="cuda")
        pre_scale = torch.randn(
            (N,), generator=generator, device="cuda", dtype=torch.float16
        )

        shared = dict(
            post_scale0=scale0,
            post_scale1=scale1,
            bias0=bias0,
            bias1=bias1,
            pre_scale=pre_scale,
            pad_to_16=m < 16,
            pair_tiles=m == 16,
        )
        control = partial(
            qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
            input0,
            input1,
            **shared,
            bounded_rounding=False,
        )
        candidate = partial(
            qvq_cuda_hadamard_pair_swiglu_precondition_multiblock,
            input0,
            input1,
            **shared,
            bounded_rounding=True,
        )

        control_timing, expected = common._graph_timing(
            torch,
            lambda control=control: (control(),),
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        candidate_timing, actual = common._graph_timing(
            torch,
            lambda candidate=candidate: (candidate(),),
            warmup=args.warmup,
            samples=args.samples,
            replays_per_sample=args.replays_per_sample,
        )
        exact = torch.equal(actual[0], expected[0])
        if not exact:
            raise RuntimeError(f"bounded rounding changed M{m} output")
        speedup = control_timing["median_ms"] / candidate_timing["median_ms"]
        rows.append(
            {
                "m": m,
                "matrix_shape": [m, 2048, 8192],
                "control": control_timing,
                "candidate": candidate_timing,
                "speedup_vs_phase64": speedup,
                "better_than_phase64": speedup > 1.0,
                "bit_exact": exact,
                "pair_tiles": m == 16,
                "fast_path_abs_sum_limit": 64000.0,
                "candidate_registers_unpaired": 44,
                "candidate_registers_paired": 45,
                "candidate_stack_bytes": 0,
                "candidate_local_bytes": 0,
            }
        )
        print(
            f"M{m}: phase64={control_timing['median_ms'] * 1000:.3f}us "
            f"bounded={candidate_timing['median_ms'] * 1000:.3f}us "
            f"speedup={speedup:.4f}x better={speedup > 1}",
            flush=True,
        )

    if fingerprint != _fingerprint():
        raise RuntimeError("benchmark source changed during timing")
    payload = {
        "git_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "device": device,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "host_launch_gaps_included": False,
        },
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
