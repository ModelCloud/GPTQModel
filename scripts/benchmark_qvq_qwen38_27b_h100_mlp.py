#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the exact Qwen3.8-27B MLP site on the physical H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_hopper_large_m as timing_utils

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
HIDDEN = 5120
INTERMEDIATE = 17408
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_qwen38_27b_h100_mlp.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--previous",
        type=Path,
        help="Optional prior result with matching W/M rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen38_27b_h100/mlp_baseline.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(m not in M_VALUES for m in args.m_values):
        parser.error("M must be one of the native decode/prefill buckets through 4096")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def qwen38_qvq_mlp(torch, bits: float, device):
    shared = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    gate = common._qvq_child(
        torch,
        "gate_proj",
        INTERMEDIATE,
        bits,
        1,
        51001,
        device,
        shared,
        in_features=HIDDEN,
    )
    up = common._qvq_child(
        torch,
        "up_proj",
        INTERMEDIATE,
        bits,
        3,
        51002,
        device,
        shared,
        in_features=HIDDEN,
    )
    down = common._qvq_child(
        torch,
        "down_proj",
        HIDDEN,
        bits,
        2,
        51003,
        device,
        torch.ones(INTERMEDIATE, device=device, dtype=torch.float32),
        in_features=INTERMEDIATE,
    )
    gate.output_hadamard = False
    up.output_hadamard = False
    down.input_hadamard = False
    for child in (gate, up, down):
        child.SV.fill_(0.002)
        child._dtype_cache_clear()

    class Qwen38MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = gate
            self.up_proj = up
            self.down_proj = down
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return Qwen38MLP().eval()


def _gptq_mlp(torch, kernel: str, device):
    gate, up = common._gptq_modules(
        torch,
        kernel,
        ("gate_proj", "up_proj"),
        (INTERMEDIATE, INTERMEDIATE),
        device,
        in_features=HIDDEN,
    )
    (down,) = common._gptq_modules(
        torch,
        kernel,
        ("down_proj",),
        (HIDDEN,),
        device,
        in_features=INTERMEDIATE,
    )

    def call(x):
        return down(torch.nn.functional.silu(gate(x)) * up(x))

    return (gate, up, down), call


def _geomean(values) -> float:
    values = tuple(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward
    from gptqmodel.nn_modules.qvq_grouped_runtime import (
        install_qvq_hopper_groups,
        qvq_grouped_runtime_telemetry,
    )

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    fingerprint = _fingerprint()
    inputs = {
        m: (
            torch.randn(
                (m, HIDDEN),
                generator=torch.Generator(device=device).manual_seed(52000 + m),
                device=device,
            )
            * 0.02
        ).half()
        for m in args.m_values
    }
    previous = {}
    if args.previous is not None:
        payload = json.loads(args.previous.read_text())
        previous = {
            (float(row["bits"]), int(row["m"])): row["qvq"]
            for row in payload["rows"]
        }

    baselines = {}
    for kernel in ("marlin", "machete"):
        modules, call = _gptq_mlp(torch, kernel, device)
        for m, x in inputs.items():
            baselines[(kernel, m)], _ = timing_utils._graph_timing(
                torch, lambda call=call, x=x: (call(x),), args, device_info
            )
        del modules, call
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for bits in args.rates:
        mlp = qwen38_qvq_mlp(torch, bits, device)
        if install_qvq_hopper_groups(mlp, qkv=False) != {"gate_up": 1}:
            raise RuntimeError("failed to install Qwen gate/up runtime")
        for m, x in inputs.items():
            timing, (actual,) = timing_utils._graph_timing(
                torch, lambda mlp=mlp, x=x: (mlp(x),), args, device_info
            )
            with torch.inference_mode():
                gate = qvq_dense_oracle_forward(mlp.gate_proj, x, device=device).half()
                up = qvq_dense_oracle_forward(mlp.up_proj, x, device=device).half()
                intermediate = torch.nn.functional.silu(gate) * up
                expected = qvq_dense_oracle_forward(
                    mlp.down_proj, intermediate, device=device
                ).half()
            error = timing_utils._errors(torch, (actual,), (expected,))
            if error["max_abs"] > 2e-3:
                raise RuntimeError(f"dense-P32 error gate failed: {error}")
            marlin = baselines[("marlin", m)]
            machete = baselines[("machete", m)]
            last = previous.get((bits, m))
            logical_flops = 2 * m * (
                HIDDEN * INTERMEDIATE * 2 + INTERMEDIATE * HIDDEN
            )
            row = {
                "bits": bits,
                "m": m,
                "gate_up_mkn": [m, HIDDEN, INTERMEDIATE],
                "down_mkn": [m, INTERMEDIATE, HIDDEN],
                "qvq": timing,
                "marlin_w4": marlin,
                "machete_w4": machete,
                "speedup_vs_marlin_w4": marlin["median_us"] / timing["median_us"],
                "speedup_vs_machete_w4": machete["median_us"] / timing["median_us"],
                "effective_tflops": logical_flops / (timing["median_us"] * 1e6),
                "dense_oracle_error": error,
                "previous_qvq": last,
                "better_than_last_benchmark": (
                    timing["median_us"] < last["median_us"] if last else None
                ),
            }
            rows.append(row)
            print(
                f"W{bits:g} M{m}: qvq={timing['median_us']:.3f}us "
                f"marlin={marlin['median_us']:.3f}us "
                f"machete={machete['median_us']:.3f}us",
                flush=True,
            )
        telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
        if telemetry["plain_fallbacks"] or telemetry["fused_mlp_fallbacks"]:
            raise RuntimeError(f"Qwen grouped runtime fell back: {telemetry}")
        del mlp
        gc.collect()
        torch.cuda.empty_cache()

    if fingerprint != _fingerprint():
        raise RuntimeError("benchmark sources changed during execution")
    payload = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "hf_model": "Qwen/Qwen3.8-27B",
        "hf_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "cpu_time_included": False,
        },
        "workload": "complete Qwen3.8-27B gate/up/SiLU/product/down MLP",
        "rows": rows,
        "geomean_vs_marlin_w4": _geomean(
            row["speedup_vs_marlin_w4"] for row in rows
        ),
        "geomean_vs_machete_w4": _geomean(
            row["speedup_vs_machete_w4"] for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
