#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the complete Llama 3.2 1B MLP at native large M on H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_hopper_large_m as large_m

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
HIDDEN = 2048
INTERMEDIATE = 8192
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_hopper_large_m_mlp.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--previous",
        action="append",
        type=Path,
        default=[],
        help="prior result artifact(s) used for the strict better/regression column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qvq_hopper_large_m/full_mlp.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(m < 1 or m > 16384 for m in args.m_values):
        parser.error("MLP rows must be in [1, 16384]")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def _qvq_mlp(torch, bits, device):
    shared = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    gate = common._qvq_child(
        torch, "gate_proj", INTERMEDIATE, bits, 1, 11001, device, shared
    )
    up = common._qvq_child(
        torch, "up_proj", INTERMEDIATE, bits, 3, 11002, device, shared
    )
    down = common._qvq_child(
        torch,
        "down_proj",
        HIDDEN,
        bits,
        2,
        11003,
        device,
        torch.ones(INTERMEDIATE, device=device, dtype=torch.float32),
        in_features=INTERMEDIATE,
    )
    for child in (gate, up, down):
        child.SV.fill_(0.002)
        child._dtype_cache_clear()

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = gate
            self.up_proj = up
            self.down_proj = down
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return MLP().eval()


def _gptq_mlp(torch, kernel, device):
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
    previous = {}
    for path in args.previous:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", ()):
            key = (float(row["bits"]), int(row["m"]))
            if key in previous:
                raise RuntimeError(f"duplicate previous benchmark row for {key}")
            previous[key] = row["qvq"]
    inputs = {
        m: (
            torch.randn(
                (m, HIDDEN),
                generator=torch.Generator(device=device).manual_seed(12000 + m),
                device=device,
            )
            * 0.02
        ).half()
        for m in args.m_values
    }

    comparator = {}
    for kernel in ("marlin", "machete"):
        modules, call = _gptq_mlp(torch, kernel, device)
        for m, x in inputs.items():
            timing, _ = large_m._graph_timing(
                torch, lambda call=call, x=x: (call(x),), args, device_info
            )
            comparator[(kernel, m)] = timing
        del modules, call
        gc.collect()
        torch.cuda.empty_cache()

    results = []
    for bits in args.rates:
        mlp = _qvq_mlp(torch, bits, device)
        plain = {}
        for m, x in inputs.items():
            plain[m], _ = large_m._graph_timing(
                torch, lambda mlp=mlp, x=x: (mlp(x),), args, device_info
            )
        installed = install_qvq_hopper_groups(mlp, qkv=False)
        if installed != {"gate_up": 1}:
            raise RuntimeError(f"failed to install gate/up group: {installed}")
        for m, x in inputs.items():
            timing, (actual,) = large_m._graph_timing(
                torch, lambda mlp=mlp, x=x: (mlp(x),), args, device_info
            )
            runtime_telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
            gate = qvq_dense_oracle_forward(mlp.gate_proj, x, device=device).half()
            up = qvq_dense_oracle_forward(mlp.up_proj, x, device=device).half()
            intermediate = torch.nn.functional.silu(gate) * up
            expected = qvq_dense_oracle_forward(
                mlp.down_proj, intermediate, device=device
            ).half()
            error = large_m._errors(torch, (actual,), (expected,))
            if error["max_abs"] > 2e-3:
                raise RuntimeError(
                    f"dense P32 MLP accuracy failed for W{bits:g} M{m}: {error}"
                )
            marlin = comparator[("marlin", m)]
            machete = comparator[("machete", m)]
            previous_qvq = previous.get((float(bits), m))
            if previous_qvq is None and m > 4096:
                # Before row multiplexing, these shapes fell through to the
                # ordinary per-module CUDA path. Preserve that measured path
                # as the strict pre-feature benchmark instead of emitting an
                # unhelpful N/A for the first supported >4096 result.
                previous_qvq = plain[m]
            logical_flops = 2 * m * (HIDDEN * INTERMEDIATE * 2 + INTERMEDIATE * HIDDEN)
            result = {
                "bits": bits,
                "m": m,
                "mkn": [m, HIDDEN, INTERMEDIATE],
                "down_mkn": [m, INTERMEDIATE, HIDDEN],
                "qvq": timing,
                "plain_qvq": plain[m],
                "marlin_w4": marlin,
                "machete_w4": machete,
                "speedup_vs_plain_qvq": plain[m]["median_us"] / timing["median_us"],
                "speedup_vs_marlin_w4": marlin["median_us"] / timing["median_us"],
                "speedup_vs_machete_w4": machete["median_us"] / timing["median_us"],
                "effective_tflops": logical_flops / (timing["median_us"] * 1e6),
                "better_than_plain_qvq": timing["median_us"]
                < plain[m]["median_us"],
                "previous_qvq": previous_qvq,
                "better_than_last_benchmark": (
                    timing["median_us"] < previous_qvq["median_us"]
                    if previous_qvq is not None
                    else None
                ),
                "dense_oracle_error": error,
                "selected_chunk_rows": (
                    runtime_telemetry["h100_large_m_chunk_rows"]
                    if m > 4096
                    else None
                ),
            }
            results.append(result)
            print(
                f"W{bits:g} MKN=({m},{HIDDEN},{INTERMEDIATE})/"
                f"({m},{INTERMEDIATE},{HIDDEN}): qvq={timing['median_us']:.3f}us "
                f"marlin={marlin['median_us']:.3f}us "
                f"machete={machete['median_us']:.3f}us "
                "better_last="
                + (
                    "N/A"
                    if result["better_than_last_benchmark"] is None
                    else "Yes"
                    if result["better_than_last_benchmark"]
                    else "No"
                ),
                flush=True,
            )
        telemetry = qvq_grouped_runtime_telemetry(mlp)[0]
        if telemetry["fused_mlp_fallbacks"]:
            raise RuntimeError(f"large-M MLP unexpectedly fell back: {telemetry}")
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
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "cpu_time_included": False,
        },
        "workload": "complete Llama 3.2 1B gate/up/SiLU/product/down MLP",
        "comparison": "native grouped large-M QVQ versus ordinary QVQ and W4 baselines",
        "previous_artifacts": [str(path) for path in args.previous],
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
