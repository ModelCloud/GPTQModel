#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the exact Phase-5 fused Llama MLP path on the physical H100."""

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

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
HIDDEN = 2048
INTERMEDIATE = 8192
SOURCE_PATHS = (
    Path("gptqmodel/models/base.py"),
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_a41_phase4_production.py"),
    Path("scripts/benchmark_qvq_a41_phase5_mlp.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a41_phase5_h100/fused_mlp_vs_baselines.json"),
    )
    parser.add_argument(
        "--previous-artifact",
        type=Path,
        default=Path(
            "artifacts/a41_phase7_h100/production_mlp_refresh_vs_baselines.json"
        ),
        help="Last committed comparable matrix used for per-row regression telemetry.",
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    if args.idle_samples < 3 or args.idle_interval < 0 or args.idle_memory_mib < 0:
        parser.error(
            "idle gate requires at least three samples and nonnegative thresholds"
        )
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _call_mlp(torch, gate, up, down, x):
    return down(torch.nn.functional.silu(gate(x)) * up(x))


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
    return gate, up, down


def _qvq_mlp(torch, bits: float, device):
    from torch import nn

    hidden_su = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    intermediate_su = torch.ones(INTERMEDIATE, device=device, dtype=torch.float32)
    gate = common._qvq_child(
        torch,
        "gate_proj",
        INTERMEDIATE,
        bits,
        1,
        11000 + int(bits * 10),
        device,
        hidden_su,
        in_features=HIDDEN,
    )
    up = common._qvq_child(
        torch,
        "up_proj",
        INTERMEDIATE,
        bits,
        3,
        11100 + int(bits * 10),
        device,
        hidden_su,
        in_features=HIDDEN,
    )
    down = common._qvq_child(
        torch,
        "down_proj",
        HIDDEN,
        bits,
        2,
        11200 + int(bits * 10),
        device,
        intermediate_su,
        in_features=INTERMEDIATE,
    )
    with torch.no_grad():
        gate.SV.fill_(0.002)
        up.SV.fill_(0.002)
        down.SV.fill_(0.002)

    class LlamaLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = gate
            self.up_proj = up
            self.down_proj = down
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return LlamaLikeMLP().eval()


def _run(args):
    import torch

    from gptqmodel.nn_modules.qvq_grouped_runtime import (
        install_qvq_hopper_groups,
        qvq_grouped_runtime_telemetry,
        uninstall_qvq_hopper_groups,
    )

    source_fingerprint = _source_fingerprint()
    previous_payload = json.loads(args.previous_artifact.read_text())
    previous_rows = {
        (float(row["bits"]), int(row["m"])): row
        for row in previous_payload["rows"]
    }
    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
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

    baseline_timings = {}
    for kernel in ("marlin", "machete"):
        gate, up, down = _gptq_mlp(torch, kernel, device)
        for m in args.m_values:
            timing, _ = common._graph_timing(
                torch,
                lambda gate=gate, up=up, down=down, x=inputs[m]: (
                    _call_mlp(torch, gate, up, down, x),
                ),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            baseline_timings[(m, kernel)] = timing
        del gate, up, down
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for bits in args.rates:
        mlp = _qvq_mlp(torch, bits, device)
        plain = {}
        expected = {}
        for m in args.m_values:
            timing, outputs = common._graph_timing(
                torch,
                lambda mlp=mlp, x=inputs[m]: (mlp(x),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            plain[m] = timing
            expected[m] = outputs[0]

        counts = install_qvq_hopper_groups(
            mlp,
            qkv=False,
            gate_up_activation=False,
        )
        if counts != {"gate_up": 1}:
            raise RuntimeError(f"failed to install paired-recovery control: {counts}")
        paired_recovery = {}
        for m in args.m_values:
            timing, outputs = common._graph_timing(
                torch,
                lambda mlp=mlp, x=inputs[m]: (mlp(x),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            if not torch.equal(outputs[0], expected[m]):
                raise RuntimeError(f"paired recovery changed W{bits:g} M{m} MLP output")
            paired_recovery[m] = timing
        uninstall_qvq_hopper_groups(mlp)

        counts = install_qvq_hopper_groups(mlp, qkv=False, gate_up_activation=True)
        if counts != {"gate_up": 1} or not hasattr(
            mlp, "_gptqmodel_qvq_fused_mlp_runtime"
        ):
            raise RuntimeError(f"failed to install fused MLP path: {counts}")
        for m in args.m_values:
            timing, outputs = common._graph_timing(
                torch,
                lambda mlp=mlp, x=inputs[m]: (mlp(x),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            if not torch.equal(outputs[0], expected[m]):
                raise RuntimeError(f"fused W{bits:g} M{m} MLP output changed")
            marlin = baseline_timings[(m, "marlin")]
            machete = baseline_timings[(m, "machete")]
            logical_flops = (
                2 * m * (HIDDEN * (2 * INTERMEDIATE) + INTERMEDIATE * HIDDEN)
            )
            previous = previous_rows[(float(bits), int(m))]["fused_mlp_qvq"]
            rows.append(
                {
                    "bits": bits,
                    "m": m,
                    "matrix_shapes": [
                        [m, HIDDEN, INTERMEDIATE],
                        [m, HIDDEN, INTERMEDIATE],
                        [m, INTERMEDIATE, HIDDEN],
                    ],
                    "plain_qvq": plain[m],
                    "paired_recovery_qvq": paired_recovery[m],
                    "fused_mlp_qvq": timing,
                    "marlin_w4": marlin,
                    "machete_w4": machete,
                    "speedup_vs_plain": plain[m]["median_ms"] / timing["median_ms"],
                    "speedup_vs_paired_recovery": paired_recovery[m]["median_ms"]
                    / timing["median_ms"],
                    "speedup_vs_marlin_w4": marlin["median_ms"] / timing["median_ms"],
                    "speedup_vs_machete_w4": machete["median_ms"] / timing["median_ms"],
                    "speedup_vs_previous_benchmark": previous["median_ms"]
                    / timing["median_ms"],
                    "better_than_previous_benchmark": timing["median_ms"]
                    < previous["median_ms"],
                    "better_than_paired_recovery_stage": timing["median_ms"]
                    < paired_recovery[m]["median_ms"],
                    "plain_qvq_effective_tflops": logical_flops
                    / (plain[m]["median_ms"] * 1e9),
                    "paired_recovery_effective_tflops": logical_flops
                    / (paired_recovery[m]["median_ms"] * 1e9),
                    "fused_mlp_effective_tflops": logical_flops
                    / (timing["median_ms"] * 1e9),
                    "marlin_effective_tflops": logical_flops
                    / (marlin["median_ms"] * 1e9),
                    "machete_effective_tflops": logical_flops
                    / (machete["median_ms"] * 1e9),
                }
            )
            print(
                f"W{bits:g} M{m}: fused={timing['median_ms'] * 1000:.3f}us "
                f"paired={paired_recovery[m]['median_ms'] * 1000:.3f}us "
                f"plain={plain[m]['median_ms'] * 1000:.3f}us "
                f"marlin={marlin['median_ms'] * 1000:.3f}us "
                f"machete={machete['median_ms'] * 1000:.3f}us",
                flush=True,
            )
        telemetry = qvq_grouped_runtime_telemetry(mlp)
        if (
            len(telemetry) != 1
            or telemetry[0]["fused_mlp_fallbacks"]
            or not telemetry[0]["fused_mlp_launches"]
            or (
                16 in args.m_values
                and not telemetry[0]["h100_paired_recovery_tiles_launches"]
            )
        ):
            raise RuntimeError(f"unexpected fused MLP telemetry: {telemetry}")
        del mlp, expected, plain, paired_recovery
        gc.collect()
        torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint():
        raise RuntimeError(
            "benchmark sources changed while the H100 matrix was running"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    payload = {
        "git_base_commit": commit,
        "source_fingerprint": source_fingerprint,
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "host_launch_gaps_included": False,
        },
        "workload": "Llama 3.2 1B complete gate/up/SiLU/down MLP",
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
