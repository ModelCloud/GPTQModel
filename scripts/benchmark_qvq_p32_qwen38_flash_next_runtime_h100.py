#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark model-facing Flash-Next expert QVQ runtime on H100."""

from __future__ import annotations

import argparse
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

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
HIDDEN = 2560
INTERMEDIATE = 640
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_ampere_cuda.py"),
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_ampere_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("scripts/benchmark_qvq_p32_qwen38_flash_next_runtime_h100.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--dtypes", nargs="+", choices=("float16", "bfloat16"), default=("bfloat16",)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen38_flash_next_h100/runtime_phase4.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def _build_mlp(torch, *, bits: float, device):
    shared = torch.ones(HIDDEN, device=device, dtype=torch.float32)
    gate = common._qvq_child(
        torch,
        "gate_proj",
        INTERMEDIATE,
        bits,
        3,
        20269001,
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
        20269002,
        device,
        shared,
        in_features=HIDDEN,
    )
    down = common._qvq_child(
        torch,
        "down_proj",
        HIDDEN,
        bits,
        3,
        20269003,
        device,
        torch.ones(INTERMEDIATE, device=device, dtype=torch.float32),
        in_features=INTERMEDIATE,
    )
    # Match the QVQ MoE factorization: gate/up share one transformed input,
    # their outputs feed the activation directly, and down consumes that
    # intermediate without another input butterfly.
    gate.output_hadamard = False
    up.output_hadamard = False
    down.input_hadamard = False
    for child in (gate, up, down):
        child.SV.fill_(0.002)
        if child.bias is not None:
            child.bias.zero_()
        child._dtype_cache_clear()

    class FlashNextExpertMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = gate
            self.up_proj = up
            self.down_proj = down
            self.act_fn = torch.nn.SiLU()

        def forward(self, value):
            return self.down_proj(
                self.act_fn(self.gate_proj(value)) * self.up_proj(value)
            )

    return FlashNextExpertMLP().eval()


def _timed_sandwich(torch, *, candidate, baseline, args):
    candidate_a, (output_a,) = common._graph_timing(
        torch,
        candidate,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    control, (control_output,) = common._graph_timing(
        torch,
        baseline,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    candidate_b, (output_b,) = common._graph_timing(
        torch,
        candidate,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    if not torch.equal(output_a, output_b):
        raise RuntimeError("candidate graph replay is not bitwise repeatable")
    candidate_ms = math.sqrt(candidate_a["median_ms"] * candidate_b["median_ms"])
    error = output_b.float() - control_output.float()
    metrics = {
        "mean_abs": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
    }
    if metrics["mean_abs"] > 4e-3 or metrics["max_abs"] > 0.046875:
        raise RuntimeError(f"module-level accuracy gate failed: {metrics}")
    return (
        {
            "baseline": control,
            "candidate_first": candidate_a,
            "candidate_second": candidate_b,
            "candidate_sandwich_median_ms": candidate_ms,
            "speedup": control["median_ms"] / candidate_ms,
            "candidate_vs_baseline": metrics,
            "repeatable": True,
        },
        control_output.detach(),
        output_b.detach(),
    )


def _metrics(actual, expected):
    actual_fp64 = actual.double()
    error = actual_fp64 - expected.to(device=actual.device, dtype=actual_fp64.dtype)
    return {
        "mean_abs": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
    }


def _oracle_hadamard(value):
    from gptqmodel.quantization.rotation.hadamard_utils import _get_hadK_on

    width = value.shape[-1]
    base, base_width = _get_hadK_on(value, False)
    staged = value.clone().reshape(-1, width, 1)
    scratch = staged.clone()
    while staged.shape[1] > base_width:
        staged = staged.reshape(
            staged.shape[0], staged.shape[1] // 2, 2, staged.shape[2]
        )
        scratch = scratch.reshape(staged.shape)
        scratch[:, :, 0, :] = staged[:, :, 0, :] + staged[:, :, 1, :]
        scratch[:, :, 1, :] = staged[:, :, 0, :] - staged[:, :, 1, :]
        scratch = scratch.reshape(staged.shape[0], staged.shape[1], -1)
        staged, scratch = scratch, staged
    if base_width > 1:
        staged = (
            base.to(device=value.device, dtype=value.dtype).reshape(
                1, base_width, base_width
            )
            @ staged
        )
    return staged.reshape(value.shape) / math.sqrt(width)


def _dense_linear_oracle(torch, layer, value, weight, dtype):

    transformed = value.to(dtype) * layer.SU.to(device=value.device, dtype=dtype)
    if layer.input_hadamard:
        transformed = _oracle_hadamard(transformed)
    output = transformed @ weight.to(dtype)
    if layer.output_hadamard:
        output = _oracle_hadamard(output)
    output = output * layer.SV.to(device=value.device, dtype=dtype)
    if layer.bias is not None:
        output = output + layer.bias.to(device=value.device, dtype=dtype)
    return output


def _mlp_oracles(torch, model, value, dense_weights):
    value = value.cpu()
    outputs = {}
    for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
        gate = _dense_linear_oracle(
            torch, model.gate_proj, value, dense_weights[0], dtype
        )
        up = _dense_linear_oracle(torch, model.up_proj, value, dense_weights[1], dtype)
        intermediate = torch.nn.functional.silu(gate) * up
        outputs[label] = _dense_linear_oracle(
            torch, model.down_proj, intermediate, dense_weights[2], dtype
        )
    return outputs


def _oracle_gates(baseline, candidate, oracles):
    metrics = {
        "baseline_vs_fp32": _metrics(baseline, oracles["fp32"]),
        "candidate_vs_fp32": _metrics(candidate, oracles["fp32"]),
        "baseline_vs_fp64": _metrics(baseline, oracles["fp64"]),
        "candidate_vs_fp64": _metrics(candidate, oracles["fp64"]),
        "fp32_vs_fp64": _metrics(oracles["fp32"], oracles["fp64"]),
    }
    for label, values in metrics.items():
        if values["mean_abs"] > 4e-3 or values["max_abs"] > 0.046875:
            raise RuntimeError(f"{label} full-MLP oracle gate failed: {values}")
    return metrics


def _run(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.nn_modules.qvq_grouped_runtime import (
        install_qvq_hopper_groups,
        qvq_grouped_runtime_telemetry,
    )
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    if not prewarm_qvq_cuda():
        raise RuntimeError("failed to prewarm the BF16 graph-rescue fallback")
    fingerprint = _fingerprint()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    rows = []
    for bits in args.rates:
        baseline_mlp = _build_mlp(torch, bits=bits, device=device)
        candidate_mlp = _build_mlp(torch, bits=bits, device=device)
        dense_weights = tuple(
            child.get_inner_weight_tensor(dtype=torch.float64).cpu()
            for child in (
                baseline_mlp.gate_proj,
                baseline_mlp.up_proj,
                baseline_mlp.down_proj,
            )
        )
        if install_qvq_hopper_groups(candidate_mlp, qkv=False) != {"gate_up": 1}:
            raise RuntimeError("failed to install Flash-Next grouped expert runtime")
        if not hasattr(candidate_mlp, "_gptqmodel_qvq_fused_mlp_runtime"):
            raise RuntimeError("failed to install Flash-Next fused MLP runtime")
        for dtype_name in args.dtypes:
            dtype = dtype_map[dtype_name]
            for logical_rows in args.m_values:
                value = (
                    torch.randn(
                        (logical_rows, HIDDEN),
                        generator=torch.Generator(device=device).manual_seed(
                            20269100 + int(bits * 10) * 100 + logical_rows
                        ),
                        device=device,
                        dtype=dtype,
                    )
                    * 0.02
                )
                # Warm both module graphs before the candidate-first timing
                # sandwich. BF16 capture owns a prepared overflow-rescue
                # fallback even when ordinary activations stay finite.
                with torch.inference_mode():
                    baseline_mlp(value)
                    candidate_mlp(value)
                timing, baseline_output, candidate_output = _timed_sandwich(
                    torch,
                    candidate=lambda value=value, model=candidate_mlp: (model(value),),
                    baseline=lambda value=value, model=baseline_mlp: (model(value),),
                    args=args,
                )
                oracles = _mlp_oracles(torch, baseline_mlp, value, dense_weights)
                row = {
                    "bits": bits,
                    "m": logical_rows,
                    "dtype": dtype_name,
                    "gate_up_mkn": [logical_rows, HIDDEN, 2 * INTERMEDIATE],
                    "down_mkn": [logical_rows, INTERMEDIATE, HIDDEN],
                    **timing,
                    "oracles": _oracle_gates(
                        baseline_output, candidate_output, oracles
                    ),
                }
                rows.append(row)
                print(
                    f"W{bits:g} {dtype_name} M{logical_rows}: "
                    f"{timing['baseline']['median_ms'] * 1000:.3f}us -> "
                    f"{timing['candidate_sandwich_median_ms'] * 1000:.3f}us "
                    f"({timing['speedup']:.3f}x)",
                    flush=True,
                )
        telemetry = qvq_grouped_runtime_telemetry(candidate_mlp)[0]
        if (
            telemetry["plain_fallbacks"]
            or telemetry["fused_mlp_fallbacks"]
            or not telemetry["fused_mlp_launches"]
        ):
            raise RuntimeError(f"grouped runtime fell back: {telemetry}")

    if fingerprint != _fingerprint():
        raise RuntimeError("benchmark sources changed during execution")
    result = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "checkpoint": "/monster/data/model/Qwen3.8-Flash-Next",
        "device": device_info,
        "baseline": "three independent model-facing QVQLinear forwards",
        "candidate": "grouped gate/up plus fused expert MLP runtime",
        "timing": {
            "method": "candidate/control/candidate CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
        },
        "oracle": {
            "kind": "full MLP FP32 and FP64 dense accumulation",
            "mean_abs_limit": 4e-3,
            "max_abs_limit": 0.046875,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
