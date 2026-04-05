#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.utils.paroquant import build_identity_rotation_buffers


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    batch: int
    seq: int
    in_features: int
    out_features: int
    group_size: int = 128
    krot: int = 8


DEFAULT_CASES = [
    BenchCase("decode_q_proj", batch=1, seq=1, in_features=2048, out_features=2048),
    BenchCase("prefill_q_proj", batch=1, seq=128, in_features=2048, out_features=2048),
    BenchCase("batched_q_proj", batch=4, seq=128, in_features=2048, out_features=2048),
    BenchCase("prefill_k_proj", batch=1, seq=128, in_features=2048, out_features=512),
    BenchCase("prefill_gate_proj", batch=1, seq=128, in_features=2048, out_features=8192),
    BenchCase("batched_down_proj", batch=4, seq=128, in_features=8192, out_features=2048),
]

QUICK_CASES = [
    BenchCase("decode_q_proj", batch=1, seq=1, in_features=2048, out_features=2048),
    BenchCase("prefill_q_proj", batch=1, seq=128, in_features=2048, out_features=2048),
    BenchCase("batched_down_proj", batch=4, seq=128, in_features=8192, out_features=2048),
]


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_quant_buffers(case: BenchCase, bits: int = 4) -> dict[str, torch.Tensor]:
    groups = case.in_features // case.group_size
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = (torch.rand(groups, case.out_features, dtype=torch.float16) * 0.5) + 0.75
    bias = torch.randn(case.out_features, dtype=torch.float16)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=case.in_features,
        group_size=case.group_size,
        krot=case.krot,
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    return {
        "qweight": _pack_awq_tensor(int_weight, bits),
        "qzeros": _pack_awq_tensor(zero_points, bits),
        "scales": scales,
        "bias": bias,
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
    }


def _build_module(
    module_cls,
    case: BenchCase,
    buffers: dict[str, torch.Tensor],
    device: torch.device,
    bits: int = 4,
):
    module = module_cls(
        bits=bits,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=True,
        register_buffers=True,
        krot=case.krot,
    ).to(device)
    module.qweight.copy_(buffers["qweight"].to(device))
    module.qzeros.copy_(buffers["qzeros"].to(device))
    module.scales.copy_(buffers["scales"].to(device))
    module.bias.copy_(buffers["bias"].to(device))
    module.pairs.copy_(buffers["pairs"].to(device))
    module.theta.copy_(buffers["theta"].to(device))
    module.channel_scales.copy_(buffers["channel_scales"].to(device))
    module.post_init()
    module.eval()
    return module


def _dense_reference(module: ParoLinear, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        x_flat = x.reshape(-1, x.shape[-1])
        rotated = module._rotate_inputs(x_flat)
        out = module._forward_dense(rotated)
        return out.reshape(x.shape[:-1] + (module.out_features,))


def _benchmark_ms(module, x: torch.Tensor, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        for _ in range(iters):
            module(x)
        torch.cuda.synchronize(x.device)
    return (time.perf_counter() - start) * 1e3 / iters


def _format_speedup(speedup: float) -> str:
    return f"{speedup:.3f}x"


def run(device: torch.device, warmup: int, iters: int, quick: bool) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cases = QUICK_CASES if quick else DEFAULT_CASES
    accuracy_rows = []
    benchmark_rows = []
    speedups = []

    for index, case in enumerate(cases):
        torch.manual_seed(1000 + index)
        buffers = _make_quant_buffers(case)
        baseline = _build_module(ParoLinear, case, buffers, device)
        candidate = _build_module(ParoQuantTritonLinear, case, buffers, device)

        x = torch.randn((case.batch, case.seq, case.in_features), device=device, dtype=torch.float16)

        with torch.inference_mode():
            dense = _dense_reference(baseline, x)
            baseline_out = baseline(x)
            candidate_out = candidate(x)

        baseline_dense = (baseline_out - dense).abs()
        candidate_dense = (candidate_out - dense).abs()
        baseline_candidate = (baseline_out - candidate_out).abs()

        accuracy_rows.append(
            [
                case.case_id,
                f"{case.batch}x{case.seq}",
                f"{case.in_features}->{case.out_features}",
                f"{baseline_dense.max().item():.6f}",
                f"{candidate_dense.max().item():.6f}",
                f"{baseline_candidate.max().item():.6f}",
                f"{baseline_candidate.mean().item():.6f}",
            ]
        )

        baseline_ms = _benchmark_ms(baseline, x, warmup=warmup, iters=iters)
        candidate_ms = _benchmark_ms(candidate, x, warmup=warmup, iters=iters)
        speedup = baseline_ms / candidate_ms
        speedups.append(speedup)
        winner = "triton" if candidate_ms < baseline_ms else "existing"

        benchmark_rows.append(
            [
                case.case_id,
                f"{case.batch}x{case.seq}",
                f"{case.in_features}->{case.out_features}",
                f"{baseline_ms:.3f}",
                f"{candidate_ms:.3f}",
                _format_speedup(speedup),
                winner,
            ]
        )

    geo_mean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups))
    triton_wins = sum(1 for value in speedups if value > 1.0)

    return {
        "device": torch.cuda.get_device_name(device),
        "cuda_device": str(device),
        "warmup": warmup,
        "iters": iters,
        "quick": quick,
        "accuracy_headers": [
            "case",
            "batch x seq",
            "matmul",
            "existing vs dense max_abs",
            "triton vs dense max_abs",
            "existing vs triton max_abs",
            "existing vs triton mean_abs",
        ],
        "accuracy_rows": accuracy_rows,
        "benchmark_headers": [
            "case",
            "batch x seq",
            "matmul",
            "existing ms",
            "triton ms",
            "speedup",
            "winner",
        ],
        "benchmark_rows": benchmark_rows,
        "geo_mean_speedup": geo_mean_speedup,
        "triton_wins": triton_wins,
        "case_count": len(speedups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark ParoQuant Triton kernel against the existing kernel.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index within the current visible set.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations per case.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller subset of benchmark cases.")
    parser.add_argument("--json", action="store_true", help="Also emit the full result payload as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ParoQuant Triton benchmark.")

    try:
        import triton  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Triton is required for the ParoQuant Triton benchmark: {exc}") from exc

    device = torch.device(f"cuda:{args.device}")
    results = run(device=device, warmup=args.warmup, iters=args.iters, quick=args.quick)

    print(f"Device: {results['device']} ({results['cuda_device']})")
    print()
    print("Accuracy")
    print(tabulate(results["accuracy_rows"], headers=results["accuracy_headers"], tablefmt="grid"))
    print()
    print("Benchmark")
    print(tabulate(results["benchmark_rows"], headers=results["benchmark_headers"], tablefmt="grid"))
    print()
    print(
        "Summary: "
        f"triton_wins={results['triton_wins']}/{results['case_count']}, "
        f"geo_mean_speedup={results['geo_mean_speedup']:.3f}x"
    )

    if args.json:
        print()
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
