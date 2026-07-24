#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.quantization.paroquant.modules.triton.gemm import FP32_ACCUM
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
    BenchCase("decode_gate_proj", batch=1, seq=1, in_features=2048, out_features=8192),
    BenchCase("decode_down_proj", batch=1, seq=1, in_features=8192, out_features=2048),
    BenchCase("prefill_q_proj", batch=1, seq=128, in_features=2048, out_features=2048),
    BenchCase("batched_q_proj", batch=4, seq=128, in_features=2048, out_features=2048),
    BenchCase("prefill_k_proj", batch=1, seq=128, in_features=2048, out_features=512),
    BenchCase("prefill_gate_proj", batch=1, seq=128, in_features=2048, out_features=8192),
    BenchCase("prefill_down_proj", batch=1, seq=128, in_features=8192, out_features=2048),
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


def _make_quant_buffers(
    case: BenchCase,
    bits: int = 4,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    groups = case.in_features // case.group_size
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = ((torch.rand(groups, case.out_features, dtype=torch.float32) * 0.5) + 0.75).to(dtype)
    bias = torch.randn(case.out_features, dtype=torch.float32).to(dtype)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=case.in_features,
        group_size=case.group_size,
        krot=case.krot,
        dtype=dtype,
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
    dtype: torch.dtype = torch.float16,
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
    ).to(device=device, dtype=dtype)
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


def _benchmark_stats(module, x: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(x.device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            module(x)
            ends[index].record()
        torch.cuda.synchronize(x.device)

    samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)]
    ordered = sorted(samples_us)
    return {
        "mean_us": statistics.mean(samples_us),
        "p50_us": statistics.median(samples_us),
        "p95_us": ordered[int(0.95 * (iters - 1))],
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _format_speedup(speedup: float) -> str:
    return f"{speedup:.3f}x"


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}.")


def _megakernel_metadata_bytes(module: ParoQuantTritonLinear) -> int:
    tensors = (
        module._megakernel_partner,
        module._megakernel_decode_partner,
        module._megakernel_cos,
        module._megakernel_sin,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)


def _megakernel_scratch_bytes(module: ParoQuantTritonLinear) -> int:
    return sum(
        partials.numel() * partials.element_size() + counters.numel() * counters.element_size()
        for partials, counters in module._megakernel_splitk_scratch.values()
    )


def _peak_forward_allocation_bytes(module, x: torch.Tensor) -> int:
    """Measure peak live allocation above the warmed steady-state baseline."""
    with torch.inference_mode():
        module(x)
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
        baseline_bytes = torch.cuda.memory_allocated(x.device)
        output = module(x)
        torch.cuda.synchronize(x.device)
        peak_bytes = torch.cuda.max_memory_allocated(x.device)
        del output
    return max(0, peak_bytes - baseline_bytes)


def _nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return ",".join(sorted(versions)) or None


def run(
    device: torch.device,
    dtype: torch.dtype,
    cases: list[BenchCase],
    warmup: int,
    iters: int,
    *,
    quick: bool,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    accuracy_rows = []
    benchmark_rows = []
    speedups = []
    case_results = []

    for index, case in enumerate(cases):
        torch.manual_seed(1000 + index)
        buffers = _make_quant_buffers(case, dtype=dtype)
        baseline = _build_module(ParoLinear, case, buffers, device, dtype=dtype)
        candidate = _build_module(ParoQuantTritonLinear, case, buffers, device, dtype=dtype)

        x = torch.randn((case.batch, case.seq, case.in_features), device=device, dtype=dtype)

        with torch.inference_mode():
            dense = _dense_reference(baseline, x)
            baseline_out = baseline(x)
            candidate_out = candidate(x)
            x_flat = x.reshape(-1, x.shape[-1])
            kind = candidate._classify_forward_kind(x, x_flat)
            selected_plan = candidate._select_plan(kind, x_flat)

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

        baseline_stats = _benchmark_stats(baseline, x, warmup=warmup, iters=iters)
        candidate_stats = _benchmark_stats(candidate, x, warmup=warmup, iters=iters)
        baseline_peak_bytes = _peak_forward_allocation_bytes(baseline, x)
        candidate_peak_bytes = _peak_forward_allocation_bytes(candidate, x)
        speedup = baseline_stats["mean_us"] / candidate_stats["mean_us"]
        p50_speedup = baseline_stats["p50_us"] / candidate_stats["p50_us"]
        speedups.append(speedup)
        winner = "triton" if speedup > 1.0 else "existing"
        rows = case.batch * case.seq
        baseline_tps = rows * 1e6 / baseline_stats["mean_us"]
        candidate_tps = rows * 1e6 / candidate_stats["mean_us"]
        metadata_bytes = _megakernel_metadata_bytes(candidate)
        scratch_bytes = _megakernel_scratch_bytes(candidate)

        benchmark_rows.append(
            [
                case.case_id,
                f"{case.batch}x{case.seq}",
                f"{case.in_features}->{case.out_features}",
                selected_plan,
                f"{baseline_stats['p50_us']:.2f}/{baseline_stats['mean_us']:.2f}/{baseline_stats['p95_us']:.2f}",
                f"{candidate_stats['p50_us']:.2f}/{candidate_stats['mean_us']:.2f}/{candidate_stats['p95_us']:.2f}",
                _format_speedup(speedup),
                _format_speedup(p50_speedup),
                f"{baseline_tps:.1f}",
                f"{candidate_tps:.1f}",
                f"{metadata_bytes / 1024:.1f}/{scratch_bytes / 1024:.1f}",
                f"{baseline_peak_bytes / 1024:.1f}/{candidate_peak_bytes / 1024:.1f}",
                winner,
            ]
        )
        case_results.append(
            {
                "case": asdict(case),
                "kind": kind,
                "selected_plan": selected_plan,
                "baseline": baseline_stats,
                "candidate": candidate_stats,
                "speedup_mean": speedup,
                "speedup_p50": p50_speedup,
                "baseline_module_token_tps": baseline_tps,
                "candidate_module_token_tps": candidate_tps,
                "megakernel_metadata_bytes": metadata_bytes,
                "megakernel_scratch_bytes": scratch_bytes,
                "baseline_peak_forward_allocation_bytes": baseline_peak_bytes,
                "candidate_peak_forward_allocation_bytes": candidate_peak_bytes,
                "accuracy": {
                    "baseline_dense_max_abs": baseline_dense.max().item(),
                    "candidate_dense_max_abs": candidate_dense.max().item(),
                    "baseline_candidate_max_abs": baseline_candidate.max().item(),
                    "baseline_candidate_mean_abs": baseline_candidate.mean().item(),
                },
            }
        )

    geo_mean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups))
    triton_wins = sum(1 for value in speedups if value > 1.0)

    props = torch.cuda.get_device_properties(device)
    return {
        "device": props.name,
        "device_uuid": str(props.uuid),
        "cuda_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_driver_version": _nvidia_driver_version(),
        "triton_version": __import__("triton").__version__,
        "dtype": str(dtype),
        "fp32_accum": FP32_ACCUM,
        "megakernel_launch_configs": {
            "decode": {
                "block_m": {"fp16_krot8_m_le_2": 2, "fp16_krot8_m_3_to_8": 4, "other": 8},
                "block_n": 128,
                "block_k": 128,
                "num_warps": 8,
                "num_stages": 2,
                "partner_dtype": {"bf16_krot8": "int8", "other": "int16"},
                "loop_unroll_factor": {"bf16_krot8_k_lt_384": 2, "other": 1},
                "prefetch_first_partner": {"bf16_krot8_k_ge_384": True, "other": False},
                "prefetch_packed_weight": {"bf16_krot8_k_ge_1024": True, "other": False},
                "split_k": {
                    "factor": 16,
                    "shape": {"m": list(range(1, 9)), "k": 2048, "n": [512, 2048, 8192], "krot": 8},
                    "dtype": "bfloat16",
                    "sm_count": 124,
                    "execution": "eager only; CUDA graph capture uses the standard mega-kernel",
                    "compiled_launcher": "Triton 3.7 internal ABI; JIT fallback otherwise",
                },
            },
            "prefill_narrow_n": {
                "max_n": 512,
                "block_m": {
                    "n128_124sm_m1985_3968": 32,
                    "n256_124sm_m993_1984": 32,
                    "n384_124sm_m657_1312": 32,
                    "n512_124sm_m497_992": 32,
                    "default": 8,
                },
                "block_n": 128,
                "block_k": 128,
                "num_warps": 8,
                "num_stages": 2,
                "prefetch_first_partner": {"bf16_krot8_k_ge_1024": True, "other": False},
            },
            "prefill_fp16_q": {
                "shape": {
                    "k": 2048,
                    "n": list(range(640, 4097, 128)),
                    "krot": 8,
                },
                "block_m": {"half_wave_count_bm32": 32, "default": 16},
                "selection": (
                    "portable one-wave rule through N=2048; up to four N=2048 waves and first-wave "
                    "N=2176-4096 bands on 124 SMs"
                ),
                "block_n": 128,
                "block_k": 128,
                "num_warps": {"124sm_half_wave_count_bm32": 16, "default": 8},
                "num_stages": {"124sm_half_wave_count_bm32": 1, "default": 2},
            },
            "prefill": {"block_m": 16, "block_n": 128, "block_k": 128, "num_warps": 8, "num_stages": 2},
        },
        "warmup": warmup,
        "iters": iters,
        "quick": quick,
        "shard_index": shard_index,
        "num_shards": num_shards,
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
            "selected plan",
            "existing p50/mean/p95 us",
            "triton p50/mean/p95 us",
            "mean speedup",
            "p50 speedup",
            "existing tok/s",
            "triton tok/s",
            "metadata/scratch KiB",
            "peak alloc KiB existing/triton",
            "winner",
        ],
        "benchmark_rows": benchmark_rows,
        "cases": case_results,
        "geo_mean_speedup": geo_mean_speedup,
        "triton_wins": triton_wins,
        "case_count": len(speedups),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the ParoQuant Triton rotation+GEMM megakernel against the existing CUDA path."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index within the current visible set.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=200, help="Measured CUDA-event samples per case.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller subset of benchmark cases.")
    parser.add_argument(
        "--case-id",
        action="append",
        choices=tuple(case.case_id for case in DEFAULT_CASES),
        help="Run only the selected case; repeat to select more than one.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--json", action="store_true", help="Also emit the full result payload as JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ParoQuant Triton benchmark.")

    try:
        import triton  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Triton is required for the ParoQuant Triton benchmark: {exc}") from exc

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive.")

    selected_cases = QUICK_CASES if args.quick else DEFAULT_CASES
    if args.case_id:
        selected = set(args.case_id)
        selected_cases = [case for case in DEFAULT_CASES if case.case_id in selected]
    selected_cases = selected_cases[args.shard_index :: args.num_shards]
    if not selected_cases:
        raise ValueError("The selected benchmark shard contains no cases.")

    device = torch.device(f"cuda:{args.device}")
    results = run(
        device=device,
        dtype=_resolve_dtype(args.dtype),
        cases=selected_cases,
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    print(
        f"Device: {results['device']} ({results['cuda_device']}, visible={results['cuda_visible_devices']}, "
        f"uuid={results['device_uuid']}, sm={results['compute_capability']}, dtype={results['dtype']})"
    )
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
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
