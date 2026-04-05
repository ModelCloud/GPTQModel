#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate


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


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _subset_cases(cases: list[BenchCase], shard_index: int, num_shards: int) -> list[BenchCase]:
    if num_shards <= 0:
        raise ValueError("`num_shards` must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"`shard_index` must be in [0, {num_shards - 1}].")
    return [case for index, case in enumerate(cases) if index % num_shards == shard_index]


def _benchmark_ms(fn, device: torch.device, warmup: int, iters: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3 / iters


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32)
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_quant_buffers(case: BenchCase, dtype: torch.dtype, bits: int = 4) -> dict[str, torch.Tensor]:
    from gptqmodel.utils.paroquant import build_identity_rotation_buffers

    groups = case.in_features // case.group_size
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = (torch.rand(groups, case.out_features, dtype=torch.float32) * 0.5) + 0.75
    scales = scales.to(dtype=dtype)
    bias = torch.randn(case.out_features, dtype=torch.float32).to(dtype=dtype)

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


def _make_module(
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    buffers: dict[str, torch.Tensor],
    auto_cache_bf16_rotation_dtype: bool,
):
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    module = ParoLinear(
        bits=4,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=True,
        register_buffers=True,
        krot=case.krot,
        cache_runtime_dtype=False,
        auto_cache_bf16_runtime_dtype=True,
        cache_rotation_dtype=False,
        auto_cache_bf16_rotation_dtype=auto_cache_bf16_rotation_dtype,
    ).to(device)
    module.qweight.copy_(buffers["qweight"].to(device))
    module.qzeros.copy_(buffers["qzeros"].to(device))
    module.scales.copy_(buffers["scales"].to(device))
    module.bias.copy_(buffers["bias"].to(device))
    module.pairs.copy_(buffers["pairs"].to(device))
    module.theta.copy_(buffers["theta"].to(device=device, dtype=module.theta.dtype))
    module.channel_scales.copy_(buffers["channel_scales"].to(device=device, dtype=module.channel_scales.dtype))
    module.post_init()
    module.eval()
    return module


def _dense_reference(module, x: torch.Tensor) -> torch.Tensor:
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    from gptqmodel.utils.paroquant import apply_paroquant_rotation_reference

    rotated = apply_paroquant_rotation_reference(
        x,
        module.pairs,
        module.theta,
        scales=module.channel_scales,
        group_size=module.group_size,
    )
    weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=module.bits,
        group_size=module.group_size,
    ).to(device=x.device, dtype=x.dtype)
    out = torch.matmul(rotated, weight)
    if module.bias is not None:
        out = out + module.bias.to(device=x.device, dtype=x.dtype)
    return out


def _format_speedup(speedup: float) -> str:
    return f"{speedup:.3f}x"


def run(
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    quick: bool,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    rows = []
    speedups = []
    selected_cases = _subset_cases(QUICK_CASES if quick else DEFAULT_CASES, shard_index=shard_index, num_shards=num_shards)

    for index, case in enumerate(selected_cases):
        torch.manual_seed(5000 + index)
        buffers = _make_quant_buffers(case, dtype=dtype)
        baseline = _make_module(
            case,
            dtype=dtype,
            device=device,
            buffers=buffers,
            auto_cache_bf16_rotation_dtype=False,
        )
        candidate = _make_module(
            case,
            dtype=dtype,
            device=device,
            buffers=buffers,
            auto_cache_bf16_rotation_dtype=True,
        )
        x = torch.randn((case.batch, case.seq, case.in_features), device=device, dtype=dtype)

        with torch.inference_mode():
            dense = _dense_reference(baseline, x.reshape(-1, x.shape[-1])).reshape(case.batch, case.seq, case.out_features)
            baseline_out = baseline(x)
            candidate_out = candidate(x)

        baseline_dense = (baseline_out - dense).abs()
        candidate_dense = (candidate_out - dense).abs()
        baseline_candidate = (baseline_out - candidate_out).abs()

        baseline_ms = _benchmark_ms(lambda: baseline(x), device=device, warmup=warmup, iters=iters)
        candidate_ms = _benchmark_ms(lambda: candidate(x), device=device, warmup=warmup, iters=iters)
        speedup = baseline_ms / candidate_ms
        speedups.append(speedup)

        rows.append(
            {
                "case_id": case.case_id,
                "batch": case.batch,
                "seq": case.seq,
                "in_features": case.in_features,
                "out_features": case.out_features,
                "dtype": str(dtype).replace("torch.", ""),
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": speedup,
                "winner": "rotation_cache_on" if candidate_ms < baseline_ms else "rotation_cache_off",
                "baseline_dense_max_abs": baseline_dense.max().item(),
                "candidate_dense_max_abs": candidate_dense.max().item(),
                "baseline_candidate_max_abs": baseline_candidate.max().item(),
                "baseline_candidate_mean_abs": baseline_candidate.mean().item(),
            }
        )

    geo_mean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups)) if speedups else float("nan")
    return {
        "device": torch.cuda.get_device_name(device),
        "cuda_device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "iters": iters,
        "quick": quick,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rows": rows,
        "geo_mean_speedup": geo_mean_speedup,
        "candidate_wins": sum(1 for v in speedups if v > 1.0),
        "case_count": len(speedups),
    }


def _configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if args.force_rebuild_awq:
        awq_build_root = Path("/tmp") / (
            f"awq_jit_rotcache_{os.getpid()}_dev{args.device}_shard{args.shard_index}_of_{args.num_shards}"
        )
        os.environ["GPTQMODEL_AWQ_BUILD_ROOT"] = str(awq_build_root)
        os.environ["GPTQMODEL_AWQ_FORCE_REBUILD"] = "1"
    else:
        os.environ.pop("GPTQMODEL_AWQ_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_AWQ_FORCE_REBUILD", None)

    if args.force_rebuild_paroquant:
        paro_build_root = Path("/tmp") / (
            f"paroquant_ext_rotcache_{os.getpid()}_dev{args.device}_shard{args.shard_index}_of_{args.num_shards}"
        )
        os.environ["GPTQMODEL_PAROQUANT_BUILD_ROOT"] = str(paro_build_root)
        os.environ["GPTQMODEL_PAROQUANT_FORCE_REBUILD"] = "1"
    else:
        os.environ.pop("GPTQMODEL_PAROQUANT_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_PAROQUANT_FORCE_REBUILD", None)

    from gptqmodel.utils.awq import clear_awq_extension_cache, prewarm_awq_extension
    from gptqmodel.utils.paroquant import clear_paroquant_rotation_extension_cache, prewarm_paroquant_rotation_extension

    if args.force_rebuild_awq:
        clear_awq_extension_cache()
    if args.force_rebuild_paroquant:
        clear_paroquant_rotation_extension_cache()

    if not prewarm_awq_extension():
        raise RuntimeError("Failed to build/load the AWQ CUDA runtime.")
    if not prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=device,
    ):
        raise RuntimeError("Failed to build/load the ParoQuant CUDA rotation runtime.")


def _print_ascii(results: dict[str, Any]) -> None:
    print(f"Device: {results['device']} ({results['cuda_device']}, dtype={results['dtype']})")
    print()
    print("Accuracy")
    print(
        tabulate(
            [
                [
                    row["dtype"],
                    row["case_id"],
                    f"{row['batch']}x{row['seq']}",
                    f"{row['in_features']}->{row['out_features']}",
                    f"{row['baseline_dense_max_abs']:.6f}",
                    f"{row['candidate_dense_max_abs']:.6f}",
                    f"{row['baseline_candidate_max_abs']:.6f}",
                    f"{row['baseline_candidate_mean_abs']:.6f}",
                ]
                for row in results["rows"]
            ],
            headers=[
                "dtype",
                "case",
                "batch x seq",
                "shape",
                "rotation_cache_off vs dense max_abs",
                "rotation_cache_on vs dense max_abs",
                "off vs on max_abs",
                "off vs on mean_abs",
            ],
            tablefmt="plain",
        )
    )
    print()
    print("Benchmark")
    print(
        tabulate(
            [
                [
                    row["dtype"],
                    row["case_id"],
                    f"{row['batch']}x{row['seq']}",
                    f"{row['in_features']}->{row['out_features']}",
                    f"{row['baseline_ms']:.3f}",
                    f"{row['candidate_ms']:.3f}",
                    _format_speedup(row["speedup"]),
                    row["winner"],
                ]
                for row in results["rows"]
            ],
            headers=[
                "dtype",
                "case",
                "batch x seq",
                "shape",
                "rotation_cache_off ms",
                "rotation_cache_on ms",
                "speedup",
                "winner",
            ],
            tablefmt="plain",
        )
    )
    print()
    print(
        "Summary: "
        f"candidate_wins={results['candidate_wins']}/{results['case_count']}, "
        f"geo_mean_speedup={results['geo_mean_speedup']:.3f}x"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark ParoQuant BF16 rotation-metadata caching.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force-rebuild-awq", action="store_true")
    parser.add_argument("--force-rebuild-paroquant", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ParoQuant rotation-cache benchmark.")

    device = torch.device(f"cuda:{args.device}")
    _configure_runtime(args, device)
    results = run(
        device=device,
        dtype=_resolve_dtype(args.dtype),
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    _print_ascii(results)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
