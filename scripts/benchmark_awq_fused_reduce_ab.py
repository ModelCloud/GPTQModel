#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import tempfile
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

_TEMP_BUILD_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _register_temp_build_dir(prefix: str) -> Path:
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    _TEMP_BUILD_DIRS.append(temp_dir)
    return Path(temp_dir.name)


atexit.register(lambda: [temp_dir.cleanup() for temp_dir in reversed(_TEMP_BUILD_DIRS)])


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
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
    }


def _dense_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    return torch.matmul(x, dense_weight)


def _set_fused_reduce_disabled(disabled: bool) -> None:
    if disabled:
        os.environ["GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE"] = "1"
    else:
        os.environ.pop("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE", None)


def _benchmark_ms(fn, device: torch.device, warmup: int, iters: int, fused_reduce_disabled: bool) -> float:
    _set_fused_reduce_disabled(fused_reduce_disabled)
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3 / iters


def _format_speedup(speedup: float) -> str:
    return f"{speedup:.3f}x"


def _label(disabled: bool) -> str:
    return "fused_reduce_off" if disabled else "fused_reduce_on"


def _run_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    from gptqmodel.utils.awq import awq_gemm_forward

    return awq_gemm_forward(
        x,
        qweight,
        scales,
        qzeros,
        split_k_iters,
        fp32_accum=True,
    )


def _run_suite(
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    quick: bool,
    rotate_inputs: bool,
    shard_index: int,
    num_shards: int,
    split_k_iters: int,
    baseline_disable_fused_reduce: bool,
    candidate_disable_fused_reduce: bool,
) -> dict[str, Any]:
    from gptqmodel.utils.paroquant import apply_paroquant_rotation

    cases = _subset_cases(QUICK_CASES if quick else DEFAULT_CASES, shard_index=shard_index, num_shards=num_shards)
    rows = []
    speedups = []
    candidate_wins = 0
    baseline_label = _label(baseline_disable_fused_reduce)
    candidate_label = _label(candidate_disable_fused_reduce)

    for index, case in enumerate(cases):
        torch.manual_seed(1000 + index)
        buffers = _make_quant_buffers(case, dtype=dtype)
        qweight = buffers["qweight"].to(device)
        qzeros = buffers["qzeros"].to(device)
        scales = buffers["scales"].to(device)
        x = torch.randn((case.batch * case.seq, case.in_features), device=device, dtype=dtype)

        kernel_input = x
        if rotate_inputs:
            kernel_input = apply_paroquant_rotation(
                x,
                buffers["pairs"].to(device),
                buffers["theta"].to(device),
                scales=buffers["channel_scales"].to(device),
                group_size=case.group_size,
            )

        with torch.inference_mode():
            dense = _dense_reference(kernel_input, qweight, qzeros, scales, bits=4, group_size=case.group_size)
            _set_fused_reduce_disabled(baseline_disable_fused_reduce)
            baseline = _run_gemm(kernel_input, qweight, scales, qzeros, split_k_iters)
            _set_fused_reduce_disabled(candidate_disable_fused_reduce)
            candidate = _run_gemm(kernel_input, qweight, scales, qzeros, split_k_iters)

        baseline_dense = (baseline - dense).abs()
        candidate_dense = (candidate - dense).abs()
        baseline_candidate = (baseline - candidate).abs()

        baseline_ms = _benchmark_ms(
            lambda: _run_gemm(kernel_input, qweight, scales, qzeros, split_k_iters),
            device=device,
            warmup=warmup,
            iters=iters,
            fused_reduce_disabled=baseline_disable_fused_reduce,
        )
        candidate_ms = _benchmark_ms(
            lambda: _run_gemm(kernel_input, qweight, scales, qzeros, split_k_iters),
            device=device,
            warmup=warmup,
            iters=iters,
            fused_reduce_disabled=candidate_disable_fused_reduce,
        )
        speedup = baseline_ms / candidate_ms
        speedups.append(speedup)
        winner = candidate_label if candidate_ms < baseline_ms else baseline_label
        candidate_wins += int(candidate_ms < baseline_ms)

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
                "winner": winner,
                "baseline_dense_max_abs": baseline_dense.max().item(),
                "candidate_dense_max_abs": candidate_dense.max().item(),
                "baseline_candidate_max_abs": baseline_candidate.max().item(),
                "baseline_candidate_mean_abs": baseline_candidate.mean().item(),
            }
        )

    geo_mean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups)) if speedups else float("nan")
    return {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "rows": rows,
        "geo_mean_speedup": geo_mean_speedup,
        "candidate_wins": candidate_wins,
        "case_count": len(speedups),
    }


def run(
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    quick: bool,
    shard_index: int,
    num_shards: int,
    split_k_iters: int,
    baseline_disable_fused_reduce: bool,
    candidate_disable_fused_reduce: bool,
) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    return {
        "device": torch.cuda.get_device_name(device),
        "cuda_device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "iters": iters,
        "quick": quick,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "split_k_iters": split_k_iters,
        "awq": _run_suite(
            device=device,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            quick=quick,
            rotate_inputs=False,
            shard_index=shard_index,
            num_shards=num_shards,
            split_k_iters=split_k_iters,
            baseline_disable_fused_reduce=baseline_disable_fused_reduce,
            candidate_disable_fused_reduce=candidate_disable_fused_reduce,
        ),
        "paroquant": _run_suite(
            device=device,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            quick=quick,
            rotate_inputs=True,
            shard_index=shard_index,
            num_shards=num_shards,
            split_k_iters=split_k_iters,
            baseline_disable_fused_reduce=baseline_disable_fused_reduce,
            candidate_disable_fused_reduce=candidate_disable_fused_reduce,
        ),
    }


def _print_suite(name: str, results: dict[str, Any]) -> None:
    print(name)
    print(f"Configs: baseline={results['baseline_label']} candidate={results['candidate_label']}")
    print("Accuracy")
    print(
        tabulate(
            [
                [
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
                "case",
                "batch x seq",
                "shape",
                "baseline vs dense max_abs",
                "candidate vs dense max_abs",
                "baseline vs candidate max_abs",
                "baseline vs candidate mean_abs",
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
                "case",
                "batch x seq",
                "shape",
                "baseline ms",
                "candidate ms",
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
    print()


def _configure_awq_runtime(args: argparse.Namespace) -> None:
    if args.force_rebuild_awq:
        build_root = _register_temp_build_dir(
            f"awq_jit_fusedreduce_dev{args.device}_shard{args.shard_index}_of_{args.num_shards}_"
        )
        os.environ["GPTQMODEL_AWQ_BUILD_ROOT"] = str(build_root)
        os.environ["GPTQMODEL_AWQ_FORCE_REBUILD"] = "1"
    else:
        os.environ.pop("GPTQMODEL_AWQ_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_AWQ_FORCE_REBUILD", None)

    from gptqmodel.utils.awq import awq_runtime_error, clear_awq_extension_cache, prewarm_awq_extension

    if args.force_rebuild_awq:
        clear_awq_extension_cache()

    if not prewarm_awq_extension():
        raise RuntimeError(f"Failed to build/load the AWQ CUDA runtime: {awq_runtime_error()}")


def _configure_paroquant_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if args.force_rebuild_paroquant:
        build_root = _register_temp_build_dir(
            f"paroquant_ext_fusedreduce_dev{args.device}_shard{args.shard_index}_of_{args.num_shards}_"
        )
        os.environ["GPTQMODEL_PAROQUANT_BUILD_ROOT"] = str(build_root)
        os.environ["GPTQMODEL_PAROQUANT_FORCE_REBUILD"] = "1"
    else:
        os.environ.pop("GPTQMODEL_PAROQUANT_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_PAROQUANT_FORCE_REBUILD", None)

    from gptqmodel.utils.paroquant import clear_paroquant_rotation_extension_cache, prewarm_paroquant_rotation_extension

    if args.force_rebuild_paroquant:
        clear_paroquant_rotation_extension_cache()

    if not prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=device,
    ):
        raise RuntimeError("Failed to build/load the fused ParoQuant CUDA rotation extension.")


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark AWQ fused split-K reduction in AWQ and ParoQuant paths.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--split-k-iters", type=int, default=4)
    parser.add_argument("--baseline-disable-fused-reduce", action="store_true")
    parser.add_argument("--candidate-disable-fused-reduce", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force-rebuild-awq", action="store_true")
    parser.add_argument("--force-rebuild-paroquant", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AWQ fused-reduce benchmark.")

    device = torch.device(f"cuda:{args.device}")
    _configure_awq_runtime(args)
    _configure_paroquant_runtime(args, device)
    results = run(
        device=device,
        dtype=_resolve_dtype(args.dtype),
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        split_k_iters=args.split_k_iters,
        baseline_disable_fused_reduce=args.baseline_disable_fused_reduce,
        candidate_disable_fused_reduce=args.candidate_disable_fused_reduce,
    )

    print(f"Device: {results['device']} ({results['cuda_device']}, dtype={results['dtype']})")
    print()
    _print_suite("AWQ", results["awq"])
    _print_suite("ParoQuant", results["paroquant"])

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
