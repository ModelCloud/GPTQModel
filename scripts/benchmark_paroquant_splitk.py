#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the production ParoQuant split-K path against the standard mega-kernel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from tabulate import tabulate

from benchmark_paroquant_triton_ab import (
    BenchCase,
    _build_module,
    _dense_reference,
    _make_quant_buffers,
    _megakernel_metadata_bytes,
    _megakernel_scratch_bytes,
    _nvidia_driver_version,
)
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear


DEFAULT_CASES = (
    (1, 512),
    (1, 2048),
    (1, 8192),
    (2, 2048),
    (3, 2048),
    (4, 2048),
    (5, 2048),
    (6, 2048),
    (7, 2048),
    (8, 2048),
)


def _parse_case(value: str) -> tuple[int, int]:
    try:
        rows_text, out_features_text = value.split(":", maxsplit=1)
        rows = int(rows_text)
        out_features = int(out_features_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be ROWS:OUT_FEATURES") from exc
    if rows <= 0 or out_features <= 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    return rows, out_features


def _stats(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    return {
        "p50_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "std_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
    }


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _paired_benchmark(
    standard: ParoQuantTritonLinear,
    split: ParoQuantTritonLinear,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], dict[str, float]]:
    modules = {"standard": standard, "split_k16": split}
    with torch.inference_mode():
        for iteration in range(warmup):
            order = ("standard", "split_k16") if iteration % 2 == 0 else ("split_k16", "standard")
            for name in order:
                modules[name](x)
        torch.cuda.synchronize(x.device)

        events = {
            name: (
                [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
                [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
            )
            for name in modules
        }
        for iteration in range(iters):
            order = ("standard", "split_k16") if iteration % 2 == 0 else ("split_k16", "standard")
            for name in order:
                starts, ends = events[name]
                starts[iteration].record()
                modules[name](x)
                ends[iteration].record()
        torch.cuda.synchronize(x.device)

    results = {}
    for name in modules:
        starts, ends = events[name]
        samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)]
        results[name] = _stats(samples_us)
    return results["standard"], results["split_k16"]


def _paired_graph_benchmark(
    standard: ParoQuantTritonLinear,
    split: ParoQuantTritonLinear,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, float], dict[str, float], torch.Tensor, torch.Tensor]:
    """Capture both schedules, validate their static outputs, and time alternating graph replays."""
    modules = {"standard": standard, "split_k16": split}
    graphs = {}
    outputs = {}
    with torch.inference_mode():
        torch.cuda.synchronize(x.device)
        for name, module in modules.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = module(x)
            graphs[name] = graph
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize(x.device)

        for iteration in range(warmup):
            order = ("standard", "split_k16") if iteration % 2 == 0 else ("split_k16", "standard")
            for name in order:
                graphs[name].replay()
        torch.cuda.synchronize(x.device)

        events = {
            name: (
                [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
                [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
            )
            for name in graphs
        }
        for iteration in range(iters):
            order = ("standard", "split_k16") if iteration % 2 == 0 else ("split_k16", "standard")
            for name in order:
                starts, ends = events[name]
                starts[iteration].record()
                graphs[name].replay()
                ends[iteration].record()
        torch.cuda.synchronize(x.device)

    results = {}
    for name in graphs:
        starts, ends = events[name]
        samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)]
        results[name] = _stats(samples_us)
    return results["standard"], results["split_k16"], outputs["standard"], outputs["split_k16"]


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    benchmark_rows = []
    accuracy_rows = []
    case_results = []

    for case_index, (rows, out_features) in enumerate(args.case):
        torch.manual_seed(args.seed + case_index)
        case = BenchCase(
            case_id=f"m{rows}_k2048_n{out_features}",
            batch=1,
            seq=rows,
            in_features=2048,
            out_features=out_features,
            group_size=128,
            krot=8,
        )
        buffers = _make_quant_buffers(case, dtype=dtype)
        standard = _build_module(ParoQuantTritonLinear, case, buffers, device, dtype=dtype)
        split = _build_module(ParoQuantTritonLinear, case, buffers, device, dtype=dtype)
        standard.paroquant_triton_autotune_enabled = False
        split.paroquant_triton_autotune_enabled = False
        standard.paroquant_triton_megakernel_decode_splitk_enabled = False
        standard.paroquant_triton_megakernel_prefill_splitk_enabled = False
        split.paroquant_triton_megakernel_decode_splitk_enabled = True
        split.paroquant_triton_megakernel_prefill_splitk_enabled = True
        x = torch.randn((case.batch, case.seq, case.in_features), device=device, dtype=dtype)

        with torch.inference_mode():
            dense = _dense_reference(standard, x)
            standard_out = standard(x)
            split_out = split(x)
        standard_dense = (standard_out - dense).abs().float()
        split_dense = (split_out - dense).abs().float()
        cross = (split_out - standard_out).abs().float()
        torch.testing.assert_close(split_out, standard_out, rtol=0.01, atol=2.0)
        if cross.mean().item() > 0.003:
            raise AssertionError(f"split-K mean drift exceeded 0.003: {cross.mean().item():.6f}")
        if split_dense.max().item() > standard_dense.max().item() + 2.0:
            raise AssertionError("split-K exceeded the standard mega-kernel's dense maximum-error envelope")
        if split_dense.mean().item() > standard_dense.mean().item() + 0.003:
            raise AssertionError("split-K exceeded the standard mega-kernel's dense mean-error envelope")

        if args.cuda_graph:
            standard_stats, split_stats, graph_standard, graph_split = _paired_graph_benchmark(
                standard,
                split,
                x,
                warmup=args.warmup,
                iters=args.iters,
            )
            torch.testing.assert_close(graph_standard, standard_out, rtol=0, atol=0)
            torch.testing.assert_close(graph_split, split_out, rtol=0, atol=0)
        else:
            standard_stats, split_stats = _paired_benchmark(
                standard,
                split,
                x,
                warmup=args.warmup,
                iters=args.iters,
            )
        mean_speedup = standard_stats["mean_us"] / split_stats["mean_us"]
        p50_speedup = standard_stats["p50_us"] / split_stats["p50_us"]
        standard_tps = rows * 1e6 / standard_stats["mean_us"]
        split_tps = rows * 1e6 / split_stats["mean_us"]
        scratch_bytes = _megakernel_scratch_bytes(split)
        metadata_bytes = _megakernel_metadata_bytes(split)
        compiled_launcher = any(
            compiled_kernel is not False for compiled_kernel in split._megakernel_splitk_compiled.values()
        )
        benchmark_rows.append(
            [
                f"M{rows} K2048 N{out_features}",
                "compiled" if compiled_launcher else "JIT",
                f"{standard_stats['p50_us']:.3f}/{standard_stats['mean_us']:.3f}/{standard_stats['p95_us']:.3f}",
                f"{split_stats['p50_us']:.3f}/{split_stats['mean_us']:.3f}/{split_stats['p95_us']:.3f}",
                f"{p50_speedup:.3f}x",
                f"{mean_speedup:.3f}x",
                f"{standard_tps:.1f}",
                f"{split_tps:.1f}",
                f"{scratch_bytes / 1024:.1f}",
            ]
        )
        accuracy_rows.append(
            [
                f"M{rows} K2048 N{out_features}",
                f"{standard_dense.max().item():.6f}/{standard_dense.mean().item():.6f}",
                f"{split_dense.max().item():.6f}/{split_dense.mean().item():.6f}",
                int(cross.count_nonzero().item()),
                f"{cross.max().item():.6f}",
                f"{cross.mean().item():.6f}",
            ]
        )
        case_results.append(
            {
                "case": asdict(case),
                "standard": standard_stats,
                "split_k16": split_stats,
                "speedup_p50": p50_speedup,
                "speedup_mean": mean_speedup,
                "standard_module_token_tps": standard_tps,
                "split_k16_module_token_tps": split_tps,
                "megakernel_metadata_bytes": metadata_bytes,
                "split_k16_scratch_bytes": scratch_bytes,
                "split_k16_compiled_launcher": compiled_launcher,
                "cuda_graph_replay": args.cuda_graph,
                "accuracy": {
                    "standard_dense_max_abs": standard_dense.max().item(),
                    "standard_dense_mean_abs": standard_dense.mean().item(),
                    "split_k16_dense_max_abs": split_dense.max().item(),
                    "split_k16_dense_mean_abs": split_dense.mean().item(),
                    "cross_mismatched": int(cross.count_nonzero().item()),
                    "cross_max_abs": cross.max().item(),
                    "cross_mean_abs": cross.mean().item(),
                },
            }
        )

    props = torch.cuda.get_device_properties(device)
    return {
        "git_revision": _git_revision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": args.device,
        "device": props.name,
        "device_uuid": str(props.uuid),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_driver_version": _nvidia_driver_version(),
        "triton_version": __import__("triton").__version__,
        "dtype": str(dtype).removeprefix("torch."),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "ordering": f"paired {'CUDA graph replay' if args.cuda_graph else 'eager'} AB/BA alternating",
        "cuda_graph_replay": args.cuda_graph,
        "compiled_launcher_gate": "Triton 3.7 internal ABI; JIT fallback otherwise",
        "benchmark_headers": [
            "shape",
            "launcher",
            "standard p50/mean/p95 us",
            "split-K16 p50/mean/p95 us",
            "p50 speedup",
            "mean speedup",
            "standard tok/s",
            "split-K16 tok/s",
            "scratch KiB",
        ],
        "benchmark_rows": benchmark_rows,
        "accuracy_headers": [
            "shape",
            "standard dense max/mean",
            "split-K16 dense max/mean",
            "cross mismatch",
            "cross max",
            "cross mean",
        ],
        "accuracy_rows": accuracy_rows,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA index within the visible device set")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--case",
        type=_parse_case,
        action="append",
        help="Shape as ROWS:OUT_FEATURES; defaults to the measured decode production gates",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-graph", action="store_true", help="Capture each module and benchmark graph replay")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.case is None:
        args.case = list(DEFAULT_CASES)

    payload = run(args)
    print(
        f"Device: {payload['device']} (visible={payload['cuda_visible_devices']}, uuid={payload['device_uuid']}, "
        f"sm={payload['compute_capability']}, SMs={payload['sm_count']}, dtype={payload['dtype']})"
    )
    print()
    print("Accuracy")
    print(tabulate(payload["accuracy_rows"], headers=payload["accuracy_headers"], tablefmt="grid"))
    print()
    print("Paired CUDA graph replay benchmark" if args.cuda_graph else "Paired eager benchmark")
    print(tabulate(payload["benchmark_rows"], headers=payload["benchmark_headers"], tablefmt="grid"))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
