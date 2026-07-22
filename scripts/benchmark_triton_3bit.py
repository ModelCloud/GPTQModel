#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul  # noqa: E402
from gptqmodel.nn_modules.triton_utils.three_bit import (  # noqa: E402
    LAYOUT_AWQ,
    LAYOUT_GPTQ,
    Triton3BitLaunchConfig,
    dequantize_3bit,
    matmul_3bit,
    matmul_marlin_3bit,
    matmul_trilin_3bit,
    pack_3bit,
    prepare_marlin_3bit,
    prepare_trilin_3bit,
    repack_awq_to_gptq_3bit,
)


BITS = 3
DEFAULT_GROUP_SIZE = 128
ZERO = 1 << (BITS - 1)


def _parse_shape(value: str) -> tuple[int, int, int]:
    fields = value.lower().replace("x", ",").split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(f"shape must be MxKxN, got `{value}`")
    try:
        shape = tuple(int(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape must contain integers, got `{value}`") from exc
    m, k, n = shape
    if m <= 0 or k <= 0 or n <= 0 or k % 32 != 0 or n % 32 != 0:
        raise argparse.ArgumentTypeError(
            f"shape must have M>0, K divisible by 32, and N divisible by 32, got {shape}"
        )
    return shape


def _dtype(value: str) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    mapping = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16}
    if normalized not in mapping:
        raise argparse.ArgumentTypeError(f"dtype must be fp16 or bf16, got `{value}`")
    return mapping[normalized]


def _launch_config(value: str) -> Triton3BitLaunchConfig:
    fields = value.lower().replace("x", ",").split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            f"launch config must be BLOCK_MxBLOCK_NxBLOCK_KxWARPSxSTAGES, got `{value}`"
        )
    try:
        return Triton3BitLaunchConfig(*(int(field) for field in fields))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _device_metadata(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    pci = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    return {
        "device": str(device),
        "name": properties.name,
        "uuid": str(properties.uuid),
        "pci_bus_id": pci,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "l2_bytes": properties.L2_cache_size,
        "memory_clock_khz": properties.memory_clock_rate,
        "memory_bus_width_bits": properties.memory_bus_width,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": sys.version.replace("\n", " "),
    }


def _make_case(
    *,
    layout: str,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    group_size: int,
    launch_config: Triton3BitLaunchConfig | None,
    include_marlin_expanded: bool,
    include_native_trilin: bool,
) -> dict[str, torch.Tensor | Callable[[], torch.Tensor]]:
    m, k, n = shape
    effective_group_size = k if group_size == -1 else group_size
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes_cpu = torch.randint(0, 8, (k, n), dtype=torch.int32, generator=generator)
    packed_axis = 0 if layout == LAYOUT_GPTQ else 1
    qweight = pack_3bit(codes_cpu, axis=packed_axis).to(device)
    scales = (
        torch.rand((k // effective_group_size, n), dtype=torch.float32, generator=generator)
        .mul_(0.5)
        .add_(0.03125)
        .to(device=device, dtype=torch.float16)
    )
    x = torch.randn((m, k), dtype=torch.float32, generator=generator).to(device=device, dtype=dtype)
    dense_weight = dequantize_3bit(
        qweight,
        scales,
        layout=layout,
        group_size=group_size,
    ).to(dtype=dtype)

    runtime_qweight = repack_awq_to_gptq_3bit(qweight) if layout == LAYOUT_AWQ else qweight
    runtime_layout = LAYOUT_GPTQ if layout == LAYOUT_AWQ else layout

    def fused() -> torch.Tensor:
        return matmul_3bit(
            x,
            runtime_qweight,
            scales,
            layout=runtime_layout,
            group_size=group_size,
            launch_config=launch_config,
        )

    def dense() -> torch.Tensor:
        return torch.matmul(x, dense_weight)

    if layout == LAYOUT_GPTQ:
        qzeros = pack_3bit(
            torch.full((k // effective_group_size, n), ZERO, dtype=torch.int32),
            axis=1,
        ).to(device)
        g_idx = (torch.arange(k, dtype=torch.int32, device=device) // effective_group_size).contiguous()

        def materialize_matmul() -> torch.Tensor:
            return quant_matmul(
                x,
                qweight,
                scales,
                qzeros,
                g_idx,
                bits=BITS,
                pack_bits=32,
                maxq=7,
            )

    else:

        def serialized_layout_fused() -> torch.Tensor:
            return matmul_3bit(
                x,
                qweight,
                scales,
                layout=LAYOUT_AWQ,
                group_size=group_size,
                launch_config=launch_config,
            )

        def materialize_matmul() -> torch.Tensor:
            weight = dequantize_3bit(
                qweight,
                scales,
                layout=layout,
                group_size=group_size,
            ).to(dtype=dtype)
            return torch.matmul(x, weight)

    marlin_expanded = None
    if include_marlin_expanded and dtype == torch.float16:
        marlin_state = prepare_marlin_3bit(runtime_qweight, scales, group_size)
        if marlin_state is not None:

            def marlin_expanded() -> torch.Tensor:
                return matmul_marlin_3bit(
                    x,
                    marlin_state.qweight,
                    marlin_state.scales,
                    marlin_state.workspace,
                    marlin_state.empty,
                    k=k,
                    n=n,
                )

    native_trilin = None
    if include_native_trilin and prepare_trilin_3bit(runtime_qweight, scales, group_size):

        def native_trilin() -> torch.Tensor:
            return matmul_trilin_3bit(x, runtime_qweight, scales, group_size=group_size)

    if m <= 16 and callable(native_trilin):
        production_hybrid = native_trilin
    elif callable(marlin_expanded):
        production_hybrid = marlin_expanded
    else:
        production_hybrid = fused

    production_fused = fused
    alternative_layout_fused = None
    if layout == LAYOUT_AWQ:
        alternative_layout_fused = serialized_layout_fused

    return {
        "x": x,
        "qweight": qweight,
        "scales": scales,
        "dense_weight": dense_weight,
        "fused": production_fused,
        "dense": dense,
        "materialize_matmul": materialize_matmul,
        "alternative_layout_fused": alternative_layout_fused,
        "marlin_expanded": marlin_expanded,
        "native_trilin": native_trilin,
        "production_hybrid": production_hybrid,
    }


def _make_clock_warmup(
    *,
    device: torch.device,
    dtype: torch.dtype,
    iterations: int,
) -> Callable[[], None] | None:
    if iterations <= 0:
        return None
    lhs = torch.randn((4096, 4096), dtype=dtype, device=device)
    rhs = torch.randn((4096, 4096), dtype=dtype, device=device)
    output = torch.empty_like(lhs)

    def warmup_clock() -> None:
        for _ in range(iterations):
            torch.mm(lhs, rhs, out=output)
        torch.cuda.synchronize(device)

    return warmup_clock


def _measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    clock_warmup: Callable[[], None] | None,
) -> dict[str, float]:
    if clock_warmup is not None:
        clock_warmup()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    samples = [starts[index].elapsed_time(ends[index]) for index in range(iterations)]
    ordered = sorted(samples)
    p95_index = min(iterations - 1, math.ceil(iterations * 0.95) - 1)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "std_ms": statistics.stdev(samples) if iterations > 1 else 0.0,
        "min_ms": ordered[0],
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


def _numerical_error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    absolute = (actual.float() - reference.float()).abs()
    relative = absolute / reference.float().abs().clamp_min(1e-5)
    return {
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "max_rel": relative.max().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _render_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    values = [[str(row[key]) for key, _ in columns] for row in rows]
    widths = [len(label) for _, label in columns]
    for row in values:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = "| " + " | ".join(label.ljust(width) for (_, label), width in zip(columns, widths)) + " |"
    body = [
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
        for row in values
    ]
    return "\n".join([separator, header, separator, *body, separator])


def _profile(
    case: dict[str, object],
    *,
    method: str,
    layout: str,
    group_size: int,
    shape: tuple[int, int, int],
    iterations: int,
    clock_warmup: Callable[[], None] | None,
) -> None:
    method_keys = {
        "fused": "fused",
        "native-trilin": "native_trilin",
        "marlin-expanded": "marlin_expanded",
        "production-hybrid": "production_hybrid",
    }
    selected = case[method_keys[method]]
    if not callable(selected):
        raise ValueError(f"profile method `{method}` was not enabled for this benchmark case")
    if clock_warmup is not None:
        clock_warmup()
    for _ in range(20):
        selected()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(
        f"3bit:{method}:{layout}:G{group_size}:M{shape[0]}:K{shape[1]}:N{shape[2]}"
    )
    for _ in range(iterations):
        selected()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark symmetric grouped GPTQ/AWQ 3-bit kernels.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layout", choices=[LAYOUT_GPTQ, LAYOUT_AWQ, "both"], default="both")
    parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    parser.add_argument(
        "--group-size",
        type=int,
        action="append",
        dest="group_sizes",
        help="Repeatable quantization group size; use -1 for channelwise (default: 128).",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=None,
        help="Repeatable MxKxN shape (default: 1/16/128 x 4096 x 4096).",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--clock-warmup-iterations",
        type=int,
        default=128,
        help="Unmeasured 4096x4096 GEMMs before each method to stabilize GPU clocks; set 0 to disable.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--launch-config",
        type=_launch_config,
        help="Optional BLOCK_MxBLOCK_NxBLOCK_KxWARPSxSTAGES override for controlled launch sweeps.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument(
        "--profile-method",
        choices=["fused", "native-trilin", "marlin-expanded", "production-hybrid"],
        default="fused",
    )
    parser.add_argument(
        "--include-marlin-expanded",
        action="store_true",
        help="Benchmark one-time 3-bit-to-uint4b8 expansion through native Marlin.",
    )
    parser.add_argument(
        "--include-native-trilin",
        action="store_true",
        help="Benchmark the true continuous-3-bit native Trilin CUDA kernel.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"benchmark requires a CUDA device, got {device}")
    torch.cuda.set_device(device)
    metadata = _device_metadata(device)
    shapes = args.shape or [(1, 4096, 4096), (16, 4096, 4096), (128, 4096, 4096)]
    group_sizes = args.group_sizes or [DEFAULT_GROUP_SIZE]
    for group_size in group_sizes:
        for _, k, _ in shapes:
            effective_group_size = k if group_size == -1 else group_size
            if effective_group_size <= 0 or k % effective_group_size != 0:
                raise ValueError(f"--group-size {group_size} must be -1 or a positive divisor of K={k}")
    layouts = [LAYOUT_GPTQ, LAYOUT_AWQ] if args.layout == "both" else [args.layout]
    clock_warmup = _make_clock_warmup(
        device=device,
        dtype=args.dtype,
        iterations=args.clock_warmup_iterations,
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.profile and (len(shapes) != 1 or len(layouts) != 1 or len(group_sizes) != 1):
        raise ValueError("--profile requires exactly one --shape, --group-size, and concrete --layout")

    results: list[dict[str, object]] = []
    for layout in layouts:
        for group_size in group_sizes:
            for shape in shapes:
                case = _make_case(
                    layout=layout,
                    shape=shape,
                    dtype=args.dtype,
                    device=device,
                    seed=args.seed,
                    group_size=group_size,
                    launch_config=args.launch_config,
                    include_marlin_expanded=args.include_marlin_expanded,
                    include_native_trilin=args.include_native_trilin,
                )
                if args.profile:
                    _profile(
                        case,
                        method=args.profile_method,
                        layout=layout,
                        group_size=group_size,
                        shape=shape,
                        iterations=args.profile_iterations,
                        clock_warmup=clock_warmup,
                    )
                    print(
                        f"profiled method={args.profile_method} layout={layout} group_size={group_size} "
                        f"shape={shape} iterations={args.profile_iterations}"
                    )
                    return

                fused = case["fused"]
                dense = case["dense"]
                materialize_matmul = case["materialize_matmul"]
                assert callable(fused) and callable(dense) and callable(materialize_matmul)
                dense_output = dense()
                fused_error = _numerical_error(fused(), dense_output)
                materialized_error = _numerical_error(materialize_matmul(), dense_output)
                if not fused_error["finite"] or fused_error["max_abs"] > 8.0:
                    raise AssertionError(f"fused numerical check failed: {fused_error}")

                marlin_expanded = case["marlin_expanded"]
                marlin_error = None
                if callable(marlin_expanded):
                    marlin_error = _numerical_error(marlin_expanded(), dense_output)
                    if not marlin_error["finite"] or marlin_error["max_abs"] > 8.0:
                        raise AssertionError(f"expanded Marlin numerical check failed: {marlin_error}")

                native_trilin = case["native_trilin"]
                native_trilin_error = None
                if callable(native_trilin):
                    native_trilin_error = _numerical_error(native_trilin(), dense_output)
                    if not native_trilin_error["finite"] or native_trilin_error["max_abs"] > 8.0:
                        raise AssertionError(f"native Trilin numerical check failed: {native_trilin_error}")

                methods = {
                    "fused": fused,
                    "materialize+mm": materialize_matmul,
                    "dense-mm": dense,
                }
                alternative_layout_fused = case["alternative_layout_fused"]
                if callable(alternative_layout_fused):
                    methods["alternate-layout-fused"] = alternative_layout_fused
                if callable(marlin_expanded):
                    methods["native-marlin-expanded"] = marlin_expanded
                if callable(native_trilin):
                    methods["native-trilin-3bit"] = native_trilin
                production_hybrid = case["production_hybrid"]
                if callable(production_hybrid):
                    methods["production-hybrid"] = production_hybrid
                timings = {
                    name: _measure(
                        fn,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        clock_warmup=clock_warmup,
                    )
                    for name, fn in methods.items()
                }
                baseline_median = timings["materialize+mm"]["median_ms"]
                m, k, n = shape
                for name, timing in timings.items():
                    median_s = timing["median_ms"] / 1000
                    results.append(
                        {
                            "layout": layout,
                            "dtype": str(args.dtype).replace("torch.", ""),
                            "group_size": group_size,
                            "m": m,
                            "k": k,
                            "n": n,
                            "method": name,
                            **timing,
                            "tflops": 2 * m * n * k / median_s / 1e12,
                            "speedup_vs_materialize": baseline_median / timing["median_ms"],
                            "fused_max_abs": fused_error["max_abs"],
                            "fused_mean_abs": fused_error["mean_abs"],
                            "materialized_max_abs": materialized_error["max_abs"],
                            "marlin_expanded_max_abs": marlin_error["max_abs"] if marlin_error else None,
                            "native_trilin_max_abs": native_trilin_error["max_abs"] if native_trilin_error else None,
                        }
                    )

    display_rows = []
    for row in results:
        display_rows.append(
            {
                "layout": row["layout"],
                "dtype": row["dtype"],
                "group_size": row["group_size"],
                "m": row["m"],
                "k": row["k"],
                "n": row["n"],
                "method": row["method"],
                "median": f"{row['median_ms']:.4f}",
                "mean": f"{row['mean_ms']:.4f}",
                "std": f"{row['std_ms']:.4f}",
                "p95": f"{row['p95_ms']:.4f}",
                "tflops": f"{row['tflops']:.3f}",
                "speedup": f"{row['speedup_vs_materialize']:.2f}x",
                "max_abs": f"{row['fused_max_abs']:.5f}",
            }
        )
    print(
        _render_table(
            display_rows,
            [
                ("layout", "layout"),
                ("dtype", "dtype"),
                ("group_size", "group"),
                ("m", "M"),
                ("k", "K"),
                ("n", "N"),
                ("method", "method"),
                ("median", "median ms"),
                ("mean", "mean ms"),
                ("std", "std ms"),
                ("p95", "p95 ms"),
                ("tflops", "TFLOP/s"),
                ("speedup", "vs materialize"),
                ("max_abs", "fused max abs"),
            ],
        )
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "config": {
                "bits": BITS,
                "group_sizes": group_sizes,
                "desc_act": False,
                "sym": True,
                "dtype": str(args.dtype),
                "warmup": args.warmup,
                "clock_warmup_iterations": args.clock_warmup_iterations,
                "iterations": args.iterations,
                "seed": args.seed,
                "launch_config": vars(args.launch_config) if args.launch_config else None,
                "awq_runtime_repack": True,
                "include_marlin_expanded": args.include_marlin_expanded,
                "include_native_trilin": args.include_native_trilin,
            },
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
