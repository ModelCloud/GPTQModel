#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Scan Marlin packed-prefill configs over real transformer projection shapes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Callable


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_idle_preflight import (  # noqa: E402
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)


_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

import torch  # noqa: E402
from tabulate import tabulate  # noqa: E402

from gptqmodel import extension  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.utils.cpp import resolved_cuda_arch_flags  # noqa: E402
from gptqmodel.utils.marlin import _marlin_extra_cflags, _marlin_extra_cuda_cflags  # noqa: E402
from scripts.benchmark_amplin_vs_marlin import (  # noqa: E402
    _build_marlin,
    _dequantized_reference,
    _make_case,
)


DEFAULT_M_VALUES = (512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8191, 8192)
DEFAULT_CONFIGS = (1, 2, 3, 4)
BITS = 4
GROUP_SIZE = 128


@dataclass(frozen=True)
class ProjectionRole:
    name: str
    size_k: int
    size_n: int
    count: int


@dataclass(frozen=True)
class ShapeCase:
    model: str
    size_k: int
    size_n: int
    count: int
    roles: tuple[str, ...]


@dataclass(frozen=True)
class TimingStats:
    samples: int
    median_ms: float
    mean_ms: float
    std_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    round_medians_ms: tuple[float, ...]


def _parse_positive_ints(raw: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must contain comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{option} must contain positive integers")
    return values


def _parse_configs(raw: str) -> tuple[int, ...]:
    values = _parse_positive_ints(raw, option="--configs")
    if any(value not in DEFAULT_CONFIGS for value in values):
        raise argparse.ArgumentTypeError("--configs only supports 1,2,3,4")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        action="append",
        type=Path,
        dest="model_configs",
        required=True,
        help="Repeat once for each model config.json to include in the projection inventory.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument(
        "--m-values",
        type=lambda raw: _parse_positive_ints(raw, option="--m-values"),
        default=DEFAULT_M_VALUES,
        help="Comma-separated rows. Defaults include 512-8192 powers of two and partial-64 M tails.",
    )
    parser.add_argument(
        "--configs",
        type=_parse_configs,
        default=DEFAULT_CONFIGS,
        help="Comma-separated packed-prefill configs from 1 through 4.",
    )
    parser.add_argument("--shape-pattern", help="Optional regex matched against model:KxN:roles.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--dense-reference-m",
        type=int,
        default=513,
        help="M used once per K/N/dtype for an FP32 dequantized reference; 0 disables it.",
    )
    parser.add_argument("--json-out", type=Path)
    add_gpu_idle_preflight_args(parser)
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 2 or args.rounds < 1:
        parser.error("--warmup must be positive, --iters at least 2, and --rounds positive")
    if args.dense_reference_m < 0:
        parser.error("--dense-reference-m must be non-negative")
    return args


def _projection_roles_from_config(config_path: Path) -> tuple[str, list[ProjectionRole]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    attention_heads = int(config["num_attention_heads"])
    key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // attention_heads)
    q_size = attention_heads * head_dim
    kv_size = key_value_heads * head_dim
    model_name = config_path.parent.name
    roles = [
        ProjectionRole("q_proj", hidden_size, q_size, layers),
        ProjectionRole("k_proj", hidden_size, kv_size, layers),
        ProjectionRole("v_proj", hidden_size, kv_size, layers),
        ProjectionRole("o_proj", q_size, hidden_size, layers),
        ProjectionRole("gate_proj", hidden_size, intermediate_size, layers),
        ProjectionRole("up_proj", hidden_size, intermediate_size, layers),
        ProjectionRole("down_proj", intermediate_size, hidden_size, layers),
    ]
    return model_name, roles


def _shape_cases_from_config(config_path: Path) -> list[ShapeCase]:
    model_name, roles = _projection_roles_from_config(config_path)
    counts: Counter[tuple[int, int]] = Counter()
    role_names: dict[tuple[int, int], list[str]] = {}
    for role in roles:
        shape = (role.size_k, role.size_n)
        counts[shape] += role.count
        role_names.setdefault(shape, []).append(role.name)
    return [
        ShapeCase(
            model=model_name,
            size_k=size_k,
            size_n=size_n,
            count=counts[(size_k, size_n)],
            roles=tuple(role_names[(size_k, size_n)]),
        )
        for size_k, size_n in sorted(counts)
    ]


def _resolve_dtypes(name: str) -> tuple[torch.dtype, ...]:
    if name == "fp16":
        return (torch.float16,)
    if name == "bf16":
        return (torch.bfloat16,)
    return (torch.float16, torch.bfloat16)


def _dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "bf16"


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _raw_marlin_call(
    *,
    op,
    input: torch.Tensor,
    output: torch.Tensor,
    module: MarlinLinear,
    packed_prefill_config: int,
) -> torch.Tensor:
    return op(
        input,
        output,
        module.qweight,
        None,
        module.scales,
        None,
        module.qzeros,
        module.g_idx,
        module.g_idx_sort_indices,
        module.workspace,
        module.weight_type.id,
        input.shape[0],
        module.out_features,
        module.in_features,
        module.is_k_full,
        False,
        module.fp32,
        False,
        packed_prefill_config != 0,
        packed_prefill_config,
    )


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    rounds: int,
) -> dict[str, TimingStats]:
    with torch.inference_mode():
        for function in functions.values():
            for _ in range(warmup):
                function()
        torch.cuda.synchronize(device)
        if _GPU_IDLE_PREFLIGHT is not None:
            recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT)

        samples: dict[str, list[float]] = {name: [] for name in functions}
        round_medians: dict[str, list[float]] = {name: [] for name in functions}
        names = list(functions)
        for round_index in range(rounds):
            # Reverse each round to reduce clock and launch-order bias.
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                function = functions[name]
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                for index in range(iters):
                    starts[index].record()
                    function()
                    ends[index].record()
                torch.cuda.synchronize(device)
                round_values = [starts[index].elapsed_time(ends[index]) for index in range(iters)]
                samples[name].extend(round_values)
                round_medians[name].append(statistics.median(round_values))

    return {
        name: TimingStats(
            samples=len(values),
            median_ms=statistics.median(values),
            mean_ms=statistics.mean(values),
            std_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
            p95_ms=_percentile(values, 0.95),
            min_ms=min(values),
            max_ms=max(values),
            round_medians_ms=tuple(round_medians[name]),
        )
        for name, values in samples.items()
    }


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = actual_f32 - expected_f32
    abs_delta = delta.abs()
    expected_norm = torch.linalg.vector_norm(expected_f32)
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "mae": abs_delta.mean().item(),
        "rmse": torch.sqrt(torch.mean(delta.square())).item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / expected_norm.clamp_min(torch.finfo(torch.float32).tiny)
        ).item(),
        "max_abs": abs_delta.max().item(),
    }


def _dense_reference_check(
    *,
    device: torch.device,
    dtype: torch.dtype,
    size_m: int,
    size_k: int,
    size_n: int,
    seed: int,
    configs: tuple[int, ...],
) -> dict[str, object]:
    input, qweight, scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        seed=seed,
    )
    module = _build_marlin(device=device, dtype=dtype, qweight=qweight, scales=scales)
    extension_name = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
    op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
    op = extension.op(extension_name, op_name)
    output = torch.empty((size_m, size_n), dtype=dtype, device=device)
    with torch.inference_mode():
        reference = _dequantized_reference(input, qweight, scales)
        ordinary = _raw_marlin_call(
            op=op,
            input=input,
            output=output,
            module=module,
            packed_prefill_config=0,
        ).clone()
        packed = {}
        for config in configs:
            packed_output = _raw_marlin_call(
                op=op,
                input=input,
                output=output,
                module=module,
                packed_prefill_config=config,
            ).clone()
            packed[str(config)] = {
                "vs_fp32_dequant": _error_metrics(packed_output, reference),
                "vs_ordinary_marlin": _error_metrics(packed_output, ordinary),
            }
        torch.cuda.synchronize(device)
    return {
        "m": size_m,
        "ordinary_vs_fp32_dequant": _error_metrics(ordinary, reference),
        "packed_configs": packed,
    }


def _hardware_metadata(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    visible_devices = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    gpu_uuid = _GPU_IDLE_PREFLIGHT.uuid if _GPU_IDLE_PREFLIGHT is not None else None
    if not gpu_uuid:
        gpu_uuid = getattr(properties, "uuid", None)
    if not gpu_uuid and device.index is not None and device.index < len(visible_devices):
        visible_device = visible_devices[device.index]
        if visible_device.startswith("GPU-"):
            gpu_uuid = visible_device
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "device": str(device),
        "name": properties.name,
        "gpu_uuid": str(gpu_uuid) if gpu_uuid else None,
        "compute_capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "shared_memory_per_block_optin": properties.shared_memory_per_block_optin,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": driver[0].strip() if driver else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_revision": git_revision or None,
        "build": {
            "cflags": _marlin_extra_cflags(),
            "cuda_cflags": _marlin_extra_cuda_cflags(),
            "cuda_arch_flags": resolved_cuda_arch_flags(),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "nvcc_threads": os.environ.get("NVCC_THREADS"),
            "jit_opt_level": os.environ.get("GPTQMODEL_JIT_OPT_LEVEL"),
        },
    }


def _format_live_table(rows: list[dict[str, object]]) -> str:
    table = []
    for row in rows:
        table.append(
            [
                row["model"],
                row["dtype"],
                row["m"],
                row["k"],
                row["n"],
                row["route"],
                f"{row['median_ms']:.5f}",
                f"{row['p95_ms']:.5f}",
                f"{row['speedup_vs_marlin']:.3f}x",
                f"{row['max_abs_vs_marlin']:.6f}",
                "pass" if row["finite"] else "FAIL",
            ]
        )
    return tabulate(
        table,
        headers=("model", "dtype", "M", "K", "N", "route", "p50 ms", "p95 ms", "speedup", "max abs", "finite"),
        tablefmt="grid",
    )


def _scan_shape(
    *,
    case: ShapeCase,
    dtype: torch.dtype,
    m_values: tuple[int, ...],
    configs: tuple[int, ...],
    device: torch.device,
    warmup: int,
    iters: int,
    rounds: int,
    seed: int,
) -> list[dict[str, object]]:
    max_m = max(m_values)
    _, qweight, scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=1,
        size_k=case.size_k,
        size_n=case.size_n,
        seed=seed,
    )
    module = _build_marlin(device=device, dtype=dtype, qweight=qweight, scales=scales)
    extension_name = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
    op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
    op = extension.op(extension_name, op_name)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    input_storage = torch.randn(
        (max_m, case.size_k),
        device=device,
        dtype=dtype,
        generator=generator,
    ).mul_(0.25)

    rows: list[dict[str, object]] = []
    for size_m in m_values:
        input = input_storage[:size_m]
        outputs = {
            "marlin": torch.empty((size_m, case.size_n), dtype=dtype, device=device),
            **{
                f"config_{config}": torch.empty((size_m, case.size_n), dtype=dtype, device=device)
                for config in configs
            },
        }
        functions = {
            "marlin": lambda: _raw_marlin_call(
                op=op,
                input=input,
                output=outputs["marlin"],
                module=module,
                packed_prefill_config=0,
            )
        }
        for config in configs:
            functions[f"config_{config}"] = (
                lambda selected=config: _raw_marlin_call(
                    op=op,
                    input=input,
                    output=outputs[f"config_{selected}"],
                    module=module,
                    packed_prefill_config=selected,
                )
            )

        timing = _measure(
            functions,
            device=device,
            warmup=warmup,
            iters=iters,
            rounds=rounds,
        )
        with torch.inference_mode():
            reference = functions["marlin"]().clone()
            route_outputs = {name: function().clone() for name, function in functions.items()}
            torch.cuda.synchronize(device)
        baseline_ms = timing["marlin"].median_ms
        baseline_round_medians = timing["marlin"].round_medians_ms
        for route, stats in timing.items():
            errors = _error_metrics(route_outputs[route], reference)
            rows.append(
                {
                    "model": case.model,
                    "roles": list(case.roles),
                    "projection_count": case.count,
                    "dtype": _dtype_name(dtype),
                    "m": size_m,
                    "k": case.size_k,
                    "n": case.size_n,
                    "route": route,
                    **asdict(stats),
                    "speedup_vs_marlin": baseline_ms / stats.median_ms,
                    "paired_round_speedups_vs_marlin": [
                        baseline_round / route_round
                        for baseline_round, route_round in zip(baseline_round_medians, stats.round_medians_ms)
                    ],
                    "max_abs_vs_marlin": errors["max_abs"],
                    "mae_vs_marlin": errors["mae"],
                    "rmse_vs_marlin": errors["rmse"],
                    "relative_l2_vs_marlin": errors["relative_l2"],
                    "finite": errors["finite"],
                }
            )
    return rows


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Marlin packed-prefill benchmarking requires CUDA")
    torch.cuda.set_device(device)

    config_paths = tuple(args.model_configs)
    cases = [case for path in config_paths for case in _shape_cases_from_config(path)]
    if args.shape_pattern:
        pattern = re.compile(args.shape_pattern)
        cases = [
            case
            for case in cases
            if pattern.search(f"{case.model}:{case.size_k}x{case.size_n}:{','.join(case.roles)}")
        ]
    if not cases:
        raise ValueError("No projection shapes matched the requested model configs and filter")

    payload: dict[str, object] = {
        "hardware": _hardware_metadata(device),
        "gpu_idle_preflight": _GPU_IDLE_PREFLIGHT.as_dict() if _GPU_IDLE_PREFLIGHT is not None else None,
        "model_configs": [str(path) for path in config_paths],
        "shape_inventory": [asdict(case) for case in cases],
        "m_values": list(args.m_values),
        "configs": list(args.configs),
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "results": [],
        "dense_reference": [],
    }
    results: list[dict[str, object]] = []
    dense_results: list[dict[str, object]] = []
    last_live_update = time.monotonic()
    for dtype_index, dtype in enumerate(_resolve_dtypes(args.dtype)):
        for case_index, case in enumerate(cases):
            case_seed = args.seed + dtype_index * 100_000 + case_index * 1000
            case_rows = _scan_shape(
                case=case,
                dtype=dtype,
                m_values=args.m_values,
                configs=args.configs,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                rounds=args.rounds,
                seed=case_seed,
            )
            results.extend(case_rows)
            if args.dense_reference_m:
                dense_results.append(
                    {
                        "model": case.model,
                        "roles": list(case.roles),
                        "dtype": _dtype_name(dtype),
                        "k": case.size_k,
                        "n": case.size_n,
                        **_dense_reference_check(
                            device=device,
                            dtype=dtype,
                            size_m=args.dense_reference_m,
                            size_k=case.size_k,
                            size_n=case.size_n,
                            seed=case_seed + 500,
                            configs=args.configs,
                        ),
                    }
                )
            print(_format_live_table(case_rows), flush=True)
            now = time.monotonic()
            if now - last_live_update >= 60:
                print("\nComplete accumulated result table:\n", flush=True)
                print(_format_live_table(results), flush=True)
                last_live_update = now

    payload["results"] = results
    payload["dense_reference"] = dense_results
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
