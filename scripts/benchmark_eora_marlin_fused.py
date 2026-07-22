#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from tabulate import tabulate

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.utils.eora_marlin import apply_eora_marlin_fused_lora


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_FUSED_ENV = "GPTQMODEL_EORA_MARLIN_FUSED"
_COOPERATIVE_ENV = "GPTQMODEL_EORA_MARLIN_COOPERATIVE"
_CUDA_UP_ADD_ENV = "GPTQMODEL_EORA_MARLIN_CUDA_UP_ADD"


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    rows: int
    in_features: int
    out_features: int
    rank: int


DEFAULT_CASES = (
    BenchCase("decode_attn_r32", 1, 4096, 4096, 32),
    BenchCase("decode_attn_r64", 1, 4096, 4096, 64),
    BenchCase("decode_attn_r128", 1, 4096, 4096, 128),
    BenchCase("decode_attn_r256", 1, 4096, 4096, 256),
    BenchCase("decode_mlp_up_r128", 1, 4096, 11008, 128),
    BenchCase("decode_mlp_down_r128", 1, 11008, 4096, 128),
    BenchCase("batch8_attn_r128", 8, 4096, 4096, 128),
    BenchCase("batch12_attn_r128", 12, 4096, 4096, 128),
    BenchCase("prefill16_attn_r128", 16, 4096, 4096, 128),
)
VARIANTS = ("fallback", "addmm", "cooperative", "cuda_up_add")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and profile the fused EoRA/LoRA tail used by the Marlin backend."
    )
    parser.add_argument("--device", default="cuda:0", help="Logical torch device within CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--scope", choices=("tail", "marlin"), default="tail")
    parser.add_argument("--input-layout", choices=("2d", "3d"), default="2d")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated benchmark variants.")
    parser.add_argument("--case-pattern", default=None, help="Substring used to select benchmark cases.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--throughput-iters",
        type=int,
        default=None,
        help="Calls per aggregate throughput pass; defaults to --iters.",
    )
    parser.add_argument("--throughput-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Bracket measured calls with cudaProfilerStart/Stop for a bounded Nsight capture.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _select_cases(pattern: str | None) -> list[BenchCase]:
    cases = [case for case in DEFAULT_CASES if pattern is None or pattern in case.case_id]
    if not cases:
        raise ValueError(f"No EoRA benchmark case matched {pattern!r}.")
    return cases


def _select_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in variants if item not in VARIANTS]
    if invalid:
        raise ValueError(f"Unknown variants {invalid}; expected a subset of {VARIANTS}.")
    if not variants:
        raise ValueError("At least one benchmark variant is required.")
    return variants


def _set_variant(variant: str) -> None:
    os.environ[_FUSED_ENV] = "0" if variant == "fallback" else "1"
    os.environ[_COOPERATIVE_ENV] = "1" if variant == "cooperative" else "0"
    os.environ[_CUDA_UP_ADD_ENV] = "1" if variant == "cuda_up_add" else "0"


def _make_tensors(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    input_layout: str = "2d",
) -> tuple[torch.Tensor, torch.Tensor, Lora]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(
        (case.rows, case.in_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    base = torch.randn(
        (case.rows, case.out_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if input_layout == "3d":
        x = x.reshape(1, case.rows, case.in_features)
        base = base.reshape(1, case.rows, case.out_features)
    lora_a = torch.randn(
        (case.in_features, case.rank),
        dtype=dtype,
        device=device,
        generator=generator,
    ) * 0.002
    lora_b = torch.randn(
        (case.rank, case.out_features),
        dtype=dtype,
        device=device,
        generator=generator,
    ) * 0.002
    return x, base, Lora(rank=case.rank, lora_A=lora_a, lora_B=lora_b)


def _build_marlin_module(
    case: BenchCase,
    *,
    adapter: Lora,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> MarlinLinear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    module = MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=False,
        dtype=dtype,
        adapter=adapter,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                module.qweight.shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        )
        module.scales.copy_(
            torch.rand(module.scales.shape, dtype=dtype, device=device, generator=generator) * 0.01 + 0.01
        )
        module.qzeros.zero_()
        module.g_idx.zero_()
        # A path-less Lora uses the merged-weight loading lifecycle, so populate
        # the registered module buffers exactly as a checkpoint load would.
        module.lora_A.copy_(adapter.lora_A)
        module.lora_B.copy_(adapter.lora_B)
    module.eval()
    module.post_init()
    return module


@contextmanager
def _without_adapter(module: MarlinLinear) -> Iterator[None]:
    adapter = module.adapter
    module.adapter = None
    try:
        yield
    finally:
        module.adapter = adapter


def _make_call(
    *,
    scope: str,
    variant: str,
    x: torch.Tensor,
    base: torch.Tensor,
    adapter: Lora,
    module: MarlinLinear | None,
    cooperative_state,
) -> Callable[[], torch.Tensor]:
    _set_variant(variant)
    cooperative_buffer = None
    if variant == "cooperative" and scope == "tail":
        cooperative_buffer = torch.empty(
            (x.numel() // x.shape[-1], adapter.lora_A.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
    if scope == "marlin":
        assert module is not None
        module.eora_cuda_up_add = variant == "cuda_up_add"
        module.eora_cooperative_state = cooperative_state if variant == "cooperative" else None
        return lambda: module(x)
    if variant == "fallback":
        return lambda: adapter.apply(x=x, out=base)

    def fused_call() -> torch.Tensor:
        result = apply_eora_marlin_fused_lora(
            adapter,
            x=x,
            out=base,
            cooperative_buffer=cooperative_buffer,
        )
        if result is None:
            raise RuntimeError(f"EoRA fused path unexpectedly fell back for variant {variant!r}.")
        return result

    return fused_call


def _correctness_error(
    *,
    scope: str,
    call: Callable[[], torch.Tensor],
    x: torch.Tensor,
    base: torch.Tensor,
    adapter: Lora,
    module: MarlinLinear | None,
) -> tuple[float, float]:
    with torch.inference_mode():
        if scope == "marlin":
            assert module is not None
            with _without_adapter(module):
                reference_base = module(x)
        else:
            reference_base = base.clone()
        expected = reference_base.float() + (x.float() @ adapter.lora_A.float()) @ adapter.lora_B.float()
        actual_out = reference_base.clone() if scope == "tail" else None
        if actual_out is not None:
            variant = "cuda_up_add" if os.environ[_CUDA_UP_ADD_ENV] == "1" else "addmm"
            if os.environ[_FUSED_ENV] == "0":
                actual = adapter.apply(x=x, out=actual_out)
            else:
                actual = apply_eora_marlin_fused_lora(
                    adapter,
                    x=x,
                    out=actual_out,
                    cooperative_buffer=(
                        torch.empty(
                            (x.numel() // x.shape[-1], adapter.lora_A.shape[1]),
                            dtype=torch.float32,
                            device=x.device,
                        )
                        if os.environ[_COOPERATIVE_ENV] == "1"
                        else None
                    ),
                )
                if actual is None:
                    raise RuntimeError(f"EoRA fused path unexpectedly fell back for {variant!r}.")
        else:
            actual = call()
        expected_update = expected - reference_base.float()
        actual_update = actual.float() - reference_base.float()
        if expected_update.abs().max().item() > 0.0 and actual_update.abs().max().item() == 0.0:
            raise RuntimeError("Benchmark adapter path produced a zero LoRA update for nonzero adapter weights.")
        error = (actual.float() - expected).abs()
        relative = error / expected.abs().clamp_min(1e-5)
        return error.max().item(), relative.max().item()


def _measure_peak_bytes(call: Callable[[], torch.Tensor], device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        output = call()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del output
    return max(0, peak - before)


def _measure_latency(
    call: Callable[[], torch.Tensor],
    *,
    case: BenchCase,
    variant: str,
    device: torch.device,
    warmup: int,
    iters: int,
    profile: bool,
) -> list[float]:
    if warmup < 1 or iters < 1:
        raise ValueError("Warmup and iteration counts must be positive.")
    with torch.inference_mode():
        for _ in range(warmup):
            call()
    torch.cuda.synchronize(device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stream = torch.cuda.current_stream(device)
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    with torch.inference_mode():
        for index, (start, end) in enumerate(zip(starts, ends)):
            torch.cuda.nvtx.range_push(f"eora::{case.case_id}::{variant}::{index}")
            start.record(stream)
            call()
            end.record(stream)
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]


def _measure_throughput(
    call: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    iters: int,
    repeats: int,
) -> list[float]:
    """Measure sustained eager throughput with one event pair around the call loop."""
    if iters < 1 or repeats < 1:
        raise ValueError("Throughput iteration and repeat counts must be positive.")
    samples_us = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        with torch.inference_mode():
            start.record(stream)
            for _ in range(iters):
                call()
            end.record(stream)
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3 / iters)
    return samples_us


def _environment(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "python_gil": os.environ.get("PYTHON_GIL"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices.")
    dtype = _resolve_dtype(args.dtype)
    cases = _select_cases(args.case_pattern)
    variants = _select_variants(args.variants)
    if args.profile and len(variants) != 1:
        raise ValueError("--profile requires exactly one selected variant.")

    results: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        x, base, adapter = _make_tensors(
            case,
            device=device,
            dtype=dtype,
            seed=args.seed + case_index,
            input_layout=args.input_layout,
        )
        base_seed = base.clone()
        module = None
        if args.scope == "marlin":
            module = _build_marlin_module(
                case,
                adapter=adapter,
                device=device,
                dtype=dtype,
                seed=args.seed + 1000 + case_index,
            )
        cooperative_state = module.eora_cooperative_state if module is not None else None
        for variant in variants:
            base.copy_(base_seed)
            call = _make_call(
                scope=args.scope,
                variant=variant,
                x=x,
                base=base,
                adapter=adapter,
                module=module,
                cooperative_state=cooperative_state,
            )
            max_abs, max_rel = _correctness_error(
                scope=args.scope,
                call=call,
                x=x,
                base=base,
                adapter=adapter,
                module=module,
            )
            peak_bytes = _measure_peak_bytes(call, device)
            samples_us = _measure_latency(
                call,
                case=case,
                variant=variant,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                profile=args.profile,
            )
            throughput_samples_us = _measure_throughput(
                call,
                device=device,
                iters=args.throughput_iters or args.iters,
                repeats=args.throughput_repeats,
            )
            aggregate_mean_us = statistics.median(throughput_samples_us)
            calls_per_second = 1e6 / aggregate_mean_us
            results.append(
                {
                    **asdict(case),
                    "scope": args.scope,
                    "variant": variant,
                    "dtype": args.dtype,
                    "input_layout": args.input_layout,
                    "mean_us": statistics.fmean(samples_us),
                    "p50_us": _percentile(samples_us, 0.50),
                    "p95_us": _percentile(samples_us, 0.95),
                    "min_us": min(samples_us),
                    "aggregate_mean_us": aggregate_mean_us,
                    "aggregate_samples_us": throughput_samples_us,
                    "calls_per_second": calls_per_second,
                    "rows_per_second": calls_per_second * case.rows,
                    "peak_bytes": peak_bytes,
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                }
            )

    table = [
        [
            row["case_id"],
            row["scope"],
            row["variant"],
            row["dtype"],
            row["input_layout"],
            row["rows"],
            row["in_features"],
            row["out_features"],
            row["rank"],
            f'{row["p50_us"]:.2f}',
            f'{row["p95_us"]:.2f}',
            f'{row["mean_us"]:.2f}',
            f'{row["aggregate_mean_us"]:.2f}',
            f'{row["calls_per_second"]:.0f}',
            f'{row["rows_per_second"]:.0f}',
            f'{row["peak_bytes"] / 1024:.2f}',
            f'{row["max_abs_error"]:.3e}',
        ]
        for row in results
    ]
    print(
        tabulate(
            table,
            headers=(
                "case", "scope", "variant", "dtype", "layout", "m", "k", "n", "r",
                "p50_us", "p95_us", "mean_us", "stream_us", "call/s", "row/s", "peak_KiB", "max_abs",
            ),
            tablefmt="simple",
        )
    )
    environment = _environment(device)
    print("\nEnvironment")
    print(tabulate(environment.items(), headers=("field", "value"), tablefmt="simple"))

    if args.json_out is not None:
        payload = {
            "environment": environment,
            "scope": args.scope,
            "dtype": args.dtype,
            "input_layout": args.input_layout,
            "warmup": args.warmup,
            "iters": args.iters,
            "throughput_iters": args.throughput_iters or args.iters,
            "throughput_repeats": args.throughput_repeats,
            "profile": args.profile,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out.resolve()}")


if __name__ == "__main__":
    main()
