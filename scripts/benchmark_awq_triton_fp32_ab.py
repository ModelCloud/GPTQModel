#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark for the AWQ Triton scheduler (both sides use FP32 accum)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import torch
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization.awq.modules.triton.gemm import awq_dequantize_triton, awq_gemm_triton
from gptqmodel.quantization.awq.modules.triton.scheduler import (
    AwqTritonPlan,
    candidate_plans,
    clear_awq_triton_rules,
    legacy_plan,
    register_awq_triton_rule,
    select_awq_triton_plan,
)
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm


@dataclass(frozen=True)
class BenchCase:
    M: int
    K: int
    N: int
    group_size: int = 128


M_VALUES = (1, 2, 8, 16, 31, 32, 33, 64, 127, 128, 129, 256, 512, 2048)
KN_VALUES = ((2048, 512), (2048, 2048), (2048, 8192), (8192, 2048), (4096, 4096))
# Tail cases preserve legal K/group alignment; K=2080 gives an uneven split
# stride for BK=64 and N=520 exercises the packed output tail.
TAIL_VALUES = ((2080, 520), (4096, 4104))
DEFAULT_CASES = tuple(BenchCase(M, K, N) for M in M_VALUES for K, N in KN_VALUES) + tuple(
    BenchCase(M, K, N, 32) for M in M_VALUES for K, N in TAIL_VALUES
)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    factor = 32 // bits
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // factor), dtype=torch.int32)
    for col in range(packed.shape[1]):
        for i, source in enumerate(order):
            packed[:, col] |= unpacked[:, col * factor + source].to(torch.int32) << (i * bits)
    return packed


def _make_quant_buffers(case: BenchCase) -> dict[str, torch.Tensor]:
    groups = case.K // case.group_size
    weight = torch.randint(0, 16, (case.K, case.N), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, case.N), dtype=torch.int32)
    return {
        "qweight": _pack_awq_tensor(weight, 4),
        "qzeros": _pack_awq_tensor(zeros, 4),
        "scales": torch.rand(groups, case.N, dtype=torch.float16) * 0.5 + 0.75,
        "bias": torch.randn(case.N, dtype=torch.float16),
    }


def _kernel(x: torch.Tensor, buffers: dict[str, torch.Tensor], plan: AwqTritonPlan):
    if plan.path == "dense":
        weight = awq_dequantize_triton(buffers["qweight"], buffers["scales"], buffers["qzeros"])
        result = torch.matmul(x, weight.to(x.dtype))
    else:
        result = awq_gemm_triton(
            x, buffers["qweight"], buffers["scales"], buffers["qzeros"],
            fp32_accum=True, output_dtype=x.dtype, **plan.as_kwargs(),
        )
    return result


def _forward(entry: torch.Tensor, buffers: dict[str, torch.Tensor], plan: AwqTritonPlan):
    x = entry.to(torch.float16).contiguous().reshape(-1, entry.shape[-1])
    return (_kernel(x, buffers, plan) + buffers["bias"]).reshape(
        entry.shape[:-1] + (buffers["bias"].shape[0],)
    ).to(entry.dtype)


def _scheduled_forward(
    entry: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    case: BenchCase,
    explicit: AwqTritonPlan | None = None,
):
    plan = select_awq_triton_plan(
        M=case.M,
        N=case.N,
        K=case.K,
        group_size=case.group_size,
        device=entry.device,
        input_dtype=entry.dtype,
        compute_dtype=torch.float16,
        output_dtype=entry.dtype,
        fp32_accum=True,
        explicit=explicit,
    )
    return _forward(entry, buffers, plan)


def _make_module(
    case: BenchCase, buffers: dict[str, torch.Tensor], device: torch.device,
) -> AwqGEMMTritonLinear:
    module = AwqGEMMTritonLinear(
        bits=4, group_size=case.group_size, sym=False, desc_act=False,
        in_features=case.K, out_features=case.N, bias=True,
        register_buffers=True, schedule_mode="auto",
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(buffers["qweight"])
        module.qzeros.copy_(buffers["qzeros"])
        module.scales.copy_(buffers["scales"])
        module.bias.copy_(buffers["bias"])
    module.eval()
    module.post_init()
    return module


def _module_forward(
    module: AwqGEMMTritonLinear, entry: torch.Tensor, mode: str,
    schedule: AwqTritonPlan | None = None,
):
    module.awq_triton_schedule_mode = mode
    module.awq_triton_schedule = schedule
    return module(entry)


def _warmup(fn, device: torch.device, count: int) -> tuple[float, float]:
    with torch.inference_mode():
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        first = (time.perf_counter() - start) * 1e3
        start = time.perf_counter()
        for _ in range(max(0, count - 1)):
            fn()
    torch.cuda.synchronize(device)
    return first, (time.perf_counter() - start) * 1e3


def _alternate(old_fn, new_fn, device: torch.device, iters: int):
    old, new = [], []
    with torch.inference_mode():
        for i in range(iters):
            order = ((old_fn, old), (new_fn, new)) if i % 2 == 0 else ((new_fn, new), (old_fn, old))
            for fn, samples in order:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(old), statistics.median(new), old, new


def _one_ms(fn, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _device_metadata(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "driver_version": getattr(torch._C, "_cuda_getDriverVersion", lambda: None)(),
        "cuda_version": torch.version.cuda,
    }


def run(device: torch.device, warmup: int = 5, iters: int = 20, limit: int | None = None,
        explicit_plan: AwqTritonPlan | None = None, sweep: bool = False,
        sweep_iters: int = 3, shard_index: int = 0, num_shards: int = 1) -> dict[str, Any]:
    clear_awq_triton_rules()
    torch.manual_seed(0)
    indexed_cases = list(enumerate(DEFAULT_CASES))
    if limit:
        indexed_cases = indexed_cases[:limit]
    indexed_cases = indexed_cases[shard_index::num_shards]
    results = []
    for index, case in indexed_cases:
        torch.manual_seed(1000 + index)
        buffers = {key: value.to(device) for key, value in _make_quant_buffers(case).items()}
        # Deliberately non-contiguous rank-3 input for complete-forward timing.
        x = torch.randn((1, case.K, case.M), device=device, dtype=torch.float16).transpose(1, 2)
        old_plan = legacy_plan(case.M, case.K, case.N, case.group_size)
        sweep_log = []
        candidate_rejections = {}
        new_plan = select_awq_triton_plan(
            M=case.M, N=case.N, K=case.K, group_size=case.group_size,
            device=device, input_dtype=x.dtype, compute_dtype=torch.float16,
            output_dtype=x.dtype, fp32_accum=True,
            explicit=explicit_plan,
        )
        old_fn = lambda: _forward(x, buffers, old_plan)
        new_fn = lambda: _scheduled_forward(x, buffers, case, explicit_plan)
        x2d = x.contiguous().reshape(case.M, case.K)
        old_kernel = lambda: _kernel(x2d, buffers, old_plan)
        new_kernel = lambda: _kernel(x2d, buffers, new_plan)
        # Record the old path's process-local first launch before a sweep can
        # compile shared specializations.
        old_first, old_warmup = _warmup(old_fn, device, warmup)
        selected_sweep_first = None
        if sweep and explicit_plan is None:
            best_ms = float("inf")
            candidates, candidate_rejections = candidate_plans(case.M, case.N, case.K, case.group_size)
            for candidate in candidates:
                candidate_fn = lambda candidate=candidate: _forward(x, buffers, candidate)
                try:
                    first_ms, _ = _warmup(candidate_fn, device, 1)
                    samples = [_one_ms(candidate_fn, device) for _ in range(max(1, sweep_iters))]
                    candidate_ms = statistics.median(samples)
                    sweep_log.append({
                        "plan": candidate.__dict__,
                        "status": "ok",
                        "first_launch_ms": first_ms,
                        "ms": candidate_ms,
                        "samples_ms": samples,
                    })
                    if candidate_ms < best_ms:
                        best_ms, new_plan = candidate_ms, candidate
                        selected_sweep_first = first_ms
                except Exception as exc:  # compile/resource failures are data, not fatal
                    sweep_log.append({"plan": candidate.__dict__, "status": "rejected", "reason": repr(exc)})
            register_awq_triton_rule(
                device, new_plan, M=case.M, N=case.N, K=case.K,
                group_size=case.group_size, input_dtype=x.dtype,
                compute_dtype=torch.float16, output_dtype=x.dtype,
                fp32_accum=True,
            )
            new_fn = lambda: _scheduled_forward(x, buffers, case)
            new_kernel = lambda: _kernel(x2d, buffers, new_plan)
        new_first_call, new_warmup = _warmup(new_fn, device, warmup)
        new_first = selected_sweep_first if selected_sweep_first is not None else new_first_call
        torch.cuda.reset_peak_memory_stats(device)
        old_entry_ms, new_entry_ms, old_entry_samples, new_entry_samples = _alternate(
            old_fn, new_fn, device, iters
        )
        peak_entry = torch.cuda.max_memory_allocated(device)
        module = _make_module(case, buffers, device)
        # Exercise the real module entry point while toggling only scheduling;
        # this includes reshape, bias, dtype conversion, and output restoration.
        # Explicit takes precedence over mode in the selector. Always clear it
        # on the legacy side, and carry --candidate through the real module on
        # the selected side so its samples match ``selected_plan``.
        old_module_fn = lambda: _module_forward(module, x, "legacy", None)
        new_module_fn = lambda: _module_forward(module, x, "auto", explicit_plan)
        old_module_first, old_module_warmup = _warmup(old_module_fn, device, warmup)
        new_module_first, new_module_warmup = _warmup(new_module_fn, device, warmup)
        torch.cuda.reset_peak_memory_stats(device)
        old_ms, new_ms, old_samples, new_samples = _alternate(
            old_module_fn, new_module_fn, device, iters
        )
        peak_forward = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        old_kernel_ms, new_kernel_ms, old_kernel_samples, new_kernel_samples = _alternate(
            old_kernel, new_kernel, device, iters
        )
        peak_kernel = torch.cuda.max_memory_allocated(device)
        with torch.inference_mode():
            reference = torch.matmul(x2d, dequantize_gemm(
                qweight=buffers["qweight"], qzeros=buffers["qzeros"], scales=buffers["scales"],
                bits=4, group_size=case.group_size,
            ).to(device=device, dtype=x.dtype)) + buffers["bias"]
            legacy_diff = (old_fn().reshape(case.M, case.N) - reference).abs()
            selected_diff = (new_fn().reshape(case.M, case.N) - reference).abs()
        results.append({
            "M": case.M, "K": case.K, "N": case.N, "group_size": case.group_size,
            "legacy_plan": old_plan.__dict__, "selected_plan": new_plan.__dict__,
            "sweep_candidates": sweep_log,
            "sweep_rejections": candidate_rejections,
            # The measured functions include dequantization/matmul, conversion,
            # bias, and layout costs represented by this complete forward.
            "legacy_kernel_ms": old_kernel_ms, "selected_kernel_ms": new_kernel_ms,
            "selected_over_legacy_kernel": new_kernel_ms / old_kernel_ms,
            "legacy_forward_ms": old_ms, "selected_forward_ms": new_ms,
            "legacy_process_first_launch_ms": old_first,
            "selected_process_first_launch_ms": new_first,
            "first_launch_scope": "process-local; shared specializations may already be compiled",
            "legacy_warmup_ms": old_warmup, "selected_warmup_ms": new_warmup,
            "legacy_module_process_first_launch_ms": old_module_first,
            "selected_module_process_first_launch_ms": new_module_first,
            "legacy_module_warmup_ms": old_module_warmup,
            "selected_module_warmup_ms": new_module_warmup,
            "legacy_samples_ms": old_samples, "selected_samples_ms": new_samples,
            "legacy_entry_ms": old_entry_ms, "selected_entry_ms": new_entry_ms,
            "legacy_entry_samples_ms": old_entry_samples,
            "selected_entry_samples_ms": new_entry_samples,
            "legacy_kernel_samples_ms": old_kernel_samples, "selected_kernel_samples_ms": new_kernel_samples,
            "selected_over_legacy": new_ms / old_ms,
            "review_regression_over_5pct": new_ms / old_ms > 1.05,
            "peak_forward_memory_bytes": peak_forward, "peak_entry_memory_bytes": peak_entry,
            "peak_kernel_memory_bytes": peak_kernel,
            "legacy_max_abs_error": legacy_diff.max().item(),
            "legacy_mean_abs_error": legacy_diff.mean().item(),
            "max_abs_error": selected_diff.max().item(),
            "mean_abs_error": selected_diff.mean().item(),
        })
    ratios = [row["selected_over_legacy"] for row in results]
    return {
        "device": torch.cuda.get_device_name(device), "cuda_device": str(device),
        "device_metadata": _device_metadata(device),
        "torch_version": torch.__version__, "triton_version": getattr(__import__("triton"), "__version__", None),
        "git_commit": _commit(), "fp32_accum": True, "warmup": warmup, "iters": iters,
        "offline_sweep": sweep, "sweep_iters": sweep_iters,
        "timing_statistic": "median",
        "shard_index": shard_index, "num_shards": num_shards,
        "cases": results,
        "geomean_selected_over_legacy": math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate", default=None,
                        help="explicit fused BM,BN,BK,split,warps,stages for A/B (e.g. 16,64,64,2,4,2)")
    parser.add_argument("--sweep", action="store_true",
                        help="offline-measure finite legal candidates and select the fastest per case")
    parser.add_argument("--sweep-iters", type=int, default=3,
                        help="warmed timing samples per offline candidate")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="zero-based benchmark shard (cases are round-robin partitioned)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None,
                        help="write complete raw measurements to this JSON file")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AWQ Triton scheduler benchmark")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"Triton is required: {exc}") from exc
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    explicit = None
    if args.candidate:
        values = [int(value) for value in args.candidate.split(",")]
        if len(values) != 6:
            raise ValueError("--candidate requires BM,BN,BK,split,warps,stages")
        explicit = AwqTritonPlan("fused", *values)
    result = run(
        torch.device(f"cuda:{args.device}"), args.warmup, args.iters,
        args.limit, explicit, args.sweep, args.sweep_iters,
        args.shard_index, args.num_shards,
    )
    table = [[r["M"], f'{r["K"]}->{r["N"]}', r["selected_plan"]["path"],
              f'{r["legacy_forward_ms"]:.3f}', f'{r["selected_forward_ms"]:.3f}',
              f'{r["selected_over_legacy"]:.3f}x', f'{r["max_abs_error"]:.6f}', r["peak_forward_memory_bytes"]]
             for r in result["cases"]]
    print(f'Device: {result["device"]} ({result["cuda_device"]})')
    print(tabulate(
        table,
        headers=["M", "K->N", "path", "legacy ms", "selected ms", "selected/legacy", "max abs", "peak bytes"],
        tablefmt="grid",
    ))
    print(f'Geomean selected/legacy: {result["geomean_selected_over_legacy"]:.3f}x')
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
