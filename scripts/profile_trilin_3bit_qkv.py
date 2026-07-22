# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.triton_utils.three_bit import unpack_3bit
from gptqmodel.utils.trilin import prewarm_trilin_extension, trilin_matmul


K = 4096
Q_SIZE = 4096
GROUP_SIZE = 128
ZERO = 4


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = actual_fp32 - reference_fp32
    rmse = difference.square().mean().sqrt()
    reference_rms = reference_fp32.square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_rmse": (rmse / reference_rms).item(),
        "cosine": F.cosine_similarity(actual_fp32.flatten(), reference_fp32.flatten(), dim=0).item(),
    }


def _make_projection_references(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = unpack_3bit(qweight, axis=0, count=K)
    dense = (codes.float() - ZERO) * scales.repeat_interleave(GROUP_SIZE, dim=0).float()
    torch_output = torch.matmul(x, dense.to(x.dtype)).to(x.dtype)
    fp32_output = torch.matmul(x.float(), dense).to(x.dtype)
    return torch_output, fp32_output


def _measure(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    clock_warmup: Callable[[], None],
) -> dict[str, float]:
    clock_warmup()
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        function()
        ends[index].record()
    torch.cuda.synchronize()
    samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iterations)]
    ordered = sorted(samples_us)
    p95_index = min(iterations - 1, math.ceil(iterations * 0.95) - 1)
    return {
        "mean_us": statistics.mean(samples_us),
        "p50_us": statistics.median(samples_us),
        "p95_us": ordered[p95_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _print_timing(results: dict[str, dict[str, float]]) -> None:
    print("+-----------+----------+----------+----------+----------+----------+")
    print("| method    | mean us  | p50 us   | p95 us   | min us   | max us   |")
    print("+-----------+----------+----------+----------+----------+----------+")
    for method, result in results.items():
        print(
            f"| {method:<9} | {result['mean_us']:>8.3f} | {result['p50_us']:>8.3f} | "
            f"{result['p95_us']:>8.3f} | {result['min_us']:>8.3f} | {result['max_us']:>8.3f} |"
        )
    print("+-----------+----------+----------+----------+----------+----------+")


def _trace(
    methods: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    *,
    method_names: list[str],
    iterations: int,
) -> None:
    for function in methods.values():
        for _ in range(100):
            function()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for method_name in method_names:
        torch.cuda.nvtx.range_push(f"trilin_qkv_{method_name}")
        for _ in range(iterations):
            methods[method_name]()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile exact-shape Trilin 3-bit Q/K/V decode fusion.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], required=True)
    parser.add_argument("--kv-size", type=int, choices=[1024, 4096], required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clock-warmup", type=int, default=64)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--sanitizer-smoke", action="store_true")
    parser.add_argument("--trace", choices=["none", "control", "candidate", "both"], default="none")
    parser.add_argument("--trace-iterations", type=int, default=500)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = _dtype(args.dtype)
    prewarm_trilin_extension()
    torch.manual_seed(509)

    def make_projection(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        qweight = torch.randint(
            -(1 << 31),
            (1 << 31) - 1,
            (K // 32 * 3, n),
            device=device,
            dtype=torch.int32,
        )
        scales = torch.rand((K // GROUP_SIZE, n), device=device).mul_(0.20).add_(0.03125).half()
        return qweight, scales

    q_qweight, q_scales = make_projection(Q_SIZE)
    k_qweight, k_scales = make_projection(args.kv_size)
    v_qweight, v_scales = make_projection(args.kv_size)
    x = torch.randn((1, K), device=device, dtype=dtype)

    clock_lhs = torch.randn((4096, 4096), device=device, dtype=dtype)
    clock_rhs = torch.randn((4096, 4096), device=device, dtype=dtype)
    clock_output = torch.empty_like(clock_lhs)

    def warmup_clock() -> None:
        for _ in range(args.clock_warmup):
            torch.mm(clock_lhs, clock_rhs, out=clock_output)
        torch.cuda.synchronize(device)

    def control() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            trilin_matmul(x, q_qweight, q_scales),
            trilin_matmul(x, k_qweight, k_scales),
            trilin_matmul(x, v_qweight, v_scales),
        )

    methods: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {"control": control}
    if hasattr(torch.ops.gptqmodel_trilin, "qkv"):
        qkv_op = torch.ops.gptqmodel_trilin.qkv

        def candidate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            combined = qkv_op(
                x,
                q_qweight,
                q_scales,
                k_qweight,
                k_scales,
                v_qweight,
                v_scales,
            )
            return torch.split(combined, (Q_SIZE, args.kv_size, args.kv_size), dim=-1)

        methods["candidate"] = candidate

    if args.sanitizer_smoke:
        if "candidate" not in methods:
            raise RuntimeError("Trilin fused QKV operator is unavailable")
        methods["candidate"]()
        torch.cuda.synchronize(device)
        print(f"sanitizer smoke passed: dtype={args.dtype}, shape=1x{K}, q=4096, kv={args.kv_size}")
        return

    quality: dict[str, dict[str, dict[str, float]]] = {}
    if not args.skip_quality and "candidate" in methods:
        candidate_outputs = methods["candidate"]()
        control_outputs = control()
        projection_tensors = (
            ("q", q_qweight, q_scales),
            ("k", k_qweight, k_scales),
            ("v", v_qweight, v_scales),
        )
        for index, (name, qweight, scales) in enumerate(projection_tensors):
            torch_reference, fp32_reference = _make_projection_references(x, qweight, scales)
            quality[name] = {
                "candidate_vs_control": _metrics(candidate_outputs[index], control_outputs[index]),
                "candidate_vs_torch_3bit": _metrics(candidate_outputs[index], torch_reference),
                "candidate_vs_fp32_oracle": _metrics(candidate_outputs[index], fp32_reference),
                "torch_3bit_vs_fp32_oracle": _metrics(torch_reference, fp32_reference),
            }

    requested_trace = [args.trace] if args.trace not in {"none", "both"} else list(methods)
    missing_methods = set(requested_trace) - methods.keys()
    if missing_methods:
        raise RuntimeError(f"requested unavailable methods: {sorted(missing_methods)}")

    timing_samples: dict[str, list[dict[str, float]]] = {name: [] for name in methods}
    for round_index in range(args.rounds):
        order = list(methods) if round_index % 2 == 0 else list(reversed(methods))
        for method_name in order:
            timing_samples[method_name].append(
                _measure(
                    methods[method_name],
                    warmup=args.warmup,
                    iterations=args.iterations,
                    clock_warmup=warmup_clock,
                )
            )
    timing = {
        method_name: {
            metric: statistics.mean(sample[metric] for sample in samples)
            for metric in ("mean_us", "p50_us", "p95_us", "min_us", "max_us")
        }
        for method_name, samples in timing_samples.items()
    }
    metadata = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "dtype": args.dtype,
        "m": 1,
        "k": K,
        "q_size": Q_SIZE,
        "kv_size": args.kv_size,
        "bits": 3,
        "group_size": GROUP_SIZE,
        "sym": True,
        "desc_act": False,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
    }
    result: dict[str, object] = {"metadata": metadata, "quality": quality, "timing": timing}
    if "candidate" in timing:
        result["speedup"] = {
            "mean": timing["control"]["mean_us"] / timing["candidate"]["mean_us"],
            "p50": timing["control"]["p50_us"] / timing["candidate"]["p50_us"],
            "p95": timing["control"]["p95_us"] / timing["candidate"]["p95_us"],
        }
    print(json.dumps(result, sort_keys=True))
    _print_timing(timing)
    if "speedup" in result:
        speedup = result["speedup"]
        assert isinstance(speedup, dict)
        print(
            "control/candidate speedup: "
            f"mean={speedup['mean']:.4f}x, p50={speedup['p50']:.4f}x, p95={speedup['p95']:.4f}x"
        )

    if args.trace != "none":
        _trace(methods, method_names=requested_trace, iterations=args.trace_iterations)


if __name__ == "__main__":
    main()
