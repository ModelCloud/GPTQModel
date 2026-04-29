#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel, triton_available as gguf_triton_triton_available


@dataclass(frozen=True)
class BenchCase:
    name: str
    bits: str
    in_features: int
    out_features: int
    rows: int
    group_size: int = -1


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep, fmt(headers), sep]
    for row in rows:
        out.append(fmt(row))
    out.append(sep)
    return "\n".join(out)


def _build_module(case: BenchCase, *, dtype: torch.dtype) -> GGUFTorchLinear:
    linear = nn.Linear(case.in_features, case.out_features, bias=False, dtype=dtype).cpu().eval()
    torch.manual_seed(0)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)

    module = GGUFTorchLinear(
        bits=case.bits,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=False,
        register_buffers=False,
    )
    module.pack_original(linear, scales=torch.empty(0), zeros=torch.empty(0), g_idx=None)
    module.post_init()
    module.gguf_fused_cuda_max_rows = max(case.rows, 1)
    module.gguf_fused_cuda_min_matrix_elements = 0
    module.gguf_fused_cpu_max_rows = max(case.rows, 1)
    module.gguf_fused_cpu_min_matrix_elements = 0
    return module


def _bench_once(fn, *, sync_cuda: bool) -> float:
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if sync_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _run_case(
    case: BenchCase,
    *,
    dtype: torch.dtype,
    device: str,
    trials: int,
    warmup: int,
    include_triton: bool,
) -> None:
    module = _build_module(case, dtype=dtype)
    module = module.to(device).eval()
    triton_module = None
    x = torch.randn(case.rows, case.in_features, device=device, dtype=dtype)
    sync_cuda = device == "cuda"
    can_bench_triton = (
        include_triton
        and device == "cuda"
        and dtype == torch.float16
        and module.gguf_tensor_qtype in {"Q4_K", "Q5_K", "Q6_K"}
        and gguf_triton_triton_available()
    )
    if can_bench_triton:
        triton_module = GGUFTritonKernel(
            bits=case.bits,
            group_size=case.group_size,
            sym=True,
            desc_act=False,
            in_features=case.in_features,
            out_features=case.out_features,
            bias=False,
            register_buffers=True,
        ).to(device).eval()
        triton_module.load_state_dict(module.state_dict(), strict=True)

    for _ in range(warmup):
        module._forward_dequant_matmul(x)
        module._forward_fused_k(x)
        if can_bench_triton:
            triton_module(x)

    baseline_trials: list[float] = []
    fused_trials: list[float] = []
    triton_trials: list[float] = []
    for _ in range(trials):
        baseline_trials.append(_bench_once(lambda: module._forward_dequant_matmul(x), sync_cuda=sync_cuda))
        fused_trials.append(_bench_once(lambda: module._forward_fused_k(x), sync_cuda=sync_cuda))
        if can_bench_triton:
            triton_trials.append(_bench_once(lambda: triton_module(x), sync_cuda=sync_cuda))

    baseline_out = module._forward_dequant_matmul(x)
    fused_out = module._forward_fused_k(x)
    if can_bench_triton:
        triton_out = triton_module(x)
        triton_diff = (baseline_out.to(torch.float32) - triton_out.to(torch.float32)).abs()
        triton_mae = triton_diff.mean().item()
        triton_max_abs = triton_diff.max().item()
    diff = (baseline_out.to(torch.float32) - fused_out.to(torch.float32)).abs()
    mae = diff.mean().item()
    max_abs = diff.max().item()

    if can_bench_triton:
        trial_rows = []
        for idx, (baseline_ms, fused_ms, triton_ms) in enumerate(zip(baseline_trials, fused_trials, triton_trials), start=1):
            trial_rows.append(
                [
                    str(idx),
                    f"{baseline_ms:.3f}",
                    f"{fused_ms:.3f}",
                    f"{triton_ms:.3f}",
                    f"{baseline_ms / fused_ms:.2f}x" if fused_ms > 0 else "inf",
                    f"{baseline_ms / triton_ms:.2f}x" if triton_ms > 0 else "inf",
                ]
            )
    else:
        trial_rows = []
        for idx, (baseline_ms, fused_ms) in enumerate(zip(baseline_trials, fused_trials), start=1):
            speedup = baseline_ms / fused_ms if fused_ms > 0 else float("inf")
            delta_pct = ((baseline_ms - fused_ms) / baseline_ms * 100.0) if baseline_ms > 0 else 0.0
            trial_rows.append(
                [
                    str(idx),
                    f"{baseline_ms:.3f}",
                    f"{fused_ms:.3f}",
                    f"{speedup:.2f}x",
                    f"{delta_pct:.1f}%",
                ]
            )

    summary_rows = [
        [
            "baseline",
            f"{sum(baseline_trials) / len(baseline_trials):.3f}",
            f"{min(baseline_trials):.3f}",
            f"{max(baseline_trials):.3f}",
            "-",
        ],
        [
            "fused",
            f"{sum(fused_trials) / len(fused_trials):.3f}",
            f"{min(fused_trials):.3f}",
            f"{max(fused_trials):.3f}",
            f"{(sum(baseline_trials) / len(baseline_trials)) / (sum(fused_trials) / len(fused_trials)):.2f}x",
        ],
    ]
    if can_bench_triton:
        summary_rows.append(
            [
                "triton",
                f"{sum(triton_trials) / len(triton_trials):.3f}",
                f"{min(triton_trials):.3f}",
                f"{max(triton_trials):.3f}",
                f"{(sum(baseline_trials) / len(baseline_trials)) / (sum(triton_trials) / len(triton_trials)):.2f}x",
            ]
        )

    print()
    print(
        f"CASE {case.name} device={device} bits={case.bits} rows={case.rows} "
        f"shape={case.out_features}x{case.in_features} dtype={str(dtype).removeprefix('torch.')}"
    )
    if can_bench_triton:
        print(_ascii_table(["trial", "baseline_ms", "fused_ms", "triton_ms", "torch_speedup", "triton_speedup"], trial_rows))
    else:
        print(_ascii_table(["trial", "baseline_ms", "fused_ms", "speedup", "delta_pct"], trial_rows))
    print(_ascii_table(["path", "mean_ms", "min_ms", "max_ms", "speedup_vs_baseline"], summary_rows))
    print(f"correctness: mae={mae:.6f} max_abs={max_abs:.6f}")
    if can_bench_triton:
        print(f"triton_correctness: mae={triton_mae:.6f} max_abs={triton_max_abs:.6f}")


def _parse_case(spec: str) -> BenchCase:
    parts = spec.split(":")
    if len(parts) not in (5, 6):
        raise ValueError(
            f"Invalid case `{spec}`. Expected name:bits:in_features:out_features:rows[:group_size]."
        )
    name, bits, in_features, out_features, rows, *rest = parts
    return BenchCase(
        name=name,
        bits=bits,
        in_features=int(in_features),
        out_features=int(out_features),
        rows=int(rows),
        group_size=int(rest[0]) if rest else -1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B benchmark GGUF K-type dense vs fused forward.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=[],
        help="Benchmark case as name:bits:in_features:out_features:rows[:group_size].",
    )
    parser.add_argument("--trials", type=int, default=5, help="Measured trials per case.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per path.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16", help="Benchmark dtype.")
    parser.add_argument(
        "--include-triton",
        action="store_true",
        help="Also benchmark the experimental CUDA Triton fused GGUF path when available.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "both"),
        default="auto",
        help="Benchmark device selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if args.device == "auto":
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    elif args.device == "both":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    else:
        devices = [args.device]

    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    if "cpu" in devices and dtype == torch.float16:
        raise RuntimeError("CPU benchmarks should use --dtype bf16 or --dtype fp32.")

    cases = (
        [_parse_case(spec) for spec in args.cases]
        if args.cases
        else [
            BenchCase("attn_q4_k_m_r1", "q4_k_m", 2048, 2048, 1),
            BenchCase("attn_q5_k_m_r1", "q5_k_m", 2048, 2048, 1),
            BenchCase("attn_q6_k_r1", "q6_k", 2048, 2048, 1),
            BenchCase("mlp_q4_k_m_r8", "q4_k_m", 2048, 8192, 8),
            BenchCase("mlp_q5_k_m_r8", "q5_k_m", 2048, 8192, 8),
            BenchCase("mlp_q6_k_r8", "q6_k", 2048, 8192, 8),
        ]
    )

    print(
        f"devices={','.join(devices)} dtype={str(dtype).removeprefix('torch.')} "
        f"trials={args.trials} warmup={args.warmup}"
    )
    for device in devices:
        for case in cases:
            _run_case(
                case,
                dtype=dtype,
                device=device,
                trials=args.trials,
                warmup=args.warmup,
                include_triton=args.include_triton,
            )


if __name__ == "__main__":
    main()
