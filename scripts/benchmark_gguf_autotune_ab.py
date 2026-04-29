#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


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


def _build_module(
    case: BenchCase,
    *,
    dtype: torch.dtype,
    device: str,
    autotune: bool,
    force_candidate: bool,
) -> GGUFTorchLinear:
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
    if force_candidate:
        module.gguf_fused_cuda_max_rows = max(case.rows, 1)
        module.gguf_fused_cuda_min_matrix_elements = 0
        module.gguf_fused_cpu_max_rows = max(case.rows, 1)
        module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = autotune
    module.clear_autotune()
    module = module.to(device).eval()
    return module


def _bench_once(fn, *, sync_cuda: bool) -> float:
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if sync_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _plan_label(module: GGUFTorchLinear, x: torch.Tensor) -> str:
    if not module.autotune_enabled:
        return "fused" if module._is_fused_k_forward_candidate(x) else "none"
    decision = module.get_autotune_result()
    if decision is None:
        return "none"
    return "fused" if decision else "dense"


def _run_case(
    case: BenchCase,
    *,
    dtype: torch.dtype,
    device: str,
    trials: int,
    warmup: int,
    force_candidate: bool,
) -> None:
    sync_cuda = device == "cuda"
    x = torch.randn(case.rows, case.in_features, device=device, dtype=dtype)

    static_module = _build_module(case, dtype=dtype, device=device, autotune=False, force_candidate=force_candidate)
    autotune_module = _build_module(case, dtype=dtype, device=device, autotune=True, force_candidate=force_candidate)

    for _ in range(warmup):
        static_module(x)
        autotune_module(x)

    # Untimed warmup to settle dispatch decisions before measurement.
    static_module(x)
    autotune_module(x)

    static_plan = _plan_label(static_module, x)
    autotune_plan = _plan_label(autotune_module, x)

    static_trials: list[float] = []
    autotune_trials: list[float] = []
    for _ in range(trials):
        static_trials.append(_bench_once(lambda: static_module(x), sync_cuda=sync_cuda))
        autotune_trials.append(_bench_once(lambda: autotune_module(x), sync_cuda=sync_cuda))

    static_out = static_module(x)
    autotune_out = autotune_module(x)
    diff = (static_out.to(torch.float32) - autotune_out.to(torch.float32)).abs()
    mae = diff.mean().item()
    max_abs = diff.max().item()

    trial_rows: list[list[str]] = []
    for idx, (static_ms, autotune_ms) in enumerate(zip(static_trials, autotune_trials), start=1):
        speedup = static_ms / autotune_ms if autotune_ms > 0 else float("inf")
        delta_pct = ((static_ms - autotune_ms) / static_ms * 100.0) if static_ms > 0 else 0.0
        trial_rows.append(
            [
                str(idx),
                f"{static_ms:.3f}",
                f"{autotune_ms:.3f}",
                f"{speedup:.2f}x",
                f"{delta_pct:.1f}%",
            ]
        )

    static_mean = sum(static_trials) / len(static_trials)
    autotune_mean = sum(autotune_trials) / len(autotune_trials)
    summary_rows = [
        [
            "static",
            static_plan,
            f"{static_mean:.3f}",
            f"{min(static_trials):.3f}",
            f"{max(static_trials):.3f}",
            "-",
        ],
        [
            "autotune",
            autotune_plan,
            f"{autotune_mean:.3f}",
            f"{min(autotune_trials):.3f}",
            f"{max(autotune_trials):.3f}",
            f"{static_mean / autotune_mean:.2f}x",
        ],
    ]

    print()
    print(
        f"CASE {case.name} device={device} bits={case.bits} rows={case.rows} "
        f"shape={case.out_features}x{case.in_features} dtype={str(dtype).removeprefix('torch.')}"
    )
    print(_ascii_table(["trial", "static_ms", "autotune_ms", "speedup", "delta_pct"], trial_rows))
    print(_ascii_table(["mode", "plan", "mean_ms", "min_ms", "max_ms", "speedup_vs_static"], summary_rows))
    print(f"correctness: mae={mae:.6f} max_abs={max_abs:.6f}")


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
    parser = argparse.ArgumentParser(
        description="A/B benchmark GGUF static dispatch vs autotuned dispatch, excluding autotune setup cost."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=[],
        help="Benchmark case as name:bits:in_features:out_features:rows[:group_size].",
    )
    parser.add_argument("--trials", type=int, default=5, help="Measured trials per case.")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup forwards before the measured warmup call.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16", help="Benchmark dtype.")
    parser.add_argument(
        "--force-candidate",
        action="store_true",
        help="Override thresholds so every case is eligible for fused-vs-dense dispatch tuning.",
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
        f"trials={args.trials} warmup={args.warmup} force_candidate={args.force_candidate}"
    )
    for device in devices:
        for case in cases:
            _run_case(
                case,
                dtype=dtype,
                device=device,
                trials=args.trials,
                warmup=args.warmup,
                force_candidate=args.force_candidate,
            )


if __name__ == "__main__":
    main()
