#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel, triton_available as gguf_triton_available


@dataclass(frozen=True)
class BenchCase:
    name: str
    bits: str
    in_features: int
    out_features: int
    rows: int
    bias: bool = False


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


def _parse_case(spec: str) -> BenchCase:
    parts = spec.split(":")
    if len(parts) not in (5, 6):
        raise ValueError(f"Invalid case `{spec}`. Expected name:bits:in_features:out_features:rows[:bias].")
    name, bits, in_features, out_features, rows, *rest = parts
    return BenchCase(
        name=name,
        bits=bits,
        in_features=int(in_features),
        out_features=int(out_features),
        rows=int(rows),
        bias=bool(int(rest[0])) if rest else False,
    )


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(fn: Callable[[], torch.Tensor], *, device: str, warmup: int, trials: int) -> tuple[list[float], torch.Tensor]:
    last = None
    for _ in range(warmup):
        last = fn()
        _sync(device)

    samples = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        last = fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    assert last is not None
    return samples, last


def _supports_triton(case: BenchCase) -> bool:
    return gguf_triton_available() and case.bits in {"q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k"}


def _build_modules(
    case: BenchCase,
    *,
    dtype: torch.dtype,
) -> tuple[GGUFTorchLinear, GGUFCppKernel, GGUFCudaKernel, GGUFTritonKernel | None]:
    torch.manual_seed(0)
    linear = nn.Linear(case.in_features, case.out_features, bias=case.bias, dtype=torch.float16).cpu().eval()
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)
        if linear.bias is not None:
            linear.bias.normal_(mean=0.0, std=0.01)

    torch_kernel = GGUFTorchLinear(
        bits=case.bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=case.bias,
        register_buffers=True,
    ).eval()
    torch_kernel.pack_original(linear, scales=None, zeros=None)

    cpu_kernel = GGUFCppKernel(
        bits=case.bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=case.bias,
        register_buffers=True,
    ).eval()
    cpu_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)

    cuda_kernel = GGUFCudaKernel(
        bits=case.bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=case.bias,
        register_buffers=True,
    ).eval()
    cuda_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)

    triton_kernel = None
    if _supports_triton(case):
        triton_kernel = GGUFTritonKernel(
            bits=case.bits,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=case.in_features,
            out_features=case.out_features,
            bias=case.bias,
            register_buffers=True,
        ).eval()
        triton_kernel.load_state_dict(torch_kernel.state_dict(), strict=True)

    return torch_kernel, cpu_kernel, cuda_kernel, triton_kernel


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_cpu(case: BenchCase, *, dtype: torch.dtype, warmup: int, trials: int) -> tuple[list[list[str]], list[str]]:
    torch_kernel, cpu_kernel, _, _ = _build_modules(case, dtype=dtype)
    x = torch.randn(case.rows, case.in_features, dtype=dtype, device="cpu")

    torch_trials, torch_out = _bench(lambda: torch_kernel(x), device="cpu", warmup=warmup, trials=trials)
    cpu_trials, cpu_out = _bench(lambda: cpu_kernel(x), device="cpu", warmup=warmup, trials=trials)

    diff = (torch_out.float() - cpu_out.float()).abs()
    trial_rows = []
    for idx, (torch_ms, cpp_ms) in enumerate(zip(torch_trials, cpu_trials), start=1):
        speedup = torch_ms / cpp_ms if cpp_ms > 0 else float("inf")
        trial_rows.append([str(idx), f"{torch_ms:.3f}", f"{cpp_ms:.3f}", f"{speedup:.2f}x"])

    summary = [
        case.name,
        case.bits,
        f"{case.rows}x{case.in_features}",
        f"{case.out_features}x{case.in_features}",
        f"{_mean(torch_trials):.3f}",
        f"{_mean(cpu_trials):.3f}",
        "n/a",
        f"{(_mean(torch_trials) / _mean(cpu_trials)):.2f}x",
        "n/a",
        f"{diff.mean().item():.6f}",
        f"{diff.max().item():.6f}",
        "n/a",
        "n/a",
    ]
    return trial_rows, summary


def _run_cuda(case: BenchCase, *, dtype: torch.dtype, warmup: int, trials: int) -> tuple[list[list[str]], list[str]]:
    torch_kernel, _, cuda_kernel, triton_kernel = _build_modules(case, dtype=dtype)
    torch_kernel = torch_kernel.to("cuda")
    cuda_kernel = cuda_kernel.to("cuda")
    if triton_kernel is not None:
        triton_kernel = triton_kernel.to("cuda")
    x = torch.randn(case.rows, case.in_features, dtype=dtype, device="cuda")

    torch_trials, torch_out = _bench(lambda: torch_kernel(x), device="cuda", warmup=warmup, trials=trials)
    cuda_trials, cuda_out = _bench(lambda: cuda_kernel(x), device="cuda", warmup=warmup, trials=trials)
    triton_trials = None
    triton_out = None
    if triton_kernel is not None:
        triton_trials, triton_out = _bench(lambda: triton_kernel(x), device="cuda", warmup=warmup, trials=trials)

    diff = (torch_out.float() - cuda_out.float()).abs()
    triton_diff = None if triton_out is None else (torch_out.float() - triton_out.float()).abs()
    trial_rows = []
    for idx, (torch_ms, cpp_ms) in enumerate(zip(torch_trials, cuda_trials), start=1):
        cpp_speedup = torch_ms / cpp_ms if cpp_ms > 0 else float("inf")
        if triton_trials is None:
            triton_ms = "n/a"
            triton_speedup = "n/a"
        else:
            trial_triton_ms = triton_trials[idx - 1]
            triton_ms = f"{trial_triton_ms:.3f}"
            triton_speedup = f"{(torch_ms / trial_triton_ms):.2f}x" if trial_triton_ms > 0 else "inf"
        trial_rows.append([str(idx), f"{torch_ms:.3f}", f"{cpp_ms:.3f}", triton_ms, f"{cpp_speedup:.2f}x", triton_speedup])

    summary = [
        case.name,
        case.bits,
        f"{case.rows}x{case.in_features}",
        f"{case.out_features}x{case.in_features}",
        f"{_mean(torch_trials):.3f}",
        f"{_mean(cuda_trials):.3f}",
        "n/a" if triton_trials is None else f"{_mean(triton_trials):.3f}",
        f"{(_mean(torch_trials) / _mean(cuda_trials)):.2f}x",
        "n/a" if triton_trials is None else f"{(_mean(torch_trials) / _mean(triton_trials)):.2f}x",
        f"{diff.mean().item():.6f}",
        f"{diff.max().item():.6f}",
        "n/a" if triton_diff is None else f"{triton_diff.mean().item():.6f}",
        "n/a" if triton_diff is None else f"{triton_diff.max().item():.6f}",
    ]
    return trial_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GGUF cpp kernels against GGUF torch.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=[],
        help="Case as name:bits:in_features:out_features:rows[:bias].",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype-cpu", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--dtype-cuda", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype_cpu]
    cuda_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype_cuda]
    cases = (
        [_parse_case(spec) for spec in args.cases]
        if args.cases
        else [
            BenchCase("attn_q4_k_m_r1", "q4_k_m", 2048, 2048, 1),
            BenchCase("attn_q4_k_m_r8", "q4_k_m", 2048, 2048, 8),
            BenchCase("mlp_q4_k_m_r8", "q4_k_m", 2048, 8192, 8),
            BenchCase("mlp_q5_k_m_r8", "q5_k_m", 2048, 8192, 8),
            BenchCase("mlp_q6_k_r8", "q6_k", 2048, 8192, 8),
        ]
    )

    print(
        f"device={args.device} trials={args.trials} warmup={args.warmup} "
        f"dtype_cpu={str(cpu_dtype).removeprefix('torch.')} dtype_cuda={str(cuda_dtype).removeprefix('torch.')}"
    )

    if args.device in {"cpu", "both"}:
        cpu_summary = []
        print("\nCPU per-trial")
        for case in cases:
            trial_rows, summary = _run_cpu(case, dtype=cpu_dtype, warmup=args.warmup, trials=args.trials)
            print(f"\nCASE {case.name} bits={case.bits} rows={case.rows}")
            print(_ascii_table(["trial", "gguf_torch_ms", "gguf_cpp_cpu_ms", "speedup"], trial_rows))
            cpu_summary.append(summary)
        print("\nCPU summary")
        print(
            _ascii_table(
                [
                    "case",
                    "bits",
                    "rowsxin",
                    "outxin",
                    "gguf_torch_ms",
                    "gguf_cpp_cpu_ms",
                    "gguf_triton_ms",
                    "speedup",
                    "triton_speedup",
                    "mae",
                    "max_abs",
                    "triton_mae",
                    "triton_max_abs",
                ],
                cpu_summary,
            )
        )

    if args.device in {"cuda", "both"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is False.")
        cuda_summary = []
        print("\nCUDA per-trial")
        for case in cases:
            trial_rows, summary = _run_cuda(case, dtype=cuda_dtype, warmup=args.warmup, trials=args.trials)
            print(f"\nCASE {case.name} bits={case.bits} rows={case.rows}")
            print(
                _ascii_table(
                    ["trial", "gguf_torch_ms", "gguf_cpp_cuda_ms", "gguf_triton_ms", "cpp_speedup", "triton_speedup"],
                    trial_rows,
                )
            )
            cuda_summary.append(summary)
        print("\nCUDA summary")
        print(
            _ascii_table(
                [
                    "case",
                    "bits",
                    "rowsxin",
                    "outxin",
                    "gguf_torch_ms",
                    "gguf_cpp_cuda_ms",
                    "gguf_triton_ms",
                    "cpp_speedup",
                    "triton_speedup",
                    "cpp_mae",
                    "cpp_max_abs",
                    "triton_mae",
                    "triton_max_abs",
                ],
                cuda_summary,
            )
        )


if __name__ == "__main__":
    main()
