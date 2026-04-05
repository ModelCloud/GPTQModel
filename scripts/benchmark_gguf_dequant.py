#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gguf
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


@dataclass(frozen=True)
class BenchCase:
    name: str
    bits: str
    in_features: int
    out_features: int
    group_size: int


def _build_module(case: BenchCase, dtype: torch.dtype) -> GGUFTorchLinear:
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
    return module.cpu().eval()


def _gguf_qtype(bits: str) -> gguf.GGMLQuantizationType:
    mapping = {
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q4_k": gguf.GGMLQuantizationType.Q4_K,
        "q4_k_s": gguf.GGMLQuantizationType.Q4_K,
        "q4_k_m": gguf.GGMLQuantizationType.Q4_K,
        "q5_k": gguf.GGMLQuantizationType.Q5_K,
        "q5_k_s": gguf.GGMLQuantizationType.Q5_K,
        "q5_k_m": gguf.GGMLQuantizationType.Q5_K,
        "q6_k": gguf.GGMLQuantizationType.Q6_K,
    }
    return mapping[bits]


def _bench(fn, *, iters: int, warmup: int, sync_cuda: bool) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return sum(samples_ms) / len(samples_ms), min(samples_ms), max(samples_ms)


def _print_row(label: str, mean_ms: float, min_ms: float, max_ms: float) -> None:
    print(f"{label:28s} mean={mean_ms:8.3f} ms min={min_ms:8.3f} max={max_ms:8.3f}")


def run_case(case: BenchCase, *, dtype: torch.dtype, device: str, iters: int, warmup: int) -> None:
    module_cpu = _build_module(case, dtype=dtype)
    qweight_np = module_cpu.qweight.detach().cpu().numpy()
    qtype = _gguf_qtype(case.bits)

    print()
    print(
        f"CASE {case.name} bits={case.bits} "
        f"shape={case.out_features}x{case.in_features} group_size={case.group_size}"
    )

    mean_ms, min_ms, max_ms = _bench(
        lambda: gguf.dequantize(qweight_np, qtype),
        iters=iters,
        warmup=warmup,
        sync_cuda=False,
    )
    _print_row("gguf.dequantize cpu", mean_ms, min_ms, max_ms)

    mean_ms, min_ms, max_ms = _bench(
        lambda: module_cpu.dequantize_weight(device="cpu", dtype=torch.float32),
        iters=iters,
        warmup=warmup,
        sync_cuda=False,
    )
    _print_row("gptqmodel dequant cpu fp32", mean_ms, min_ms, max_ms)

    if device == "cuda":
        module_gpu = module_cpu.to("cuda").eval()
        x = torch.randn(1, case.in_features, device="cuda", dtype=dtype)

        mean_ms, min_ms, max_ms = _bench(
            lambda: module_gpu.dequantize_weight(device="cuda", dtype=dtype),
            iters=iters,
            warmup=warmup,
            sync_cuda=True,
        )
        _print_row(f"gptqmodel dequant cuda {str(dtype).removeprefix('torch.')}", mean_ms, min_ms, max_ms)

        def cold_forward():
            module_gpu.clear_weight_cache()
            module_gpu(x)

        def hot_forward():
            module_gpu(x)

        mean_ms, min_ms, max_ms = _bench(
            cold_forward,
            iters=iters,
            warmup=warmup,
            sync_cuda=True,
        )
        _print_row("gptqmodel forward cold", mean_ms, min_ms, max_ms)

        hot_forward()
        torch.cuda.synchronize()
        mean_ms, min_ms, max_ms = _bench(
            hot_forward,
            iters=max(iters * 2, 10),
            warmup=warmup,
            sync_cuda=True,
        )
        _print_row("gptqmodel forward hot", mean_ms, min_ms, max_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-benchmark GGUF dequantization and forward paths.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=[],
        help="Benchmark case as name:bits:in_features:out_features[:group_size]. "
        "Example: attn:q4_k_m:2048:2048:128",
    )
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations per benchmark.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per benchmark.")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Target GPU forward/dequant dtype when CUDA is available.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Benchmark target for dequant/forward. `auto` chooses CUDA when available.",
    )
    return parser.parse_args()


def _parse_case(spec: str) -> BenchCase:
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            f"Invalid case `{spec}`. Expected name:bits:in_features:out_features[:group_size]."
        )
    name, bits, in_features, out_features, *rest = parts
    group_size = int(rest[0]) if rest else -1
    return BenchCase(
        name=name,
        bits=bits,
        in_features=int(in_features),
        out_features=int(out_features),
        group_size=group_size,
    )


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    else:
        device = args.device

    cases = (
        [_parse_case(spec) for spec in args.cases]
        if args.cases
        else [
            BenchCase("attn_q4_0", "q4_0", 2048, 2048, 128),
            BenchCase("attn_q4_k_m", "q4_k_m", 2048, 2048, 128),
            BenchCase("mlp_q4_0", "q4_0", 2048, 8192, 128),
            BenchCase("mlp_q4_k_m", "q4_k_m", 2048, 8192, 128),
        ]
    )

    print(f"device={device} dtype={str(dtype).removeprefix('torch.')} cuda_available={torch.cuda.is_available()}")
    for case in cases:
        run_case(case, dtype=dtype, device=device, iters=args.iters, warmup=args.warmup)


if __name__ == "__main__":
    main()
