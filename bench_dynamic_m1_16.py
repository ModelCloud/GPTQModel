#!/usr/bin/env python3
"""Benchmark Amplin dynamic routing for M=1-16 small-batch shapes vs raw Marlin.

This script forces a fresh micro-benchmark selection for every (M, K, N, dtype)
so it tests the dynamic routing path rather than the committed static table.
It is intended for commit-by-commit regression hunting.
"""
import gc
import json
import math
import sys
from pathlib import Path
from statistics import geometric_mean


SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parent
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from gptqmodel import extension
from gptqmodel.utils import amplin
from scripts.benchmark_amplin_model_shapes import SHAPES
from scripts.benchmark_amplin_vs_marlin import (
    _build_marlin,
    _dequantized_reference,
    _dtype_name,
    _make_case,
    _measure,
    _raw_marlin_call,
)


def _clear_amplin_caches() -> None:
    # Older commits: global weight cache
    if hasattr(amplin, "_WEIGHT_CACHE"):
        amplin._WEIGHT_CACHE.clear()
    # Newer commits: thread-local caches and single-residency manager
    if hasattr(amplin, "clear_thread_caches"):
        amplin.clear_thread_caches()
    if hasattr(amplin, "_ORIGINAL_WEIGHT_RESIDENCY"):
        amplin._ORIGINAL_WEIGHT_RESIDENCY.clear()
    if hasattr(amplin, "clear_dynamic_routing_table"):
        amplin.clear_dynamic_routing_table()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="m1_16_dynamic_bench.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--clear-static", action="store_true", help="Ignore the bundled static routing table.")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if args.clear_static:
        if hasattr(amplin, "set_routing_table"):
            amplin.set_routing_table({})

    m_values = (1, 2, 4, 6, 8, 16)
    dtypes = (torch.float16, torch.bfloat16)
    rows = []
    for dtype in dtypes:
        for spec in SHAPES:
            for size_m in m_values:
                _clear_amplin_caches()

                input_t, qweight, scales = _make_case(
                    device=device,
                    dtype=dtype,
                    size_m=size_m,
                    size_k=spec.size_k,
                    size_n=spec.size_n,
                    seed=args.seed,
                )
                ref = _dequantized_reference(input_t, qweight, scales)

                # Force a fresh dynamic micro-benchmark for this shape.
                out = amplin.dynamic(input_t, qweight, scales, iters=args.iters)
                err = (out.to(torch.float32) - ref).abs().max().item()

                marlin_legal = spec.size_n % 64 == 0
                if marlin_legal:
                    marlin_module = _build_marlin(device=device, dtype=dtype, qweight=qweight, scales=scales)
                    marlin_ext = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
                    marlin_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
                    marlin_op = extension.op(marlin_ext, marlin_name)
                    functions = {
                        "amplin": lambda: amplin.dynamic(input_t, qweight, scales, iters=None),
                        "marlin": lambda: _raw_marlin_call(op=marlin_op, input=input_t, module=marlin_module),
                    }
                    stats = _measure(
                        functions,
                        device=device,
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                        pre_timing_check=None,
                    )
                    amplin_t = stats["amplin"].batch_event_median_us
                    marlin_t = stats["marlin"].batch_event_median_us
                    speedup = marlin_t / amplin_t
                else:
                    functions = {"amplin": lambda: amplin.dynamic(input_t, qweight, scales, iters=None)}
                    stats = _measure(
                        functions,
                        device=device,
                        warmup=args.warmup,
                        iters=args.iters,
                        rounds=args.rounds,
                        pre_timing_check=None,
                    )
                    amplin_t = stats["amplin"].batch_event_median_us
                    marlin_t = 0.0
                    speedup = float("nan")

                key = (size_m, spec.size_k, spec.size_n, _dtype_name(dtype))
                selected = amplin.get_dynamic_routing_table().get(key, amplin.get_static_routing_table().get(key, ""))
                rows.append(
                    {
                        "model": spec.model,
                        "role": spec.role,
                        "m": size_m,
                        "k": spec.size_k,
                        "n": spec.size_n,
                        "dtype": _dtype_name(dtype),
                        "amplin_us": round(amplin_t, 2),
                        "marlin_us": round(marlin_t, 2),
                        "speedup": round(speedup, 3) if not math.isnan(speedup) else None,
                        "max_abs_error": err,
                        "selected_kernel": selected,
                        "marlin_legal": marlin_legal,
                    }
                )

    Path(args.out).write_text(json.dumps(rows, indent=2))
    legal = [r for r in rows if r["marlin_legal"]]
    wins = sum(1 for r in legal if r["speedup"] is not None and r["speedup"] > 1.0)
    losses = [r for r in legal if r["speedup"] is not None and r["speedup"] <= 1.0]
    speedups = [r["speedup"] for r in legal if r["speedup"] is not None]
    print(f"Total={len(rows)} legal={len(legal)} wins={wins} losses={len(losses)} geomean={geometric_mean(speedups):.3f}")
    if losses:
        print("Losses:")
        for r in losses[:30]:
            print(
                f"  {r['model']} {r['role']} M={r['m']} K={r['k']} N={r['n']} {r['dtype']} "
                f"amplin={r['amplin_us']} marlin={r['marlin_us']} speedup={r['speedup']} kernel={r['selected_kernel']}"
            )


if __name__ == "__main__":
    main()
