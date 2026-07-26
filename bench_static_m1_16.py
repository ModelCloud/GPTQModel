#!/usr/bin/env python3
"""Benchmark the committed Amplin static routing table for M=1-16 small-batch shapes."""
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
from gptqmodel.utils.amplin import _call_kernel_chunked, select_kernel
from scripts.benchmark_amplin_model_shapes import SHAPES
from scripts.benchmark_amplin_vs_marlin import (
    _build_marlin,
    _dequantized_reference,
    _dtype_name,
    _make_case,
    _measure,
    _raw_marlin_call,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="m1_16_static_bench.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    m_values = (1, 2, 4, 6, 8, 16)
    dtypes = (torch.float16, torch.bfloat16)
    rows = []
    for dtype in dtypes:
        for spec in SHAPES:
            for size_m in m_values:
                input_t, qweight, scales = _make_case(
                    device=device,
                    dtype=dtype,
                    size_m=size_m,
                    size_k=spec.size_k,
                    size_n=spec.size_n,
                    seed=args.seed,
                )
                if hasattr(amplin, "clear_thread_caches"):
                    amplin.clear_thread_caches()
                if hasattr(amplin, "_ORIGINAL_WEIGHT_RESIDENCY"):
                    amplin._ORIGINAL_WEIGHT_RESIDENCY.clear()
                ref = _dequantized_reference(input_t, qweight, scales)
                if hasattr(amplin, "clear_dynamic_routing_table"):
                    amplin.clear_dynamic_routing_table()

                # Pre-pack once for the target decode batch, matching AmplinLinear.post_init.
                dispatch = select_kernel(
                    qweight,
                    scales,
                    size_m=size_m,
                    dtype=dtype,
                    device=device,
                    logical_n=spec.size_n,
                    update_dynamic_table=False,
                )
                out = _call_kernel_chunked(dispatch, input_t)
                err = (out.to(torch.float32) - ref).abs().max().item()

                marlin_legal = spec.size_n % 64 == 0
                if marlin_legal:
                    marlin_module = _build_marlin(device=device, dtype=dtype, qweight=qweight, scales=scales)
                    marlin_ext = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
                    marlin_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
                    marlin_op = extension.op(marlin_ext, marlin_name)
                    functions = {
                        "amplin": lambda d=dispatch: _call_kernel_chunked(d, input_t),
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
                    functions = {"amplin": lambda d=dispatch: _call_kernel_chunked(d, input_t)}
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

                selected = dispatch.name
                rows.append({
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
                })

    Path(args.out).write_text(json.dumps(rows, indent=2))
    legal = [r for r in rows if r["marlin_legal"]]
    wins = sum(1 for r in legal if r["speedup"] is not None and r["speedup"] > 1.0)
    losses = [r for r in legal if r["speedup"] is not None and r["speedup"] <= 1.0]
    speedups = [r["speedup"] for r in legal if r["speedup"] is not None]
    print(f"Total={len(rows)} legal={len(legal)} wins={wins} losses={len(losses)} geomean={geometric_mean(speedups):.3f}")
    if losses:
        print("Losses:")
        for r in losses:
            print(
                f"  {r['model']} {r['role']} M={r['m']} K={r['k']} N={r['n']} {r['dtype']} "
                f"amplin={r['amplin_us']} marlin={r['marlin_us']} speedup={r['speedup']} kernel={r['selected_kernel']}"
            )


if __name__ == "__main__":
    main()
