# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired SM80 BF16 projection gate for the Lite NF12 candidate.

Example:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 PYTHON_GIL=0 python scripts/bench_nf12_lite_bf16.py \
        --device cuda:0 --json-out /tmp/nf12_lite_bf16.json

The dense control receives the same BF16 activation and original BF16 weight.
Encoding happens once, outside timing, as it does during Lite model load. CUDA
graphs amortize host dispatch. These are shape probes, not serving throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import gptqmodel_nf12 as nf12


# N, K, M. Router/FP32-gate shapes are BF16 kernel diagnostics only: Lite must
# retain FP32 routing arithmetic until a model-level decision oracle approves it.
CASES = {
    "index_score_decode": (32, 5120, 6),
    "index_key_decode": (128, 512, 6),
    "compressor_kv_decode": (512, 5120, 6),
    "markov_decode": (129280, 256, 5),
    "vocab_decode": (129280, 5120, 5),
    "gate_shape_m1": (384, 5120, 1),
    "gate_shape_m6": (384, 5120, 6),
    "gate_shape_m32": (384, 5120, 32),
    "gate_shape_m512": (384, 5120, 512),
    "mtp_gate_shape_m1": (128, 5120, 1),
    "mtp_gate_shape_m5": (128, 5120, 5),
    "mtp_gate_shape_m32": (128, 5120, 32),
    "mtp_gate_shape_m512": (128, 5120, 512),
    "compressor_kv_prefill": (512, 5120, 512),
}


def capture(fn, repeat: int) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    return graph


def graph_us(graph: torch.cuda.CUDAGraph, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def bench_case(name: str, shape: tuple[int, int, int], repeat: int,
               rounds: int, target: float, device: torch.device) -> dict:
    n, k, m = shape
    generator = torch.Generator(device=device).manual_seed(18201 + n + m)
    weight = (torch.randn((n, k), generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((m, k), generator=generator, device=device) * 0.05).bfloat16()
    packed = nf12.encode(weight)
    def dense_fn():
        return F.linear(x, weight)

    def nf12_fn():
        return nf12.linear(x, packed)
    dense = dense_fn()
    actual = nf12_fn()
    torch.testing.assert_close(actual, dense, rtol=2e-3, atol=2e-3)
    if set(vars(packed)) != {"packed", "escapes", "exp_base", "shape"}:
        raise AssertionError("NF12Weight retained an unexpected dense tensor")
    top = min(n, 8)
    topk_mismatch_rows = int((actual.topk(top, dim=-1).indices !=
                              dense.topk(top, dim=-1).indices).any(dim=-1).sum().item())
    dense_graph = capture(dense_fn, repeat)
    nf12_graph = capture(nf12_fn, repeat)
    torch.cuda.synchronize()
    for _ in range(4):
        dense_graph.replay()
        nf12_graph.replay()
    torch.cuda.synchronize()
    dense_times, nf12_times = [], []
    for _ in range(rounds):
        # ABBA balances drift from clocks and neighboring production work.
        dense_times.append(graph_us(dense_graph, repeat))
        nf12_times.append(graph_us(nf12_graph, repeat))
        nf12_times.append(graph_us(nf12_graph, repeat))
        dense_times.append(graph_us(dense_graph, repeat))
    dense_us = statistics.median(dense_times)
    nf12_us = statistics.median(nf12_times)
    result = {
        "case": name, "n": n, "k": k, "m": m,
        "dense_bf16_us": round(dense_us, 3),
        "nf12_bf16_us": round(nf12_us, 3),
        "speedup": round(dense_us / nf12_us, 3),
        "target": target,
        "passes_speed": dense_us >= target * nf12_us,
        "max_abs_error": float((actual.float() - dense.float()).abs().max().item()),
        "topk_mismatch_rows": topk_mismatch_rows,
        "packed_bytes": packed.nbytes,
        "dense_bytes": weight.nbytes,
    }
    del dense_graph, nf12_graph, dense, actual
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", action="append", choices=sorted(CASES),
                        help="Run only this case; may be repeated")
    parser.add_argument("--repeat", type=int, default=32,
                        help="Calls per captured graph")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Paired ABBA rounds")
    parser.add_argument("--target", type=float, default=1.15)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if any speed gate fails")
    parser.add_argument("--require-decisions", action="store_true",
                        help="Exit nonzero if any top-8 selected IDs differ")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.rounds < 1 or args.target <= 0:
        parser.error("repeat, rounds and target must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        parser.error("an NVIDIA SM80 or newer CUDA device is required")
    nf12.load()
    results = []
    for name in (args.case or CASES):
        result = bench_case(name, CASES[name], args.repeat, args.rounds,
                            args.target, device)
        results.append(result)
        print(f"{name}: NF12 {result['nf12_bf16_us']:.3f} us, "
              f"dense {result['dense_bf16_us']:.3f} us, "
              f"{result['speedup']:.3f}x, speed gate "
              f"{'PASS' if result['passes_speed'] else 'FAIL'}, "
              f"top-k mismatch rows {result['topk_mismatch_rows']}", flush=True)
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "nf12_package": str(Path(nf12.__file__).resolve()),
        "repeat": args.repeat, "rounds": args.rounds, "target": args.target,
        "cases": results,
        "all_speed_gates_pass": all(r["passes_speed"] for r in results),
        "all_decisions_match": all(r["topk_mismatch_rows"] == 0 for r in results),
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return int((args.strict and not report["all_speed_gates_pass"]) or
               (args.require_decisions and not report["all_decisions_match"]))


if __name__ == "__main__":
    raise SystemExit(main())
