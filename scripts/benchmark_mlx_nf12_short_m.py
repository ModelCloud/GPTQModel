# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired NF12 Metal benchmark for short batched-decode and prefill rows.

This is the Apple-silicon counterpart of ``benchmark_nf12_short_m.py``: it uses
the same ``(N, K)`` shapes and the same M values so the Metal numbers can be
read next to the CUDA and ROCm tables in
``docs/nf12-short-m-dispatch.md``.

The comparison baseline is dense BF16 ``x @ W.T`` on the same resident input,
and the reported ratio is dense time divided by NF12 time, so values below 1.0
are slower than dense.

A synchronized per-call measurement is not usable here: one ``mx.eval`` costs
roughly 240 us on this host regardless of the work it waits for, which is an
order of magnitude above the kernels being measured. Each trial therefore
enqueues a fixed number of calls and evaluates them once, which measures
steady-state throughput rather than single-shot latency. The same harness is
used for both sides of every ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch


DEFAULT_SHAPES = (
    (2048, 2048),
    (512, 2048),
    (8192, 2048),
    (2048, 8192),
    (4096, 4096),
    (128256, 2048),
)
DEFAULT_ROWS = (1, 4, 8, 16)


def _median_us(fn: Callable[[], object], *, mx, warmup: int, trials: int, iterations: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        mx.eval(*[fn() for _ in range(iterations)])
        samples.append((time.perf_counter_ns() - start) / iterations / 1_000)
    return statistics.median(samples)


def _build(n: int, k: int):
    import mlx.core as mx

    from gptqmodel.nn_modules.qlinear.mlx_nf12 import MlxNF12Linear
    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(k, n, bias=False)
    source = NF12Linear(
        bits=12, in_features=k, out_features=n, bias=False, dtype=torch.bfloat16,
    )
    source.pack_original(linear)
    layer = MlxNF12Linear(in_features=k, out_features=n, num_escapes=source.nf12_escapes.shape[0])
    layer.packed = mx.array(source.nf12_packed.numpy())
    layer.escapes = mx.array(source.nf12_escapes.numpy())
    layer.exp_base = mx.array(source.nf12_exp_base.numpy())
    layer.freeze()
    weight = mx.array(linear.weight.detach().float().numpy()).astype(mx.bfloat16)
    return layer, weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=None, help="Comma separated NxK pairs, for example 2048x2048")
    parser.add_argument("--rows", default=None, help="Comma separated M values, for example 1,4,8,16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = tuple(
            tuple(int(part) for part in item.lower().split("x")) for item in args.shapes.split(",")
        )
    rows = DEFAULT_ROWS
    if args.rows:
        rows = tuple(int(item) for item in args.rows.split(","))

    device = mx.device_info()
    print(f"device: {device.get('device_name')} ({device.get('architecture')})")
    print(f"mlx: {mx.__version__}")
    results = []
    print(f"{'shape':>14} {'M':>3} {'dense us':>9} {'nf12 us':>9} {'ratio':>7}")
    for n, k in shapes:
        layer, weight = _build(n, k)
        for m in rows:
            x = mx.array(
                np.random.default_rng(m).normal(0, 0.2, (m, k)).astype(np.float32)
            ).astype(mx.bfloat16)
            # Keep the enqueued graph near a fixed footprint: a 192 MB weight
            # matrix cannot hold 40 live copies.
            iterations = max(4, min(args.iterations, int(2e9 // max(n * k * 2, 1))))
            dense = _median_us(
                lambda: mx.matmul(x, weight.T),
                mx=mx, warmup=args.warmup, trials=args.trials, iterations=iterations,
            )
            nf12 = _median_us(
                lambda: layer(x),
                mx=mx, warmup=args.warmup, trials=args.trials, iterations=iterations,
            )
            ratio = dense / nf12
            results.append({
                "n": n, "k": k, "m": m,
                "dense_us": round(dense, 3), "nf12_us": round(nf12, 3),
                "ratio": round(ratio, 3),
            })
            print(f"{n}x{k:>7} {m:>3} {dense:>9.1f} {nf12:>9.1f} {ratio:>6.2f}x", flush=True)

    if args.json_out:
        payload = {
            "device": device,
            "mlx": mx.__version__,
            "method": "batched queue, median of trials",
            "warmup": args.warmup,
            "trials": args.trials,
            "iterations": args.iterations,
            "results": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
