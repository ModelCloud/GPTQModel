# SPDX-License-Identifier: Apache-2.0

"""Benchmark complete MLX QVQ linear modules for LOCAL-RING vs P32."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Make the benchmark use the checkout that owns it when invoked by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel.utils.qvq_mlx import QVQMLXLinear

DEFAULT_SHAPES = (
    (1, 2048, 256),
    (1, 2048, 2048),
    (1, 2048, 8192),
    (1, 8192, 2048),
    (4, 2048, 8192),
    (8, 2048, 8192),
    (16, 8192, 8192),
)


def _module(
    m: int,
    k: int,
    n: int,
    *,
    lr: bool,
    trellis,
    bank_ids,
    bank_alt_id,
):
    return QVQMLXLinear(
        bits=2,
        in_features=k,
        out_features=n,
        trellis=trellis,
        SU=mx.ones((k,), dtype=mx.float32),
        SV=mx.ones((n,), dtype=mx.float32),
        vector_size=2,
        trellis_window=16,
        bank_ids=bank_ids,
        v2b2_p32=not lr,
        v2b2_p32_lr=lr,
        bank_alt_id=bank_alt_id,
    )


def _measure_pair(
    lr_module,
    p32_module,
    x,
    *,
    warmup: int,
    samples: int,
    compile_module: bool,
    rng: np.random.Generator,
):
    runners = [
        mx.compile(lr_module) if compile_module else lr_module,
        mx.compile(p32_module) if compile_module else p32_module,
    ]

    def run(runner):
        output = runner(x)
        mx.eval(output)
        mx.synchronize()

    # Compilation/materialization is never part of a timed sample.  This also
    # keeps --compile --warmup 0 from recording the first specialization cost.
    if compile_module:
        for runner in runners:
            run(runner)
    for _ in range(warmup):
        order = [0, 1]
        rng.shuffle(order)
        for index in order:
            run(runners[index])
    elapsed = [[], []]
    for _ in range(samples):
        order = [0, 1]
        rng.shuffle(order)
        for index in order:
            start = time.perf_counter()
            run(runners[index])
            elapsed[index].append((time.perf_counter() - start) * 1e3)
    values = [np.asarray(samples) for samples in elapsed]
    return tuple((float(np.median(value)), float(np.percentile(value, 95))) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap each fixed-shape module in mx.compile before timing (shape-specialized research mode).",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")

    rng = np.random.default_rng(args.seed)
    print("shape | LR p50 ms | P32 p50 ms | speedup | LR p95 ms | P32 p95 ms")
    for m, k, n in DEFAULT_SHAPES:
        x = mx.array(rng.standard_normal((m, k)).astype(np.float16))
        tile_count = (k // 32) * (n // 8)
        # LR32 K32xN8 and P32 K16xN16 have the same number of W2 payload
        # tiles.  Sharing the exact payload and selector bytes removes a
        # benchmark confounder while preserving each format's decoder.
        trellis = mx.array(rng.integers(-2**31, 2**31, (tile_count, 16), dtype=np.int32))
        bank_ids = mx.array(rng.integers(0, 256, tile_count, dtype=np.uint8))
        bank_alt_id = mx.array(np.array([1], dtype=np.uint8))
        lr_module = _module(
            m,
            k,
            n,
            lr=True,
            trellis=trellis,
            bank_ids=bank_ids,
            bank_alt_id=bank_alt_id,
        )
        p32_module = _module(
            m,
            k,
            n,
            lr=False,
            trellis=trellis,
            bank_ids=bank_ids,
            bank_alt_id=bank_alt_id,
        )
        (lr, p32) = _measure_pair(
            lr_module,
            p32_module,
            x,
            warmup=args.warmup,
            samples=args.samples,
            compile_module=args.compile,
            rng=rng,
        )
        print(
            f"({m},{k},{n}) | {lr[0]:.5f} | {p32[0]:.5f} | {p32[0] / lr[0]:.3f} "
            f"| {lr[1]:.5f} | {p32[1]:.5f}"
        )


if __name__ == "__main__":
    main()
