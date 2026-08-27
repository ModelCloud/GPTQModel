# SPDX-License-Identifier: Apache-2.0

"""Benchmark the MLX LOCAL-RING kernel against selector-aware P32.

This deliberately measures the public ``qvq_mlx_gemv`` path, including its
LR split-K reduction, rather than timing a decoded dense matrix multiply.
Both arms use synthetic W2 payloads with the same input, warmup, and sample
policy.  The output is FP32 because that is the production QVQMLXLinear path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Make the benchmark use the checkout that owns it when invoked by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel.utils.qvq_mlx import qvq_mlx_gemv

DEFAULT_SHAPES = (
    (1, 2048, 256),
    (1, 2048, 2048),
    (1, 2048, 8192),
    (1, 8192, 2048),
    (4, 2048, 8192),
    (8, 2048, 8192),
    (16, 8192, 8192),
)


def _arrays(rng: np.random.Generator, m: int, k: int, n: int, *, lr: bool):
    x = mx.array(rng.standard_normal((m, k)).astype(np.float16))
    tile_count = (k // 32) * (n // 8) if lr else (k // 16) * (n // 16)
    trellis = mx.array(rng.integers(-2**31, 2**31, (tile_count, 16), dtype=np.int32))
    bank_ids = mx.array(rng.integers(0, 256, tile_count, dtype=np.uint8))
    bank_alt_id = mx.array(np.array([1], dtype=np.uint8))
    return x, trellis, bank_ids, bank_alt_id


def _measure(
    x,
    trellis,
    bank_ids,
    bank_alt_id,
    n: int,
    *,
    lr: bool,
    warmup: int,
    samples: int,
):
    # Mirror QVQMLXLinear: this immutable metadata is resolved once during
    # setup so the timed inner loop contains no MLX ``array.item()`` boundary.
    bank_alt_id_value = int(bank_alt_id.item())

    def run():
        output = qvq_mlx_gemv(
            x,
            trellis,
            2,
            out_features=n,
            bank_ids=bank_ids,
            bank_alt_id=bank_alt_id,
            v2b2_p32=not lr,
            v2b2_p32_lr=lr,
            output_fp32=True,
            _bank_alt_id_value=bank_alt_id_value,
        )
        mx.eval(output)
        mx.synchronize()

    for _ in range(warmup):
        run()
    elapsed = []
    for _ in range(samples):
        start = time.perf_counter()
        run()
        elapsed.append((time.perf_counter() - start) * 1e3)
    values = np.asarray(elapsed)
    return float(np.median(values)), float(np.percentile(values, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--samples", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")

    rng = np.random.default_rng(args.seed)
    print("shape | LR p50 ms | P32 p50 ms | speedup | LR p95 ms | P32 p95 ms")
    for m, k, n in DEFAULT_SHAPES:
        lr_args = _arrays(rng, m, k, n, lr=True)
        p32_args = _arrays(rng, m, k, n, lr=False)
        lr = _measure(*lr_args, n, lr=True, warmup=args.warmup, samples=args.samples)
        p32 = _measure(*p32_args, n, lr=False, warmup=args.warmup, samples=args.samples)
        print(
            f"({m},{k},{n}) | {lr[0]:.5f} | {p32[0]:.5f} | {p32[0] / lr[0]:.3f} "
            f"| {lr[1]:.5f} | {p32[1]:.5f}"
        )


if __name__ == "__main__":
    main()
