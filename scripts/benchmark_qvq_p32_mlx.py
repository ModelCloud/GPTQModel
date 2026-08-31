# SPDX-License-Identifier: Apache-2.0

"""Benchmark the promoted standard-P32 MLX path on Apple silicon."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_mlx import (
    QVQMLXLinear,
    _qvq_mlx_hadamard,
    _qvq_mlx_hadamard_matrix,
)
from gptqmodel.utils.qvq_p32_mlx import (
    qvq_mlx_p32_window_gemv,
    qvq_mlx_repack_p32_planar_to_window,
)

DEFAULT_RATES = (1, 1.5, 2, 2.5, 3, 3.5)
DEFAULT_SHAPES = (
    (1, 2048, 256),
    (1, 2048, 2048),
    (1, 2048, 8192),
    (1, 8192, 2048),
    (2, 2048, 8192),
    (4, 2048, 8192),
    (8, 2048, 8192),
    (16, 8192, 8192),
)


def _payload(bits: float, k: int, n: int, generator: torch.Generator):
    transition_bits = qvq_transition_bits(bits)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, tile_count),
        generator=generator,
        dtype=torch.int32,
    )
    planar = planar_pack_rows(edges, transition_bits).T.contiguous()
    selectors = torch.randint(0, 2, (tile_count * 8,), generator=generator, dtype=torch.uint8)
    return planar, pack_qvq_binary_bank_ids(selectors)


def _measure(function, *, warmup: int, samples: int) -> tuple[float, float]:
    def evaluate():
        output = function()
        if isinstance(output, tuple):
            mx.eval(*output)
        else:
            mx.eval(output)

    for _ in range(warmup):
        evaluate()
    mx.synchronize()
    elapsed = []
    for _ in range(samples):
        start = time.perf_counter()
        evaluate()
        mx.synchronize()
        elapsed.append((time.perf_counter() - start) * 1e3)
    values = np.asarray(elapsed)
    return float(np.median(values)), float(np.percentile(values, 95))


def _parse_rates(value: str) -> tuple[float, ...]:
    rates = tuple(float(item) for item in value.split(","))
    if not rates or any(rate not in DEFAULT_RATES for rate in rates):
        raise argparse.ArgumentTypeError("rates must be a comma-separated subset of 1,1.5,2,2.5,3,3.5")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=_parse_rates, default=DEFAULT_RATES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")

    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    results = []
    print("rate shape | inner p50/p95 ms | transform p50/p95 ms | module p50/p95 ms | EBPW")
    for bits in args.rates:
        for m, k, n in DEFAULT_SHAPES:
            planar_torch, selectors_torch = _payload(bits, k, n, generator)
            planar = mx.array(planar_torch.numpy())
            selectors = mx.array(selectors_torch.numpy())
            bank_alt = mx.array(np.array([2], dtype=np.uint8))
            window = qvq_mlx_repack_p32_planar_to_window(planar, bits)
            mx.eval(window)
            x_inner = mx.array(rng.standard_normal((m, k)).astype(np.float32))
            x_module = x_inner.astype(mx.float16)
            transform_probe = mx.array(rng.standard_normal((m, n)).astype(np.float32))
            input_hadamard = _qvq_mlx_hadamard_matrix(k)
            output_hadamard = _qvq_mlx_hadamard_matrix(n)
            module = QVQMLXLinear(
                bits=bits,
                in_features=k,
                out_features=n,
                trellis=planar,
                SU=mx.ones((k,), dtype=mx.float32),
                SV=mx.ones((n,), dtype=mx.float32),
                bank_ids=selectors,
                v2b2_p32=True,
                bank_alt_id=bank_alt,
            )

            inner = _measure(
                partial(
                    qvq_mlx_p32_window_gemv,
                    x_inner,
                    window,
                    bits,
                    out_features=n,
                    bank_ids=selectors,
                    bank_alt_id=2,
                ),
                warmup=args.warmup,
                samples=args.samples,
            )
            def run_transform(
                input_value=x_inner,
                input_matrix=input_hadamard,
                output_value=transform_probe,
                output_matrix=output_hadamard,
            ):
                return (
                    _qvq_mlx_hadamard(input_value, input_matrix),
                    _qvq_mlx_hadamard(output_value, output_matrix),
                )

            transform = _measure(
                run_transform,
                warmup=args.warmup,
                samples=args.samples,
            )
            combined = _measure(
                partial(module, x_module),
                warmup=args.warmup,
                samples=args.samples,
            )
            payload_bits = (
                planar_torch.numel() * 32
                + selectors_torch.numel() * 8
                + 8
                + (k + n) * 32
            )
            ebpw = payload_bits / (k * n)
            row = {
                "rate": bits,
                "m": m,
                "k": k,
                "n": n,
                "inner_p50_ms": inner[0],
                "inner_p95_ms": inner[1],
                "transform_p50_ms": transform[0],
                "transform_p95_ms": transform[1],
                "module_p50_ms": combined[0],
                "module_p95_ms": combined[1],
                "effective_bpw": ebpw,
            }
            results.append(row)
            print(
                f"W{bits:g} ({m},{k},{n}) | {inner[0]:.5f}/{inner[1]:.5f} | "
                f"{transform[0]:.5f}/{transform[1]:.5f} | "
                f"{combined[0]:.5f}/{combined[1]:.5f} | {ebpw:.6f}"
            )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "device": mx.device_info(),
                    "warmup": args.warmup,
                    "samples": args.samples,
                    "storage_accounting": "canonical P32 payload + packed selectors + FP32 SU/SV + one alternative-bank byte",
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
