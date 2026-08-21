#!/usr/bin/env python3
"""Benchmark QVQ CPU fused YAQA factored feedback against the Python reference."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.utils.qvq_cpu import qvq_cpu_yaqa_feedback  # noqa: E402


def _python_feedback(source, left, right, output_feedback, first_input, first_output, count, bias):
    tile = 16
    tiles = []
    for t in range(count):
        i0 = (first_input + t) * tile
        o0 = (first_output - t) * tile
        cross = left[i0 : i0 + tile, o0:] @ output_feedback[o0:, o0 : o0 + tile]
        tile_src = source[i0 : i0 + tile, o0 : o0 + tile]
        tile_bias = 0.0 if bias is None else bias[i0 : i0 + tile, o0 : o0 + tile]
        tiles.append(tile_src + tile_bias + cross + left[i0 : i0 + tile, o0 : o0 + tile] + right[i0 : i0 + tile, o0 : o0 + tile])
    return torch.stack(tiles)


def _timings(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    in_features = args.in_features
    out_features = args.out_features
    tile = 16
    if in_features % tile or out_features % tile:
        raise ValueError("Both dimensions must be divisible by 16")

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(12345)
    source = torch.randn((in_features, out_features), generator=generator, dtype=torch.float32).cpu()
    left = torch.randn_like(source, generator=generator) * 0.01
    right = torch.randn_like(source, generator=generator) * 0.01
    output_feedback = torch.tril(
        torch.randn((out_features, out_features), generator=generator, dtype=torch.float32) * 0.01,
        diagonal=-1,
    )
    bias = torch.randn_like(source, generator=generator) * 0.001

    input_blocks = in_features // tile
    output_blocks = out_features // tile

    def _native():
        corrected = []
        for first_input in range(input_blocks):
            for first_output in range(output_blocks):
                if first_input + first_output != input_blocks - 1:
                    continue
                count = min(output_blocks - first_output, input_blocks - first_input)
                corrected.append(
                    qvq_cpu_yaqa_feedback(
                        source, left, right, output_feedback, first_input, first_output, count, bias
                    )
                )
        return torch.cat(corrected)

    def _python():
        corrected = []
        for first_input in range(input_blocks):
            for first_output in range(output_blocks):
                if first_input + first_output != input_blocks - 1:
                    continue
                count = min(output_blocks - first_output, input_blocks - first_input)
                corrected.append(
                    _python_feedback(source, left, right, output_feedback, first_input, first_output, count, bias)
                )
        return torch.cat(corrected)

    native_times, native_result = _timings(_native, warmup=args.warmup, iterations=args.iterations)
    python_times, python_result = _timings(_python, warmup=args.warmup, iterations=args.iterations)

    torch.testing.assert_close(native_result, python_result, rtol=0.0, atol=1e-6)

    native_med = 1000 * statistics.median(native_times)
    python_med = 1000 * statistics.median(python_times)
    speedup = python_med / native_med if native_med > 0 else float("inf")
    print(f"in={in_features} out={out_features}")
    print(f"  native median: {native_med:.3f} ms")
    print(f"  python median: {python_med:.3f} ms")
    print(f"  speedup: {speedup:.2f}x")
    print(f"  max abs diff: {(native_result - python_result).abs().max().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
