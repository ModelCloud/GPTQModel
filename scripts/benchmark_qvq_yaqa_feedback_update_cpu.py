#!/usr/bin/env python3
"""Benchmark the QVQ CPU fused factored YAQA cache update against the Python reference."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.utils.qvq_cpu import qvq_cpu_yaqa_feedback_update  # noqa: E402

TILE = 16


def _python_update(left, right, input_feedback, output_feedback, reconstructed, first_input, first_output, count):
    in_features = left.shape[0]
    out_features = left.shape[1]
    input_blocks = in_features // TILE
    output_blocks = out_features // TILE
    input_indices = torch.arange(first_input, first_input + count)
    output_indices = torch.arange(first_output, first_output - count, -1)

    left_factors = input_feedback.view(input_blocks, TILE, in_features)[input_indices].transpose(1, 2)
    updates = torch.bmm(left_factors, reconstructed)
    left_columns = left.view(in_features, output_blocks, TILE).permute(1, 0, 2)
    left_columns[output_indices] -= updates

    right_factors = output_feedback.view(output_blocks, TILE, out_features)[output_indices]
    right_updates = torch.bmm(reconstructed, right_factors)
    right_rows = right.view(input_blocks, TILE, out_features)
    right_rows[input_indices] -= right_updates


def _anti_diagonals(input_blocks, output_blocks):
    for first_input in range(input_blocks):
        for first_output in range(output_blocks):
            if first_input + first_output != input_blocks - 1:
                continue
            yield first_input, first_output, min(output_blocks - first_output, input_blocks - first_input)


def _timings(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    in_features = args.in_features
    out_features = args.out_features
    if in_features % TILE or out_features % TILE:
        raise ValueError("Both dimensions must be divisible by 16")
    input_blocks = in_features // TILE
    output_blocks = out_features // TILE

    generator = torch.Generator().manual_seed(1234)
    left0 = torch.randn((in_features, out_features), generator=generator, dtype=torch.float32)
    right0 = torch.randn((in_features, out_features), generator=generator, dtype=torch.float32)
    input_feedback = torch.randn((in_features, in_features), generator=generator, dtype=torch.float32) * 0.01
    output_feedback = torch.tril(
        torch.randn((out_features, out_features), generator=generator, dtype=torch.float32) * 0.01,
        diagonal=-1,
    ).contiguous()
    diagonals = list(_anti_diagonals(input_blocks, output_blocks))
    reconstructions = [
        (torch.randn((count, TILE, TILE), generator=generator, dtype=torch.float32) * 0.05).contiguous()
        for _, _, count in diagonals
    ]

    def _native():
        left = left0.clone()
        right = right0.clone()
        for (first_input, first_output, count), reconstructed in zip(diagonals, reconstructions):
            qvq_cpu_yaqa_feedback_update(
                left, right, input_feedback, output_feedback, reconstructed, first_input, first_output, count
            )
        return left, right

    def _python():
        left = left0.clone()
        right = right0.clone()
        for (first_input, first_output, count), reconstructed in zip(diagonals, reconstructions):
            _python_update(
                left, right, input_feedback, output_feedback, reconstructed, first_input, first_output, count
            )
        return left, right

    native_left, native_right = _native()
    python_left, python_right = _python()
    left_diff = (native_left - python_left).abs().max().item()
    right_diff = (native_right - python_right).abs().max().item()

    reference_left = left0.double()
    reference_right = right0.double()
    for (first_input, first_output, count), reconstructed in zip(diagonals, reconstructions):
        _python_update(
            reference_left,
            reference_right,
            input_feedback.double(),
            output_feedback.double(),
            reconstructed.double(),
            first_input,
            first_output,
            count,
        )
    scale_left = reference_left.abs().max().item()
    scale_right = reference_right.abs().max().item()
    native_ref_left = (native_left.double() - reference_left).abs().max().item()
    native_ref_right = (native_right.double() - reference_right).abs().max().item()
    python_ref_left = (python_left.double() - reference_left).abs().max().item()
    python_ref_right = (python_right.double() - reference_right).abs().max().item()

    native_times = _timings(_native, args.warmup, args.iterations)
    python_times = _timings(_python, args.warmup, args.iterations)
    native_med = 1000 * statistics.median(native_times)
    python_med = 1000 * statistics.median(python_times)

    print(f"in={in_features} out={out_features} diagonals={len(diagonals)}")
    print(f"  native median: {native_med:.3f} ms")
    print(f"  python median: {python_med:.3f} ms")
    print(f"  speedup: {python_med / native_med if native_med > 0 else float('inf'):.2f}x")
    print(f"  native vs python max abs diff left: {left_diff:.3e}  right: {right_diff:.3e}")
    print(f"  fp64 reference scale left: {scale_left:.3e}  right: {scale_right:.3e}")
    print(f"  native vs fp64 rel err left: {native_ref_left / scale_left:.3e}  right: {native_ref_right / scale_right:.3e}")
    print(f"  python vs fp64 rel err left: {python_ref_left / scale_left:.3e}  right: {python_ref_right / scale_right:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
