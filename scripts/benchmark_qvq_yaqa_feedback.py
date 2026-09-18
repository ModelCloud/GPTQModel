#!/usr/bin/env python3
"""Benchmark exact and opt-in TF32 YAQA factored-feedback CUDA paths."""

from __future__ import annotations

import argparse
import os

import torch

from gptqmodel.utils.qvq_cuda import _qvq_cuda_yaqa_feedback_op


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=8192)
    parser.add_argument("--families", type=int, default=2)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if min(args.in_features, args.out_features, args.families, args.count, args.iters) < 1:
        parser.error("all dimensions and iteration counts must be positive")
    first_output_block = args.count - 1
    if args.in_features < args.count * 16 or args.out_features < args.count * 16:
        parser.error("input and output dimensions must cover count 16x16 tiles")

    generator = torch.Generator(device="cuda").manual_seed(20260921)
    source = torch.randn(
        (args.in_features, args.out_features), generator=generator, device="cuda"
    ) * 0.05
    left = torch.randn(
        (args.families, args.in_features, args.out_features),
        generator=generator,
        device="cuda",
    ) * 0.01
    right = torch.randn_like(left, generator=generator) * 0.01
    output_feedback = torch.randn(
        (args.out_features, args.out_features), generator=generator, device="cuda"
    ) * 0.01
    op = _qvq_cuda_yaqa_feedback_op()
    outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, float] = {}
    for label, enabled in (("fp32", False), ("tf32", True)):
        os.environ["GPTQMODEL_QVQ_YAQA_FAST_TF32"] = "1" if enabled else "0"
        for _ in range(4):
            outputs[label] = op(
                source, left, right, output_feedback, 0, first_output_block, args.count, None
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            outputs[label] = op(
                source, left, right, output_feedback, 0, first_output_block, args.count, None
            )
        end.record()
        end.synchronize()
        timings[label] = start.elapsed_time(end) / args.iters
    os.environ.pop("GPTQMODEL_QVQ_YAQA_FAST_TF32", None)

    maximum_delta = (outputs["tf32"] - outputs["fp32"]).abs().amax().item()
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"fp32={timings['fp32']:.6f} ms")
    print(f"tf32={timings['tf32']:.6f} ms")
    print(f"speedup={timings['fp32'] / timings['tf32']:.4f}x")
    print(f"maximum_absolute_delta={maximum_delta:.9g}")


if __name__ == "__main__":
    main()
