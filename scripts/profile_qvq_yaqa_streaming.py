#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Profile the compact YAQA Sketch-B contraction without model-load noise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_idle_preflight import bootstrap_gpu_idle_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=5120)
    parser.add_argument("--out-features", type=int, default=17408)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        raise ValueError("all profiling dimensions and iteration counts must be positive")

    preflight = bootstrap_gpu_idle_preflight()
    if preflight is None:
        raise RuntimeError("formal YAQA profiling requires the GPU idle preflight")

    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA H200":
        raise RuntimeError("this profile requires exactly one visible NVIDIA H200")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260905)
    activation = torch.randn(
        (args.batch_size, args.sequence_length, args.in_features),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    gradient = torch.randn(
        (args.batch_size, args.sequence_length, args.out_features),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    def run(index: int):
        projection_generator = torch.Generator(device=device).manual_seed(20260905 + index)
        projection = torch.empty(
            (args.batch_size, args.out_features + args.in_features, args.rank),
            dtype=torch.float32,
            device=device,
        ).normal_(generator=projection_generator)
        output_projection = projection[:, : args.out_features]
        input_projection = projection[:, args.out_features :]
        if args.batch_size == 1:
            input_source = activation[0].T @ (gradient[0] @ output_projection[0])
            output_source = gradient[0].T @ (activation[0] @ input_projection[0])
        else:
            input_source = torch.bmm(
                activation.transpose(1, 2),
                torch.bmm(gradient, output_projection),
            ).sum(dim=0)
            output_source = torch.bmm(
                gradient.transpose(1, 2),
                torch.bmm(activation, input_projection),
            ).sum(dim=0)
        return (
            input_source,
            output_source,
        )

    for index in range(args.warmup):
        run(index)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.cuda.nvtx.range("yaqa.streaming_projected"):
        outputs = [run(args.warmup + index) for index in range(args.iterations)]
    end.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    checksum = sum(float(item.square().mean().item()) for output in outputs for item in output)
    print(
        {
            "milliseconds": start.elapsed_time(end),
            "milliseconds_per_iteration": start.elapsed_time(end) / args.iterations,
            "checksum": checksum,
            "h200_uuid": preflight.uuid,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
