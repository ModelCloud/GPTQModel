# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark resident 2-bit tile searches; set PYTHONPATH to the checkout to test."""

import argparse
import hashlib
import json
import statistics

import torch

from gptqmodel.exllamav3.ext import exllamav3_ext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tiles", type=int, nargs="+", default=[1, 32, 128])
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "device": props.name,
                "torch": torch.__version__,
                "optin_shared_bytes": props.shared_memory_per_block_optin,
            }
        ),
        flush=True,
    )
    for codebook in ("3inst", "mcg", "mul1"):
        for count in args.tiles:
            slots = min(count, props.multi_processor_count)
            source = torch.randn((count, 256), generator=torch.Generator().manual_seed(3528)).cuda()
            output = torch.empty_like(source)
            indices = torch.empty_like(source, dtype=torch.int16)
            costs = torch.empty((slots, 2, 16384), device="cuda", dtype=torch.float16)
            history = torch.empty((slots, 256, 16384), device="cuda", dtype=torch.int16)

            def run():
                exllamav3_ext.quantize_tiles(
                    source, output, indices, costs, history, 2, codebook == "mcg", codebook == "mul1"
                )

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            print(
                json.dumps(
                    {
                        "codebook": codebook,
                        "tiles": count,
                        "slots": slots,
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                        "scratch_bytes": costs.nbytes + history.nbytes,
                        "output_sha256": hashlib.sha256(
                            output.cpu().numpy().tobytes() + indices.cpu().numpy().tobytes()
                        ).hexdigest(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
