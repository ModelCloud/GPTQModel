#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare the bundled TT-XLA SFPU LM head to its matched BF16 schedule."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_nf12_sfpu_decode import _pack_four, _to_tiles, _weight
from bench_nf12_tenstorrent import _errors, _runtime, _same

from gptqmodel.utils.nf12_tenstorrent import round_up
from gptqmodel.utils.nf12_ttxla import _install_resolver
from gptqmodel.utils.nf12_ttxla_sfpu import _launch, _pack_xla_planes


def _operands(weight, base, device):
    bits = weight.view(torch.int16).int() & 0xffff
    tiles = _to_tiles(bits).transpose(-2, -1).contiguous()
    planes, patches = _pack_four(tiles, base)
    dense = tiles.reshape(-1, 32).to(torch.int16).view(torch.bfloat16).to(device)
    packed, metadata = _pack_xla_planes(planes.to(torch.int16), patches, base)
    return packed.to(device), dense, metadata.to(device)


def _measure_pair(calls, sync, repeats):
    for call in calls:
        sync(call())
    elapsed = [[], []]
    outputs = [None, None]
    for repeat in range(repeats):
        for index in (repeat % 2, 1 - repeat % 2):
            outputs[index] = None
            sync()
            start = time.perf_counter_ns()
            outputs[index] = calls[index]()
            sync(outputs[index])
            elapsed[index].append((time.perf_counter_ns() - start) / 1000)
    return outputs, tuple((statistics.median(times), sorted(times)[math.ceil(0.95 * repeats) - 1])
                          for times in elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rows", type=int, default=1, help="Logical activation rows, padded to 32 on device")
    parser.add_argument("--prefetch-groups", type=int, default=1)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32 or (args.checkpoint is None and args.size % 128):
        parser.error("Use a positive multiple of 128 for synthetic weights or 32 for checkpoint weights")
    if min(args.prefetch_groups, args.repeats, args.rows) < 1:
        parser.error("Prefetch groups, repeats and rows must be positive")

    torch.set_num_threads(3)
    base, weight = _weight(args.size, args.checkpoint, False)
    n, k = weight.shape
    if k % (128 * args.prefetch_groups):
        parser.error("Prefetch groups must divide the number of K=128 groups")
    if args.checkpoint is None:
        weight[0, 0] = 1.0  # Finite escapes straddle the first and last four-tile blocks.
        weight[-1, -1] = -2.0
    with torch.inference_mode(), _runtime("ttxla", args.chip) as (_, device, copy, sync):
        _install_resolver()
        x = torch.randn((args.rows, k), generator=torch.Generator().manual_seed(54)).bfloat16()
        activation = F.pad(x, (0, 0, 0, round_up(args.rows, 32) - args.rows)).to(device)
        measurements = []
        for exponent, weights in ((base, weight),) if args.checkpoint else ((base, weight), (base - 1, weight / 2)):
            packed, dense, metadata = _operands(weights, exponent, device)
            def launch(w, groups):
                return _launch(activation, w, metadata, n, exponent, groups)[:args.rows]

            calls = (partial(launch, dense, 1), partial(launch, packed, args.prefetch_groups))
            outputs, times = _measure_pair(calls, sync, args.repeats)
            dense_cpu, packed_cpu = (copy(out) for out in outputs)
            _same(dense_cpu, packed_cpu)
            oracle = _errors(packed_cpu, F.linear(x.float(), weights.float()).bfloat16())
            assert oracle["scaled_rmse"] <= 0.002, oracle
            kt = k // 32
            shared_l1 = 2048 * (2 * kt + 2)
            for label, weight_bytes, l1_bytes, (median, p95) in zip(
                ("matched_bf16_linear", "sfpu_linear"),
                (weights.nbytes, packed.nbytes + round_up(metadata.shape[0], 32) * 128),
                (shared_l1, shared_l1 + 2048 * (1 + min(args.prefetch_groups * 6, kt // 4 * 3)) + 128),
                times,
                strict=True,
            ):
                measurements.append({"kind": label, "shape": [n, k], "exp_base": exponent, "m": args.rows,
                                     "weight_bytes": weight_bytes, "l1_bytes_per_worker": l1_bytes,
                                     "median_us": median, "p95_us": p95, "effective_GBps": weight_bytes / median / 1000,
                                     "torch_fp32_oracle": oracle})
        for record in measurements:
            print(json.dumps({**record, "matched_linear_exact": True}), flush=True)


if __name__ == "__main__":
    main()
