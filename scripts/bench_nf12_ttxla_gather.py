#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Interleave compact and ordinary NF12 with the resident TT-XLA BF16 gather."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from bench_nf12_tenstorrent import _gather_oracle, _runtime, _same
from safetensors import safe_open

from gptqmodel.utils.nf12_tenstorrent import prepare_nf12


def _verify_compact(layer, weight):
    n, k = weight.shape
    for index in (0, 15, 16, 31, 32, n - 1, n, -1, 2**32 + 15):
        ids = torch.tensor([index])
        resident = layer.upload_ids(ids)
        expected = _gather_oracle(weight, ids)
        _same(layer.gather_compact_device(resident, direct=True).cpu().reshape(1, k), expected)
        # Same padded input shape, with distinct executable output layouts.
        _same(layer.gather_device(resident, 1, direct=True).cpu(), expected)


def _measure(calls, repeats):
    for _, call in calls:
        call()
    samples = [[] for _ in calls]
    for repeat in range(repeats):
        for step in range(len(calls)):
            index = (repeat + step) % len(calls)
            start = time.perf_counter_ns()
            result = calls[index][1]()
            samples[index].append((time.perf_counter_ns() - start) / 1000)
            del result
    return tuple((statistics.median(values), sorted(values)[math.ceil(0.95 * repeats) - 1]) for values in samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("The sample count must be positive")

    torch.set_num_threads(3)
    generator = torch.Generator().manual_seed(9)
    name = "model.llm.embed.weight"
    if args.checkpoint is None:
        weight = torch.randn((129, 256), generator=generator).bfloat16()
        weight.view(torch.int16)[32, 124:132] = torch.tensor(
            [0, -32768, 1, -32767, 32640, -128, 32641, -63], dtype=torch.int16,
        )
    else:
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(args.checkpoint / index[name], framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(name)
    n, k = weight.shape
    cases = {
        "one": torch.tensor([n - 1]),
        "repeat": torch.full((32,), 16),
        "contiguous": torch.arange(32),
        "scattered": torch.cat((torch.tensor([15, 16, 31, 32, n - 1, 0, 16]), torch.randperm(n, generator=generator)[:25])),
    }
    with torch.inference_mode(), _runtime("ttxla", args.chip) as (make, device, _, _):
        layer = make(device, *prepare_nf12(weight, name), row_major=True)
        _verify_compact(layer, weight)
        dense = weight.to(device)
        for case, ids in cases.items():
            resident = layer.upload_ids(ids)
            dense_ids = ids.to(torch.int32).to(device)
            count = len(ids)
            calls = [
                ("stock_bf16", lambda: dense.index_select(0, dense_ids).cpu()),
                ("row_nf12", lambda: layer.gather_device(resident, count, direct=False).cpu()),
                ("direct_row_nf12", lambda: layer.gather_device(resident, count, direct=True).cpu()),
            ]
            if count == 1:
                calls.append(("direct_compact_nf12", lambda: layer.gather_compact_device(resident, direct=True).cpu().reshape(1, k)))
            expected = _gather_oracle(weight, ids)
            stock = calls[0][1]()
            if args.checkpoint is None:
                # TT-XLA's stock gather can change BF16 special values. The
                # NF12 implementations below still check every raw bit.
                finite_normal = torch.isfinite(expected) & (expected.abs() >= torch.finfo(torch.bfloat16).tiny)
                torch.testing.assert_close(stock[finite_normal], expected[finite_normal], atol=0, rtol=0)
            else:
                torch.testing.assert_close(stock, expected, atol=0, rtol=0, equal_nan=True)
            for _, call in calls[1:]:
                _same(call(), expected)
            for (kind, _), (median, p95) in zip(calls, _measure(calls, args.repeats), strict=True):
                print(json.dumps({"case": case, "kind": kind, "ids": count, "median_us": median, "p95_us": p95,
                                  "resident_bytes": weight.nbytes if kind == "stock_bf16" else layer.resident_bytes}), flush=True)
        if args.exhaustive:
            for start in range(0, n, 2048):
                end = min(start + 2048, n)
                resident = layer.upload_ids(torch.arange(start, end))
                _same(layer.gather_device(resident, end - start, direct=True).cpu(), weight[start:end])
            print(json.dumps({"full_row_check": True, "rows": n}), flush=True)


if __name__ == "__main__":
    main()
