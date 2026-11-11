# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired released-checkpoint embedding lookup at Lite B1 prefill/decode rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12
from scripts.bench_nf12_lite_observed_shapes import capture, graph_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if args.repeat < 1 or args.rounds < 1:
        parser.error("repeat and rounds must be positive")
    if sys._is_gil_enabled() or torch.cuda.device_count() != 1:
        parser.error("use PYTHON_GIL=0 with exactly one visible CUDA GPU")
    if torch.cuda.get_device_capability(0)[0] < 8:
        parser.error("SM80 or newer is required")
    nf12.load()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "gil_enabled": sys._is_gil_enabled(),
        "checkpoint": str(args.checkpoint),
        "method": "same IDs and released BF16 weight; CUDA graph; paired ABBA rounds",
        "repeat": args.repeat,
        "rounds": args.rounds,
        "families": [],
    }
    for name, key, rows in (
        ("embedding", "embed.weight", (1, 6, 512, 1536, 3072)),
        ("markov-embedding", "mtp.2.markov_head.embed.weight", (1, 5)),
    ):
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as shard:
            source = shard.get_tensor(key)
        assert source.dtype == torch.bfloat16
        sha256 = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
        weight = source.to("cuda:0")
        packed = nf12.encode(weight)
        assert torch.equal(packed.decode().view(torch.int16), weight.view(torch.int16))
        assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
        family = {
            "name": name, "weight_key": key, "weight_sha256": sha256,
            "shape_nk": list(weight.shape),
            "dense_weight_bytes": weight.nbytes, "packed_weight_bytes": packed.nbytes,
            "cases": [],
        }
        for m in rows:
            generator = torch.Generator(device="cuda:0").manual_seed(30911 + m)
            ids = torch.randint(weight.shape[0], (m,), device="cuda:0", dtype=torch.int32, generator=generator)
            dense_fn = partial(F.embedding, ids, weight)
            nf12_fn = partial(nf12.embedding, ids, packed)
            expected, actual = dense_fn(), nf12_fn()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            dense_graph, nf12_graph = capture(dense_fn, args.repeat), capture(nf12_fn, args.repeat)
            for _ in range(3):
                dense_graph.replay()
                nf12_graph.replay()
            dense_times, nf12_times = [], []
            for _ in range(args.rounds):
                dense_times.append(graph_us(dense_graph, args.repeat))
                nf12_times.append(graph_us(nf12_graph, args.repeat))
                nf12_times.append(graph_us(nf12_graph, args.repeat))
                dense_times.append(graph_us(dense_graph, args.repeat))
            dense_us, nf12_us = statistics.median(dense_times), statistics.median(nf12_times)
            row = {
                "m": m, "dense_us": round(dense_us, 3), "nf12_us": round(nf12_us, 3),
                "speedup": round(dense_us / nf12_us, 4), "exact_bf16_bits": True,
            }
            family["cases"].append(row)
            print(name, row, flush=True)
            del dense_graph, nf12_graph, expected, actual
        report["families"].append(family)
        del weight, packed, source
        torch.cuda.empty_cache()
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
