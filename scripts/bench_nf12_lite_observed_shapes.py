# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired real-checkpoint NF12/BF16 projections at Lite's observed B1 row counts.

Run with PYTHON_GIL=0 and one visible SM80 GPU. The checkpoint BF16 weight is
an independent timing and numerical oracle, never retained by the NF12 path.
"""

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
from gptqmodel_nf12._jit import library_directory


CASES = (
    ("main-gate", "layers.20.ffn.gate.weight", "fp32", (1, 6, 312, 313, 1536, 3072)),
    ("compressor-kv-ratio1", "layers.20.attn.compressor.wkv.weight", "bf16", (1, 6, 312, 313, 1536, 3072)),
    ("compressor-kv-ratio2", "layers.2.attn.compressor.wkv.weight", "fp32", (1, 6, 312, 313, 1536, 3072)),
    ("indexer-score", "layers.20.attn.indexer.weights_proj.weight", "bf16", (1, 6, 312, 313, 1536, 3072)),
    ("indexer-key-ratio1", "layers.20.attn.indexer.wk.weight", "bf16", (1, 6, 312, 313, 1536, 3072)),
    ("indexer-key-ratio2", "layers.2.attn.indexer.wk.weight", "bf16", (1, 6, 312, 313, 768, 1536)),
)


def capture(fn, repeat: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    return graph


def graph_us(graph: torch.cuda.CUDAGraph, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeat


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    nf12.load()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "gil_enabled": sys._is_gil_enabled(),
        "native_source_key": library_directory().parent.name,
        "checkpoint": str(args.checkpoint),
        "method": "same checkpoint BF16 weight and activation dtype; CUDA graph; ABBA rounds; TF32 off",
        "repeat": args.repeat,
        "rounds": args.rounds,
        "families": [],
    }
    for name, key, dtype, rows in CASES:
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as shard:
            source = shard.get_tensor(key)
        assert source.dtype == torch.bfloat16 and source.ndim == 2
        sha256 = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
        weight = source.to("cuda:0")
        packed = nf12.encode(weight)
        assert torch.equal(packed.decode().view(torch.int16), weight.view(torch.int16))
        if name == "main-gate":
            nf12.prepare_fp32_gate(packed)
        family = {
            "name": name, "weight_key": key, "weight_sha256": sha256,
            "shape_nk": list(weight.shape), "activation_dtype": dtype,
            "dense_weight_bytes": weight.nbytes, "packed_weight_bytes": packed.nbytes,
            "cases": [],
        }
        n, k = weight.shape
        for m in rows:
            generator = torch.Generator(device="cuda:0").manual_seed(19017 + m)
            x = torch.randn((m, k), device="cuda:0", generator=generator) * 0.05
            if dtype == "bf16":
                x = x.bfloat16()
                dense_fn = partial(F.linear, x, weight)
                nf12_fn = partial(nf12.linear, x, packed)
            else:
                dense_weight = weight.float()
                dense_fn = partial(F.linear, x, dense_weight)
                if name == "main-gate" and m == 6:
                    nf12_fn = partial(nf12.linear_fp32_staged_b1, x, packed)
                else:
                    nf12_fn = partial(nf12.linear_fp32, x, packed)
            expected, actual = dense_fn(), nf12_fn()
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
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
                "m": m, "shape_nkm": [n, k, m],
                "dense_us": round(dense_us, 3), "nf12_us": round(nf12_us, 3),
                "speedup": round(dense_us / nf12_us, 4),
                "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                "top8_mismatch_rows": int((actual.topk(min(8, n), -1).indices !=
                                           expected.topk(min(8, n), -1).indices).any(-1).sum()),
            }
            family["cases"].append(row)
            print(name, row, flush=True)
            del dense_graph, nf12_graph, expected, actual, x
        report["families"].append(family)
        del weight, packed, source
        torch.cuda.empty_cache()
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
