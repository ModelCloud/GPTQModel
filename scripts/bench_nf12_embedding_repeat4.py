# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired released-weight lookup plus four-way activation repeat on one SM80 GPU.

CUDA_VISIBLE_DEVICES=<physical GPU> PYTHON_GIL=0 PYTHONPATH=. python \
    scripts/bench_nf12_embedding_repeat4.py --json-out /tmp/nf12-repeat4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12
from scripts.bench_nf12_lite_observed_shapes import capture, graph_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--ids-json", type=Path, default=Path("docs/nf12-lite-cuda-token-ids-20261111.json"))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    if sys._is_gil_enabled() or torch.cuda.device_count() != 1:
        parser.error("requires PYTHON_GIL=0 and one visible GPU")
    torch.cuda.set_per_process_memory_fraction(0.04)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / index["embed.weight"], framework="pt", device="cpu") as shard:
        source = shard.get_tensor("embed.weight")
    weight_sha = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
    weight = source.to("cuda:0")
    packed = nf12.encode(weight)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    ids_all = json.loads(args.ids_json.read_text())
    report = {"gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
              "weight_sha256": weight_sha, "shape_nk": list(weight.shape),
              "dense_weight_bytes": weight.nbytes, "packed_weight_bytes": packed.nbytes,
              "inference_retained_expanded_weight_bytes": 0,
              "benchmark_dense_oracle_bytes": weight.nbytes,
              "repeat": args.repeat, "rounds": args.rounds,
              "method": "same released weight/IDs; CUDA graphs; alternating paired ABBA rounds",
              "cases": []}
    for kind in ("code", "random"):
        for m in (1, 6, 312, 313, 1536, 3072):
            if kind == "code":
                ids = torch.tensor(ids_all[:m], device="cuda:0", dtype=torch.int64)
            else:
                generator = torch.Generator(device="cuda:0").manual_seed(20261111 + m)
                ids = torch.randint(weight.shape[0], (m,), device="cuda:0", dtype=torch.int64,
                                    generator=generator)
            expected = F.embedding(ids, weight).unsqueeze(-2).repeat(1, 4, 1)
            actual = nf12.embedding_repeat4(ids, packed)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            fns = {
                "dense_repeat": lambda: F.embedding(ids, weight).unsqueeze(-2).repeat(1, 4, 1),
                "nf12_repeat": lambda: nf12.embedding(ids, packed).unsqueeze(-2).repeat(1, 4, 1),
                "nf12_fused": lambda: nf12.embedding_repeat4(ids, packed),
            }
            graphs = {name: capture(fn, args.repeat) for name, fn in fns.items()}
            samples = {name: [] for name in fns}
            for round_index in range(args.rounds):
                order = ("dense_repeat", "nf12_repeat", "nf12_fused", "nf12_fused", "nf12_repeat", "dense_repeat")
                if round_index % 2:
                    order = tuple(reversed(order))
                for name in order:
                    samples[name].append(graph_us(graphs[name], args.repeat))
            med = {name: statistics.median(values) for name, values in samples.items()}
            row = {"kind": kind, "m": m, "id_dtype": "int64", "exact_bf16_bits": True,
                   "samples_us": samples, "median_us": med,
                   "fused_over_dense": med["dense_repeat"] / med["nf12_fused"],
                   "fused_over_old_nf12": med["nf12_repeat"] / med["nf12_fused"]}
            report["cases"].append(row)
            print(kind, m, med, flush=True)
    report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
