# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paired full-weight row-major repeat4 versus the tuned fused HC4 public op.

CUDA_VISIBLE_DEVICES=1 PYTHON_GIL=0 PYTHONPATH=/root/nf12-next-gpu1 \
  /root/venv-lite-torch215/bin/python scripts/bench_nf12_rowmajor_repeat4_sm80.py \
  --physical-gpu 1 --variants coalesced_store --rounds 16 \
  --json-out docs/nf12-rowmajor-repeat4-gpu1-20261111.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/root/nf12-fused-embed-gpu7")
# Import the actual tuned public op from the other checkout, then extend only
# package search with this worktree's explicitly experimental helper. No source
# file or package state in the upstream checkout is modified.
sys.path.insert(0, str(UPSTREAM))
import gptqmodel_nf12 as nf12  # noqa: E402


assert Path(nf12.__file__).resolve().is_relative_to(UPSTREAM)
nf12.__path__.append(str(ROOT / "gptqmodel_nf12"))
from gptqmodel_nf12.experimental_rowmajor import (  # noqa: E402
    NF12RowMajorWeight,
    from_nf12,
)
from gptqmodel_nf12.experimental_rowmajor import (
    embedding_repeat4 as row_repeat4,
)


SIZES = (312, 313, 1536, 2048, 3072)


def configurations(m: int) -> dict[str, dict]:
    # The tuned upstream geometry's row-major analogue, one prior row-major
    # candidate, and one focused coalesced-store test.
    if m <= 512:
        return {
            "fused_analogue": {"threads": 128, "ids_per_cta": 8, "store_policy": "cg"},
            "row_vector": {"threads": 256, "ids_per_cta": 8, "store_policy": "cg"},
            "coalesced_store": {"threads": 128, "ids_per_cta": 8, "store_policy": "transpose_cg"},
        }
    if m <= 2048:
        return {
            "fused_analogue": {"threads": 128, "ids_per_cta": 1, "store_policy": "cs"},
            "row_vector": {"threads": 256, "ids_per_cta": 8, "store_policy": "cg"},
            "coalesced_store": {"threads": 128, "ids_per_cta": 1, "store_policy": "transpose_cs"},
        }
    return {
        "fused_analogue": {"threads": 64, "ids_per_cta": 1, "store_policy": "default"},
        "row_vector": {"threads": 128, "ids_per_cta": 2, "store_policy": "cg"},
        "coalesced_store": {"threads": 64, "ids_per_cta": 1, "store_policy": "transpose_cg"},
    }


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


def paired(fn_a, fn_b, repeat: int, rounds: int) -> dict:
    a, b = capture(fn_a, repeat), capture(fn_b, repeat)
    for _ in range(3):
        a.replay()
        b.replay()
    samples = {"tuned_fused": [], "row_major_fused": []}
    for _ in range(rounds):
        for name, graph in (("tuned_fused", a), ("row_major_fused", b),
                            ("row_major_fused", b), ("tuned_fused", a)):
            samples[name].append(graph_us(graph, repeat))
    del a, b
    return samples


def exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
    # Compare bounded row chunks so validation does not create a full output
    # sized temporary on top of the two resident packed layouts.
    for start in range(0, actual.shape[0], 128):
        assert torch.equal(actual[start:start + 128].view(torch.int16),
                           expected[start:start + 128].view(torch.int16))


def case_ids(kind: str, m: int, n: int, fixture: list[int]) -> torch.Tensor:
    if kind == "code":
        return torch.tensor(fixture[:m], dtype=torch.int64, device="cuda:0")
    gen = torch.Generator(device="cuda:0").manual_seed(20261111 + m)
    return torch.randint(n, (m,), dtype=torch.int64, device="cuda:0", generator=gen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--ids-json", type=Path,
                        default=ROOT / "docs/nf12-lite-cuda-token-ids-20261111.json")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--variants", type=str,
                        default="fused_analogue,row_vector,coalesced_store")
    args = parser.parse_args()
    if args.repeat < 1 or args.rounds < 1:
        parser.error("repeat and rounds must be positive")
    variants = set(args.variants.split(","))
    if not variants or not variants <= {"fused_analogue", "row_vector", "coalesced_store"}:
        parser.error("unknown row-major repeat4 variant")
    if (args.physical_gpu < 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu)
            or sys._is_gil_enabled()
            or torch.version.hip is not None or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability(0) != (8, 0)):
        parser.error("requires only the selected physical SM80 GPU visible and PYTHON_GIL=0")
    # Leave room for CUDA context above the allocator; both packed layouts and
    # graph output buffers must fit below three GiB added VRAM.
    torch.cuda.set_per_process_memory_fraction(0.025, 0)
    torch.cuda.reset_peak_memory_stats()
    nf12.load()
    source_bytes = args.ids_json.read_bytes()
    fixture = json.loads(source_bytes)
    assert len(fixture) >= max(SIZES)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / index["embed.weight"], framework="pt", device="cpu") as shard:
        source_cpu = shard.get_tensor("embed.weight")
    assert source_cpu.dtype == torch.bfloat16 and tuple(source_cpu.shape) == (129280, 5120)
    weight_sha = hashlib.sha256(source_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    dense = source_cpu.to("cuda:0")
    canonical = nf12.encode(dense)
    del dense
    torch.cuda.empty_cache()
    row = from_nf12(canonical, backend="cuda")
    assert isinstance(row, NF12RowMajorWeight)
    assert canonical.nbytes == row.nbytes == 995_314_416
    n, k = canonical.shape

    invalid_values = [-2**40, -1, 0, n - 1, n, 2**40]
    invalid = torch.tensor(invalid_values, dtype=torch.int64, device="cuda:0")
    invalid_single = torch.zeros((len(invalid_values), k), dtype=torch.bfloat16, device="cuda:0")
    invalid_single[2] = source_cpu[0].to("cuda:0")
    invalid_single[3] = source_cpu[n - 1].to("cuda:0")
    invalid_oracle = invalid_single.unsqueeze(1).repeat(1, 4, 1)
    exact(nf12.embedding_repeat4(invalid, canonical), invalid_oracle)
    for config in (value for m in SIZES for name, value in configurations(m).items()
                   if name in variants):
        exact(row_repeat4(invalid, row, **config), invalid_oracle)
    del invalid, invalid_single, invalid_oracle
    torch.cuda.empty_cache()

    upstream_head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "gpu": torch.cuda.get_device_name(0), "visible_physical_gpu": args.physical_gpu,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "gil_enabled": sys._is_gil_enabled(),
        "upstream_fused_checkout": str(UPSTREAM), "upstream_fused_commit": upstream_head,
        "upstream_fused_source_sha256": hashlib.sha256((UPSTREAM / "gptqmodel_ext/nf12/csrc/codec/embedding.cu").read_bytes()).hexdigest(),
        "rowmajor_source_sha256": hashlib.sha256((ROOT / "gptqmodel_ext/nf12/experiments/rowmajor_embedding.cu").read_bytes()).hexdigest(),
        "weight_sha256": weight_sha, "ids_fixture_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "shape": [n, k], "repeat": args.repeat, "rounds": args.rounds,
        "selected_variants": sorted(variants),
        "method": "same-process full released weight; actual tuned public embedding_repeat4; "
                  "bounded row-major configurations; CUDA graphs; paired ABBA; "
                  "int64 code prefix and CUDA random seed 20261111+M",
        "storage": {"canonical_bytes": canonical.nbytes, "rowmajor_bytes": row.nbytes,
                    "rowmajor_collision_records": row.selected_half_collisions,
                    "persistent_expanded_weight_bytes": 0},
        "invalid_ids_exact_bf16_bits": True,
        "cases": [],
    }
    for kind in ("code", "random"):
        for m in SIZES:
            ids = case_ids(kind, m, n, fixture)
            ids_cpu = ids.cpu()
            expected_single = source_cpu[ids_cpu.long()].to("cuda:0")
            expected = expected_single.unsqueeze(1).repeat(1, 4, 1)
            exact(nf12.embedding_repeat4(ids, canonical), expected)
            case = {"kind": kind, "m": m, "id_dtype": "int64",
                    "output_bytes": m * 4 * k * 2, "exact_bf16_bits": True,
                    "variants": []}
            for name, config in configurations(m).items():
                if name not in variants:
                    continue
                exact(row_repeat4(ids, row, **config), expected)
                samples = paired(lambda ids=ids: nf12.embedding_repeat4(ids, canonical),
                                 lambda ids=ids, config=config: row_repeat4(ids, row, **config),
                                 args.repeat, args.rounds)
                medians = {key: statistics.median(value) for key, value in samples.items()}
                paired_delta = [statistics.median(samples["tuned_fused"][2*r:2*r+2]) -
                                statistics.median(samples["row_major_fused"][2*r:2*r+2])
                                for r in range(args.rounds)]
                case["variants"].append({"name": name, "geometry": config,
                                         "samples_us": samples, "median_us": medians,
                                         "median_paired_delta_us": statistics.median(paired_delta),
                                         "paired_deltas_us": paired_delta})
                print(kind, m, name, {key: round(value, 3) for key, value in medians.items()},
                      "delta", round(statistics.median(paired_delta), 3), flush=True)
                torch.cuda.empty_cache()
            report["cases"].append(case)
            del ids, ids_cpu, expected_single, expected
            torch.cuda.empty_cache()
    retained = {key: value for key, value in vars(row).items() if isinstance(value, torch.Tensor)}
    assert set(retained) == {"packed", "escapes"}
    report["retained_after_repeated_forwards"] = {key: value.nbytes for key, value in retained.items()}
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    assert report["peak_reserved_bytes"] < 3 * 1024**3
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
