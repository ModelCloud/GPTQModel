# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Full released-embedding load, canonical-to-row conversion, and code-ID lookup.

CUDA_VISIBLE_DEVICES=1 PYTHON_GIL=0 PYTHONPATH=/root/nf12-next-gpu1 \
  /root/venv-lite-torch215/bin/python scripts/bench_nf12_rowmajor_layout_gpu1.py \
  --json-out docs/nf12-rowmajor-layout-gpu1-20261111.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12
from gptqmodel_nf12.experimental_rowmajor import _native, embedding as row_embedding, from_nf12
from scripts.bench_nf12_lite_observed_shapes import capture, graph_us


def exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def paired(fn_a, fn_b, repeat: int, rounds: int) -> tuple[list[float], list[float]]:
    graph_a, graph_b = capture(fn_a, repeat), capture(fn_b, repeat)
    for _ in range(3):
        graph_a.replay()
        graph_b.replay()
    samples = ([], [])
    for _ in range(rounds):
        for selected in (0, 1, 1, 0):
            samples[selected].append(graph_us((graph_a, graph_b)[selected], repeat))
    del graph_a, graph_b
    return samples


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def geometry(m: int) -> dict:
    # The code-ID sweep on this GPU found these four-lane, L1-bypassing-store
    # geometries near the top for both serving ID dtypes.
    return {"threads": 256, "ids_per_cta": 8, "cache_global_stores": True} if m == 1536 else {
        "threads": 128, "ids_per_cta": 2, "cache_global_stores": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--ids-json", type=Path,
                        default=Path("docs/nf12-lite-cuda-token-ids-20261111.json"))
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.repeat < 1 or args.rounds < 1:
        parser.error("repeat and rounds must be positive")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or sys._is_gil_enabled()
            or torch.version.hip is not None or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability(0) != (8, 0)):
        parser.error("requires only physical GPU1, PYTHON_GIL=0, and NVIDIA SM80")
    # Hard allocator budget: 2.6% of this 96-GiB GPU is 2.68 GB.
    torch.cuda.set_per_process_memory_fraction(0.026, 0)
    torch.cuda.reset_peak_memory_stats()
    native_build_start = time.perf_counter()
    _native()
    native_build_seconds = time.perf_counter() - native_build_start
    fixture_bytes = args.ids_json.read_bytes()
    fixture = json.loads(fixture_bytes)
    assert len(fixture) >= 3072

    load_start = time.perf_counter()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / index["embed.weight"], framework="pt", device="cpu") as shard:
        source_cpu = shard.get_tensor("embed.weight")
    checkpoint_read_seconds = time.perf_counter() - load_start
    assert source_cpu.dtype == torch.bfloat16 and tuple(source_cpu.shape) == (129280, 5120)
    sha256 = hashlib.sha256(source_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    dense, dense_h2d_seconds = timed(lambda: source_cpu.to("cuda:0"))
    canonical, canonical_encode_seconds = timed(lambda: nf12.encode(dense))
    del dense
    torch.cuda.empty_cache()
    gpu_row, gpu_repack_seconds = timed(lambda: from_nf12(canonical, backend="cuda"))
    assert gpu_row.selected_half_collisions == 0
    assert gpu_row.nbytes == canonical.nbytes == 995_314_416
    canonical_cpu, canonical_d2h_seconds = timed(lambda: canonical.to("cpu"))
    cpu_pack_start = time.perf_counter()
    cpu_row = from_nf12(canonical_cpu, backend="cpu")
    cpu_repack_seconds = time.perf_counter() - cpu_pack_start
    assert cpu_row.nbytes == gpu_row.nbytes
    del canonical_cpu
    print("conversion", {"checkpoint_read_s": checkpoint_read_seconds,
                         "dense_h2d_s": dense_h2d_seconds,
                         "canonical_encode_s": canonical_encode_seconds,
                         "gpu_repack_s": gpu_repack_seconds,
                         "canonical_d2h_s": canonical_d2h_seconds,
                         "cpu_repack_s": cpu_repack_seconds}, flush=True)

    report = {
        "baseline_commit": "e3141059", "gpu": torch.cuda.get_device_name(0),
        "visible_physical_gpu": 1, "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "gil_enabled": sys._is_gil_enabled(),
        "shape": list(source_cpu.shape), "weight_sha256": sha256,
        "ids_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "method": "full released dense checkpoint; CUDA canonical encode; CPU and GPU row pack; "
                  "eight-call graph replay, six paired ABBA rounds; code-ID fixture prefix",
        "repeat": args.repeat, "rounds": args.rounds,
        "load_conversion_seconds": {
            "native_compile_one_time": native_build_seconds,
            "checkpoint_read_dense_cpu": checkpoint_read_seconds,
            "dense_cpu_to_gpu": dense_h2d_seconds,
            "canonical_nf12_encode_gpu": canonical_encode_seconds,
            "canonical_gpu_to_cpu": canonical_d2h_seconds,
            "row_pack_gpu": gpu_repack_seconds,
            "row_pack_cpu": cpu_repack_seconds,
        },
        "storage": {
            "dense_bf16_bytes": source_cpu.nbytes,
            "canonical_packed_bytes": canonical.packed.nbytes,
            "canonical_escapes_bytes": canonical.escapes.nbytes,
            "canonical_total_bytes": canonical.nbytes,
            "row_packed_bytes": gpu_row.packed.nbytes,
            "row_escapes_bytes": gpu_row.escapes.nbytes,
            "row_total_bytes": gpu_row.nbytes,
            "original_escape_records": gpu_row.original_escapes,
            "selected_half_collisions": gpu_row.selected_half_collisions,
            "persistent_expanded_weight_bytes": 0,
            "temporary_dequantized_weight_tensor_bytes": 0,
        },
        "cases": [],
    }
    n, k = source_cpu.shape
    for dtype in (torch.int64, torch.int32):
        for m in (1536, 3072):
            ids_cpu = torch.tensor(fixture[:m], dtype=dtype)
            ids = ids_cpu.to("cuda:0")
            selected = geometry(m)
            expected = source_cpu[ids_cpu.long()].to("cuda:0")
            exact(row_embedding(ids, gpu_row, **selected), expected)
            exact(nf12.embedding(ids, canonical), expected)
            base_samples, row_samples = paired(
                lambda: nf12.embedding(ids, canonical),
                lambda: row_embedding(ids, gpu_row, **selected), args.repeat, args.rounds)
            case = {"kind": "code", "id_dtype": str(dtype).split(".")[-1], "m": m,
                    "geometry": selected, "exact_bf16_bits": True,
                    "canonical_pair_samples_us": {"canonical": base_samples, "row": row_samples},
                    "canonical_us": statistics.median(base_samples),
                    "row_vs_canonical_us": statistics.median(row_samples)}
            report["cases"].append(case)
            print("canonical pair", case["id_dtype"], m,
                  round(case["canonical_us"], 3), round(case["row_vs_canonical_us"], 3), flush=True)
            del ids_cpu, ids, expected
            torch.cuda.empty_cache()
    report["peak_after_canonical_pair_reserved_bytes"] = torch.cuda.max_memory_reserved()

    # Compare both independently packed layouts without keeping a third full
    # representation on GPU. The released table has no half collisions, so its
    # packed bytes and split escape bytes must be identical across backends.
    del canonical
    torch.cuda.empty_cache()
    cpu_row_gpu, cpu_row_h2d_seconds = timed(lambda: cpu_row.to("cuda:0"))
    # torch.equal on a 993-MB tensor can materialize a full-size comparison
    # buffer. Keep this validation bounded under the GPU memory cap.
    for start in range(0, n, 256):
        assert torch.equal(cpu_row_gpu.packed[start:start + 256],
                           gpu_row.packed[start:start + 256])
    assert torch.equal(cpu_row_gpu.escapes, gpu_row.escapes)
    report["cpu_gpu_full_packed_bytes_identical"] = True
    for dtype in (torch.int64, torch.int32):
        for m in (1536, 3072):
            ids = torch.tensor(fixture[:m], dtype=dtype, device="cuda:0")
            selected = geometry(m)
            exact(row_embedding(ids, cpu_row_gpu, **selected), row_embedding(ids, gpu_row, **selected))
            del ids
    del cpu_row_gpu
    torch.cuda.empty_cache()
    report["load_conversion_seconds"]["row_cpu_to_gpu"] = cpu_row_h2d_seconds
    report["load_conversion_seconds"]["dense_checkpoint_to_gpu_row_via_gpu"] = (
        checkpoint_read_seconds + dense_h2d_seconds + canonical_encode_seconds + gpu_repack_seconds)
    report["load_conversion_seconds"]["dense_checkpoint_to_gpu_row_via_cpu"] = (
        checkpoint_read_seconds + dense_h2d_seconds + canonical_encode_seconds +
        canonical_d2h_seconds + cpu_repack_seconds + cpu_row_h2d_seconds)

    dense, dense_oracle_h2d_seconds = timed(lambda: source_cpu.to("cuda:0"))
    report["load_conversion_seconds"]["dense_oracle_h2d_for_benchmark"] = dense_oracle_h2d_seconds
    for case in report["cases"]:
        m = case["m"]
        dtype = torch.int64 if case["id_dtype"] == "int64" else torch.int32
        ids = torch.tensor(fixture[:m], dtype=dtype, device="cuda:0")
        selected = case["geometry"]
        expected = F.embedding(ids, dense)
        exact(row_embedding(ids, gpu_row, **selected), expected)
        dense_samples, row_samples = paired(
            lambda: F.embedding(ids, dense),
            lambda: row_embedding(ids, gpu_row, **selected), args.repeat, args.rounds)
        case["dense_pair_samples_us"] = {"dense": dense_samples, "row": row_samples}
        case["dense_us"] = statistics.median(dense_samples)
        case["row_vs_dense_us"] = statistics.median(row_samples)
        print("dense pair", case["id_dtype"], m,
              round(case["dense_us"], 3), round(case["row_vs_dense_us"], 3), flush=True)
        del ids, expected
        torch.cuda.empty_cache()
    retained = {name: tensor for name, tensor in vars(gpu_row).items() if isinstance(tensor, torch.Tensor)}
    assert set(retained) == {"packed", "escapes"}
    report["retained_after_repeated_forwards"] = {
        name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": tensor.nbytes}
        for name, tensor in retained.items()}
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    assert report["peak_reserved_bytes"] < 3_000_000_000
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
