# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare two NF12 native embedding libraries on one released BF16 table.

Load/encode the table once through the candidate's Torch extension. Call the
baseline and candidate C APIs directly so both implementations can coexist in
one process and use the same packed pointers, IDs, and CUDA stream.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12
from scripts.bench_nf12_lite_observed_shapes import graph_us


class NativeWeight(ctypes.Structure):
    _fields_ = [
        ("packed", ctypes.c_void_p),
        ("escapes", ctypes.c_void_p),
        ("num_escapes", ctypes.c_uint32),
        ("n", ctypes.c_int64),
        ("k", ctypes.c_int64),
        ("exp_base", ctypes.c_uint8),
    ]


def native_embedding(path: Path, id_dtype: torch.dtype, weight: NativeWeight):
    lib = ctypes.CDLL(str(path.resolve()))
    name = "nf12_embedding_i64" if id_dtype == torch.int64 else "nf12_embedding"
    op = getattr(lib, name)
    op.argtypes = [ctypes.POINTER(NativeWeight), ctypes.c_void_p, ctypes.c_int64,
                   ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    op.restype = ctypes.c_int
    def call(ids: torch.Tensor, out: torch.Tensor) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        status = op(ctypes.byref(weight), ids.data_ptr(), ids.numel(), out.data_ptr(),
                    weight.k, stream)
        if status != 0:
            raise RuntimeError(f"{path}: {name} returned {status}")

    # Keep dlopen's handle alive for the closure's entire lifetime.
    call._lib = lib
    return call


def capture(calls, repeat: int) -> torch.cuda.CUDAGraph:
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(repeat):
            calls[j]()
    return graph


def check_memory(initial_free: int, limit: int) -> int:
    free, _ = torch.cuda.mem_get_info()
    if initial_free - free > limit:
        raise MemoryError(f"visible GPU added memory {initial_free - free:,} exceeds {limit:,}")
    return free


def physical_process_memory_mib() -> int:
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical.isdecimal():
        raise RuntimeError("physical GPU memory accounting requires one numeric CUDA_VISIBLE_DEVICES")
    result = subprocess.run(
        ["nvidia-smi", "-i", physical, "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    matches = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields[0] == str(os.getpid()):
            matches.append(int(fields[1]))
    if len(matches) != 1:
        raise RuntimeError(f"nvidia-smi did not report this process on physical GPU {physical}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-lib", type=Path, required=True)
    parser.add_argument("--candidate-lib", type=Path, required=True)
    parser.add_argument("--torch-lib", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--ids-json", type=Path,
                        default=Path("docs/nf12-lite-cuda-token-ids-20261111.json"))
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=20261111)
    parser.add_argument("--rows", type=int, nargs="+", default=(312, 313, 1536, 3072))
    parser.add_argument("--id-dtypes", nargs="+", choices=("int32", "int64"),
                        default=("int32", "int64"))
    parser.add_argument("--max-added-gb", type=float, default=3.0)
    parser.add_argument("--profile-case", type=int, choices=(312, 1536, 3072))
    parser.add_argument("--profile-kind", choices=("fixture", "random"), default="fixture")
    args = parser.parse_args()
    if sys._is_gil_enabled() or torch.cuda.device_count() != 1:
        parser.error("requires PYTHON_GIL=0 and exactly one visible CUDA GPU")
    if args.repeat < 1 or args.rounds < 1 or any(m < 1 or m > 4095 for m in args.rows):
        parser.error("invalid repeat, rounds, or rows")
    torch.cuda.set_per_process_memory_fraction(0.03)
    torch.cuda.init()
    initial_free, total = torch.cuda.mem_get_info()
    limit = int(args.max_added_gb * 1024**3)
    if initial_free < limit + 2 * 1024**3:
        raise MemoryError(f"visible GPU has only {initial_free:,} bytes free")
    torch.ops.load_library(str(args.candidate_lib.resolve()))
    torch.ops.load_library(str(args.torch_lib.resolve()))
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / index["embed.weight"], framework="pt", device="cpu") as shard:
        source = shard.get_tensor("embed.weight")
    assert source.dtype == torch.bfloat16 and source.shape[1] == 5120
    source_sha = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
    dense = source.to("cuda:0")
    packed = nf12.encode(dense)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    check_memory(initial_free, limit)
    if physical_process_memory_mib() * 1024**2 > limit:
        raise MemoryError("physical GPU process memory exceeds the limit after packed encoding")
    # Restore the entire released table in small, 16-row-aligned chunks so
    # validation never holds a second full BF16 table on the device.
    decode_chunk_rows = 128
    for start in range(0, packed.shape[0], decode_chunk_rows):
        length = min(decode_chunk_rows, packed.shape[0] - start)
        assert length % 16 == 0
        restored = packed.narrow_rows(start, length).decode()
        assert torch.equal(restored.view(torch.int16), dense[start:start + length].view(torch.int16))
        del restored
    check_memory(initial_free, limit)
    if physical_process_memory_mib() * 1024**2 > limit:
        raise MemoryError("physical GPU process memory exceeds the limit after table restoration")
    native_weight = NativeWeight(packed.packed.data_ptr(), packed.escapes.data_ptr(),
                                 packed.num_escapes, *packed.shape, packed.exp_base)
    fixture = json.loads(args.ids_json.read_text())
    assert isinstance(fixture, list) and len(fixture) >= 3072
    assert all(type(x) is int and 0 <= x < packed.shape[0] for x in fixture[:3072])
    module = nf12.NF12Embedding.from_weight(packed)
    module_ids = torch.tensor(fixture[:6], dtype=torch.int32, device="cuda:0")
    for _ in range(2):
        module_out = module(module_ids)
        assert torch.equal(module_out.view(torch.int16), F.embedding(module_ids, dense).view(torch.int16))
    module_buffers = dict(module.named_buffers())
    assert set(module_buffers) == {"nf12_packed", "nf12_escapes"}
    assert module_buffers["nf12_packed"].data_ptr() == packed.packed.data_ptr()
    assert module_buffers["nf12_escapes"].data_ptr() == packed.escapes.data_ptr()
    assert all(buffer.dtype != torch.bfloat16 for buffer in module_buffers.values())
    del module_out, module_ids, module
    if args.profile_case is not None:
        m = args.profile_case
        if args.profile_kind == "fixture":
            ids = torch.tensor(fixture[:m], dtype=torch.int32, device="cuda:0")
        else:
            generator = torch.Generator(device="cuda:0").manual_seed(args.random_seed + m)
            ids = torch.randint(packed.shape[0], (m,), device="cuda:0", dtype=torch.int32,
                                generator=generator)
        out = torch.empty((m, packed.shape[1]), dtype=torch.bfloat16, device="cuda:0")
        candidate = native_embedding(args.candidate_lib, torch.int32, native_weight)
        candidate(ids, out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), F.embedding(ids, dense).view(torch.int16))
        print(f"profile {args.label}: M={m}, {args.profile_kind}, exact BF16 bits", flush=True)
        return
    report = {
        "label": args.label,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "gil_enabled": sys._is_gil_enabled(),
        "checkpoint": str(args.checkpoint),
        "weight_key": "embed.weight",
        "weight_sha256": source_sha,
        "shape_nk": list(packed.shape),
        "dense_weight_bytes": dense.nbytes,
        "packed_weight_bytes": packed.nbytes,
        "packed_restoration_exact_bf16_bits": True,
        "decode_chunk_rows": decode_chunk_rows,
        "module_weight_buffer_bytes": sum(buffer.nbytes for buffer in module_buffers.values()),
        "additional_packed_cache_bytes": 0,
        "inference_retained_expanded_weight_bytes": 0,
        "inference_dequantization_scratch_bytes": 0,
        "benchmark_dense_oracle_bytes": dense.nbytes,
        "baseline_lib": str(args.baseline_lib.resolve()),
        "candidate_lib": str(args.candidate_lib.resolve()),
        "ids_json": str(args.ids_json.resolve()),
        "ids_prefix_sha256": hashlib.sha256(json.dumps(fixture[:3072], separators=(",", ":")).encode()).hexdigest(),
        "method": f"same full released weight and IDs; direct native C ABI; {args.repeat}-call CUDA graphs; "
                  "even rounds dense, baseline, candidate, candidate, baseline, dense; "
                  "odd rounds dense, candidate, baseline, baseline, candidate, dense; "
                  "random calls rotate independently sampled ID tensors",
        "repeat": args.repeat,
        "rounds": args.rounds,
        "random_seed": args.random_seed,
        "initial_free_bytes": initial_free,
        "total_gpu_bytes": total,
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "physical_process_limit_bytes": limit,
        "initial_process_used_mib": physical_process_memory_mib(),
        "cases": [],
    }
    for m in args.rows:
        for dtype in (torch.int32 if name == "int32" else torch.int64 for name in args.id_dtypes):
            baseline = native_embedding(args.baseline_lib, dtype, native_weight)
            candidate = native_embedding(args.candidate_lib, dtype, native_weight)
            for kind in ("fixture", "random"):
                if kind == "fixture" and m > len(fixture):
                    continue
                if kind == "fixture":
                    ids = torch.tensor(fixture[:m], dtype=dtype, device="cuda:0")
                    id_calls = [ids] * args.repeat
                else:
                    generator = torch.Generator(device="cuda:0").manual_seed(args.random_seed + m +
                                                                             (1 if dtype == torch.int64 else 0))
                    id_calls = [torch.randint(packed.shape[0], (m,), device="cuda:0",
                                              dtype=dtype, generator=generator)
                                for _ in range(args.repeat)]
                    ids = id_calls[0]
                out_base = torch.empty((m, packed.shape[1]), dtype=torch.bfloat16, device="cuda:0")
                out_cand = torch.empty_like(out_base)
                # Every distinct random tensor captured in the graph gets its
                # own independent dense oracle check, not just the first.
                for validation_ids in (id_calls if kind == "random" else id_calls[:1]):
                    baseline(validation_ids, out_base)
                    candidate(validation_ids, out_cand)
                    expected = F.embedding(validation_ids, dense)
                    torch.cuda.synchronize()
                    assert torch.equal(out_base.view(torch.int16), expected.view(torch.int16))
                    assert torch.equal(out_cand.view(torch.int16), expected.view(torch.int16))
                    del expected
                invalid = ids.clone()
                invalid[0] = -1
                invalid[-1] = packed.shape[0]
                for invalid_value in (-1, packed.shape[0],
                                      torch.iinfo(dtype).max):
                    invalid[0] = invalid_value
                    baseline(invalid, out_base)
                    candidate(invalid, out_cand)
                    torch.cuda.synchronize()
                    assert torch.equal(out_base.view(torch.int16), out_cand.view(torch.int16))
                    assert torch.count_nonzero(out_cand[0].view(torch.int16)).item() == 0
                    assert torch.count_nonzero(out_cand[-1].view(torch.int16)).item() == 0
                dense_calls = [lambda row=row: F.embedding(row, dense) for row in id_calls]
                base_calls = [lambda row=row, out=out_base: baseline(row, out) for row in id_calls]
                cand_calls = [lambda row=row, out=out_cand: candidate(row, out) for row in id_calls]
                dense_graph = capture(dense_calls, args.repeat)
                base_graph = capture(base_calls, args.repeat)
                cand_graph = capture(cand_calls, args.repeat)
                for graph in (dense_graph, base_graph, cand_graph):
                    graph.replay()
                torch.cuda.synchronize()
                samples = {"dense": [], "baseline": [], "candidate": []}
                graphs = {"dense": dense_graph, "baseline": base_graph, "candidate": cand_graph}
                for round_index in range(args.rounds):
                    sequence = (("dense", "baseline", "candidate", "candidate", "baseline", "dense")
                                if round_index % 2 == 0 else
                                ("dense", "candidate", "baseline", "baseline", "candidate", "dense"))
                    for name in sequence:
                        samples[name].append(round(graph_us(graphs[name], args.repeat), 4))
                medians = {name: statistics.median(values) for name, values in samples.items()}
                row = {
                    "m": m, "dtype": str(dtype), "ids_kind": kind,
                    "exact_bf16_bits": True, "invalid_ids_exact_zero_bits": True,
                    "validated_distinct_id_tensors": args.repeat if kind == "random" else 1,
                    "samples_us": samples,
                    "median_us": {name: round(value, 4) for name, value in medians.items()},
                    "candidate_over_baseline": round(medians["baseline"] / medians["candidate"], 4),
                    "candidate_over_dense": round(medians["dense"] / medians["candidate"], 4),
                    "free_bytes_after": check_memory(initial_free, limit),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "physical_process_used_mib": physical_process_memory_mib(),
                }
                if row["physical_process_used_mib"] * 1024**2 > limit:
                    raise MemoryError("physical GPU process memory exceeds the limit")
                report["cases"].append(row)
                print(args.label, m, dtype, kind, row["median_us"], flush=True)
                del invalid, dense_graph, base_graph, cand_graph
                del out_base, out_cand, ids, id_calls, dense_calls, base_calls, cand_calls
    report["max_added_bytes"] = initial_free - min(row["free_bytes_after"] for row in report["cases"])
    report["max_physical_process_used_mib"] = max(
        row["physical_process_used_mib"] for row in report["cases"])
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
