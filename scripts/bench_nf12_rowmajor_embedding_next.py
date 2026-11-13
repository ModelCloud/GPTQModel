# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPU1 row-major NF12 vectorization and cache experiment.

Run with one visible GPU and PYTHON_GIL=0. This does not alter the installed
NF12 operator or retain a decoded BF16 weight in the candidate's state.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import gptqmodel_nf12 as nf12
from gptqmodel_nf12.experimental_rowmajor import from_nf12
from scripts.bench_nf12_lite_observed_shapes import capture, graph_us


repack_rowmajor = from_nf12


def load_kernel():
    source = Path(__file__).resolve().parents[1] / "gptqmodel_ext/nf12/experiments/rowmajor_embedding.cu"
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    output = Path(tempfile.gettempdir()) / f"nf12-rowmajor-next-embed-{key}.so"
    if not output.is_file():
        temp_output = output.with_suffix(".tmp.so")
        subprocess.run(["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-shared",
                        "-Xcompiler", "-fPIC", "-gencode", "arch=compute_80,code=sm_80",
                        str(source), "-o", str(temp_output)], check=True)
        temp_output.replace(output)
    library = ctypes.CDLL(str(output))
    fn = library.nf12_rowmajor_lookup
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                   ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32, ctypes.c_int64,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int

    class Kernel:
        @staticmethod
        def lookup(ids, packed, escapes, n, k, base, threads, ids_per_cta=1, mode=1):
            if (ids.device.type != "cuda" or packed.device != ids.device or escapes.device != ids.device
                    or ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous()
                    or packed.dtype != torch.uint8 or not packed.is_contiguous()
                    or packed.numel() != n * k * 3 // 2
                    or escapes.dtype != torch.int16 or not escapes.is_contiguous()
                    or escapes.ndim != 2 or escapes.shape[1] != 4):
                raise ValueError("invalid row-major embedding tensors")
            output = torch.empty((*ids.shape, k), device=ids.device, dtype=torch.bfloat16)
            if ids.numel():
                stream = torch.cuda.current_stream(ids.device).cuda_stream
                error = fn(ids.data_ptr(), packed.data_ptr(), escapes.data_ptr(), escapes.shape[0],
                           n, k, base, ids.numel(), output.data_ptr(), ids.element_size(),
                           threads, ids_per_cta, mode, stream)
                if error:
                    raise RuntimeError(f"CUDA launch error {error}")
            return output

    return Kernel()


def check_bits(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())


def paired_samples(fn_a, fn_b, repeat: int, rounds: int) -> tuple[list[float], list[float]]:
    graph_a, graph_b = capture(fn_a, repeat), capture(fn_b, repeat)
    for _ in range(3):
        graph_a.replay()
        graph_b.replay()
    samples = ([], [])
    for _ in range(rounds):
        for i in (0, 1, 1, 0):
            samples[i].append(graph_us((graph_a, graph_b)[i], repeat))
    del graph_a, graph_b
    return samples


def case_ids(kind: str, m: int, n: int, fixture: list[int], dtype: torch.dtype) -> torch.Tensor:
    if kind == "code":
        return torch.tensor(fixture[:m], device="cuda:0", dtype=dtype)
    values = torch.randint(n, (m,), device="cuda:0", dtype=torch.int32,
                           generator=torch.Generator(device="cuda:0").manual_seed(30911 + m))
    return values.to(dtype)


def edge_test(kernel) -> dict:
    # Lower halves create selected-meta sentinel collisions; upper halves have
    # nonzero metadata. The second super-atom contains original escapes.
    bits = torch.full((16, 128), 0x3f00, dtype=torch.int16)
    bits[8:, :64] = -16640  # 0xbf00, same exponent with negative sign
    bits[:8, 64:] = -16640  # reverse sign, so upper halves collide here
    bits[0, 64] = 0x7fc0  # NaN payload
    bits[8, 64] = -32768  # signed zero
    bits[1, 65] = 1  # subnormal
    bits[9, 65] = -1
    weight_cpu = bits.view(torch.bfloat16)
    # Load the independent pure-Torch codec directly: the lightweight GPU1
    # environment does not install optional GPT-QModel frontend dependencies.
    codec_path = Path(__file__).resolve().parents[1] / "gptqmodel/utils/nf12_codec.py"
    spec = importlib.util.spec_from_file_location("nf12_rowmajor_oracle", codec_path)
    assert spec is not None and spec.loader is not None
    nf12_codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf12_codec)
    packed, escapes, base = nf12_codec.encode(weight_cpu, exp_base=63)
    source = nf12.NF12Weight(packed.cuda(), escapes.cuda(), base)
    row = repack_rowmajor(source)
    assert row.selected_half_collisions > 0
    assert row.original_escapes > 0
    observed = []
    for dtype in (torch.int32, torch.int64):
        ids = torch.tensor([-1, 0, 1, 7, 8, 15, 16, 2**31 if dtype == torch.int64 else 17],
                           device="cuda:0", dtype=dtype)
        expected = torch.zeros((len(ids), 128), device="cuda:0", dtype=torch.bfloat16)
        valid = (ids >= 0) & (ids < 16)
        expected[valid] = weight_cpu.cuda()[ids[valid].long()]
        for mode in range(7):
            for threads in (64, 128, 256, 512):
                for ids_per_cta in (1, 2, 4, 8, 16):
                    if ids_per_cta > threads // (8 if mode == 0 else 2 if mode >= 5 else 4):
                        continue
                    actual = kernel.lookup(ids, row.packed, row.escapes, 16, 128,
                                           base, threads, ids_per_cta, mode)
                    mismatch = check_bits(actual, expected)
                    assert mismatch == 0, (dtype, mode, threads, ids_per_cta, mismatch)
        observed.append({"dtype": str(dtype), "mismatch_bits": 0})
    assert set(vars(row)) == {"packed", "escapes", "exp_base", "shape", "original_escapes",
                              "selected_half_collisions"}
    return {"shape": [16, 128], "original_escapes": row.original_escapes,
            "selected_half_collisions": row.selected_half_collisions,
            "row_escape_records": row.escapes.shape[0], "cases": observed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/fast/model/DeepSeek-V4.1-Flash-1017"))
    parser.add_argument("--ids-json", type=Path,
                        default=Path("docs/nf12-lite-cuda-token-ids-20261111.json"))
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--sweep-rounds", type=int, default=4)
    args = parser.parse_args()
    if args.repeat <= 0 or args.rounds <= 0 or args.sweep_rounds <= 0:
        parser.error("repeat, rounds and sweep-rounds must be positive")
    if (sys._is_gil_enabled() or os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
            or torch.cuda.device_count() != 1 or torch.version.hip is not None):
        parser.error("run with PYTHON_GIL=0, CUDA_VISIBLE_DEVICES=1, NVIDIA CUDA")
    if torch.cuda.get_device_capability(0) != (8, 0):
        parser.error("GPU1 SM80 required")
    kernel = load_kernel()
    edge = edge_test(kernel)
    print("edge", edge, flush=True)
    ids_fixture = json.loads(args.ids_json.read_text())
    assert len(ids_fixture) >= 3072
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(args.checkpoint / index["embed.weight"], framework="pt", device="cpu") as shard:
        source_cpu = shard.get_tensor("embed.weight")
    assert source_cpu.dtype == torch.bfloat16 and tuple(source_cpu.shape) == (129280, 5120)
    weight_sha = hashlib.sha256(source_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    weight = source_cpu.cuda()
    original = nf12.encode(weight)
    del weight
    torch.cuda.empty_cache()
    row = repack_rowmajor(original)
    n, k = original.shape
    assert set(vars(row)) == {"packed", "escapes", "exp_base", "shape", "original_escapes",
                              "selected_half_collisions"}
    invalid_cases = []
    for dtype, values in ((torch.int32, [-1, 0, n - 1, n, n + 1]),
                          (torch.int64, [-2**40, -1, 0, n - 1, n, 2**40])):
        ids = torch.tensor(values, device="cuda:0", dtype=dtype)
        expected_cpu = torch.zeros((len(values), k), dtype=torch.bfloat16)
        for j, value in enumerate(values):
            if 0 <= value < n:
                expected_cpu[j] = source_cpu[value]
        for mode in range(7):
            actual = kernel.lookup(ids, row.packed, row.escapes, n, k,
                                   row.exp_base, 256, 4, mode)
            mismatch = check_bits(actual, expected_cpu.cuda())
            assert mismatch == 0, (dtype, mode, mismatch)
        invalid_cases.append({"dtype": str(dtype), "ids": values, "mismatch_bits": 0})
    del ids, actual, expected_cpu
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__, "gil_enabled": sys._is_gil_enabled(),
        "visible_physical_gpu": 1, "weight_sha256": weight_sha,
        "fixture_sha256": hashlib.sha256(args.ids_json.read_bytes()).hexdigest(),
        "shape": [n, k], "repeat": args.repeat, "rounds": args.rounds,
        "sweep_rounds": args.sweep_rounds,
        "method": "full released weight; code ID fixture prefix and seeded random IDs; "
                  "CUDA graph sweeps; same-process separate NF12/row and dense/row paired ABBA",
        "storage": {"dense_oracle_bytes": source_cpu.nbytes,
                    "original_nf12_bytes": original.nbytes,
                    "rowmajor_packed_bytes": row.packed.nbytes,
                    "rowmajor_escape_bytes": row.escapes.nbytes,
                    "rowmajor_total_bytes": row.nbytes,
                    "original_escape_records": row.original_escapes,
                    "selected_half_collisions": row.selected_half_collisions,
                    "additional_persistent_dequantized_bytes": 0,
                    "temporary_dequantized_weight_bytes": 0},
        "modes": {"0": "previous 8-lane", "1": "4-lane vector",
                  "2": "4-lane vector stcg", "3": "4-lane vector ldcg",
                  "4": "4-lane vector ldcg stcg", "5": "2-lane vector",
                  "6": "2-lane vector stcg"},
        "edge": edge, "full_weight_invalid_ids": invalid_cases,
        "cases": [],
    }
    # Sweep only the repeated-ID target. Carry its selected geometry into
    # random-ID validation to expose the cost of tuning for the actual fixture.
    geometries = ((64, 1), (128, 1), (256, 1), (512, 1),
                  (128, 2), (256, 2), (512, 2), (256, 4),
                  (512, 4), (256, 8), (512, 8), (256, 16), (512, 16))
    best_by_dtype_shape = {}
    old_by_dtype_shape = {}
    for dtype in (torch.int64, torch.int32):
        dtype_name = "int64" if dtype == torch.int64 else "int32"
        for m in (1536, 3072):
            ids = case_ids("code", m, n, ids_fixture, dtype)
            expected = nf12.embedding(ids, original)
            sweep = {}
            for mode in range(7):
                for threads, ids_per_cta in geometries:
                    if ids_per_cta > threads // (8 if mode == 0 else 2 if mode >= 5 else 4):
                        continue
                    actual = kernel.lookup(ids, row.packed, row.escapes, n, k,
                                           row.exp_base, threads, ids_per_cta, mode)
                    mismatch = check_bits(actual, expected)
                    assert mismatch == 0, (dtype_name, m, mode, threads, ids_per_cta, mismatch)
                    fn = partial(kernel.lookup, ids, row.packed, row.escapes, n, k,
                                 row.exp_base, threads, ids_per_cta, mode)
                    graph = capture(fn, args.repeat)
                    timings = [graph_us(graph, args.repeat) for _ in range(args.sweep_rounds)]
                    sweep[f"{mode}:{threads}x{ids_per_cta}"] = timings
                    del graph, fn, actual
            ranked = sorted((statistics.median(times), key) for key, times in sweep.items())
            best = next(key for _, key in ranked if not key.startswith("0:"))
            old = next(key for _, key in ranked if key.startswith("0:"))
            best_by_dtype_shape[dtype_name, m] = best
            old_by_dtype_shape[dtype_name, m] = old
            report["cases"].append({"kind": "code", "m": m, "id_dtype": dtype_name,
                                    "sweep_us": sweep, "selected": best,
                                    "previous_row_selected": old,
                                    "selected_sweep_median_us": statistics.median(sweep[best])})
            print("sweep", dtype_name, m, "best", best, round(statistics.median(sweep[best]), 3),
                  "old", old, round(statistics.median(sweep[old]), 3), flush=True)
            del ids, expected
            torch.cuda.empty_cache()
    by_case = {(c["kind"], c["m"], c["id_dtype"]): c for c in report["cases"]}
    for dtype in (torch.int64, torch.int32):
        dtype_name = "int64" if dtype == torch.int64 else "int32"
        for kind in ("code", "random"):
            for m in (1536, 3072):
                ids = case_ids(kind, m, n, ids_fixture, dtype)
                selected = best_by_dtype_shape[dtype_name, m]
                mode, geometry = selected.split(":")
                threads, ids_per_cta = map(int, geometry.split("x"))
                fn = partial(kernel.lookup, ids, row.packed, row.escapes, n, k,
                             row.exp_base, threads, ids_per_cta, int(mode))
                expected = nf12.embedding(ids, original)
                actual = fn()
                mismatch = check_bits(actual, expected)
                assert mismatch == 0, (kind, dtype_name, m, mismatch)
                base_times, candidate_times = paired_samples(
                    partial(nf12.embedding, ids, original), fn, args.repeat, args.rounds)
                case = by_case.get((kind, m, dtype_name))
                if case is None:
                    case = {"kind": kind, "m": m, "id_dtype": dtype_name,
                            "selected": selected}
                    report["cases"].append(case)
                    by_case[kind, m, dtype_name] = case
                case["nf12_pair_samples_us"] = {"nf12": base_times, "candidate": candidate_times}
                case["nf12_us"] = statistics.median(base_times)
                case["candidate_vs_nf12_us"] = statistics.median(candidate_times)
                case["exact_bf16_bits_vs_nf12"] = True
                if kind == "code":
                    old = old_by_dtype_shape[dtype_name, m]
                    old_mode, old_geometry = old.split(":")
                    old_threads, old_ids_per_cta = map(int, old_geometry.split("x"))
                    old_fn = partial(kernel.lookup, ids, row.packed, row.escapes, n, k,
                                     row.exp_base, old_threads, old_ids_per_cta, int(old_mode))
                    old_times, optimized_times = paired_samples(old_fn, fn, args.repeat, args.rounds)
                    case["previous_row_pair_samples_us"] = {
                        "previous_row": old_times, "candidate": optimized_times}
                    case["previous_row_us"] = statistics.median(old_times)
                    case["candidate_vs_previous_row_us"] = statistics.median(optimized_times)
                print("nf12 pair", dtype_name, kind, m,
                      round(case["nf12_us"], 3), round(case["candidate_vs_nf12_us"], 3),
                      flush=True)
                del ids, expected, actual, fn
                torch.cuda.empty_cache()
    report["peak_after_nf12_phase_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_after_nf12_phase_reserved_bytes"] = torch.cuda.max_memory_reserved()
    del original
    torch.cuda.empty_cache()
    weight = source_cpu.cuda()
    for case in report["cases"]:
        kind, m, dtype_name = case["kind"], case["m"], case["id_dtype"]
        dtype = torch.int64 if dtype_name == "int64" else torch.int32
        ids = case_ids(kind, m, n, ids_fixture, dtype)
        mode, geometry = case["selected"].split(":")
        threads, ids_per_cta = map(int, geometry.split("x"))
        fn = partial(kernel.lookup, ids, row.packed, row.escapes, n, k,
                     row.exp_base, threads, ids_per_cta, int(mode))
        expected = F.embedding(ids, weight)
        actual = fn()
        mismatch = check_bits(actual, expected)
        assert mismatch == 0, (kind, dtype_name, m, mismatch)
        dense_times, candidate_times = paired_samples(
            partial(F.embedding, ids, weight), fn, args.repeat, args.rounds)
        case["dense_pair_samples_us"] = {"dense": dense_times, "candidate": candidate_times}
        case["dense_us"] = statistics.median(dense_times)
        case["candidate_vs_dense_us"] = statistics.median(candidate_times)
        case["exact_bf16_bits_vs_dense"] = True
        print("dense pair", dtype_name, kind, m,
              round(case["dense_us"], 3), round(case["candidate_vs_dense_us"], 3), flush=True)
        del ids, expected, actual, fn
        torch.cuda.empty_cache()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    retained = {name: value for name, value in vars(row).items()
                if isinstance(value, torch.Tensor)}
    assert set(retained) == {"packed", "escapes"}
    assert retained["packed"].dtype == torch.uint8 and retained["packed"].nbytes == n * k * 3 // 2
    assert retained["escapes"].dtype == torch.int16 and retained["escapes"].shape[1] == 4
    report["candidate_retained_after_repeated_forwards"] = {
        name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": tensor.nbytes}
        for name, tensor in retained.items()}
    assert report["peak_reserved_bytes"] <= 3_000_000_000, report["peak_reserved_bytes"]
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
