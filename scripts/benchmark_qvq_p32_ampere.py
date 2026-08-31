#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact continuous-window P32 WMMA against planar P32 on sm_80."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("NINJAFLAGS", "-j8")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "8")
os.environ.setdefault("NVCC_THREADS", "2")

from scripts import benchmark_qvq_cuda_lr as benchmark_utils

RATES = (2.0, 2.5, 3.0, 3.5)
SHAPES = {
    "full_q_gate": (5120, 12288),
    "full_kv": (5120, 1024),
    "attention_out": (6144, 5120),
    "linear_qkv": (5120, 10240),
    "linear_z": (5120, 6144),
    "mlp_gate_up": (5120, 17408),
    "mlp_down": (17408, 5120),
}
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_ampere_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_ampere_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_gemv_cuda.cu"),
    Path("scripts/benchmark_qvq_p32_ampere.py"),
    Path("tests/test_qvq_p32_ampere.py"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=(1, 16))
    parser.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=tuple(SHAPES))
    parser.add_argument("--split-count", type=int, default=0, help="0 uses the live-device auto policy")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a100_p32_window/qwen38_p32_ampere.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("--rates supports only 2, 2.5, 3, and 3.5")
    if any(m < 1 or m > 16 for m in args.m_values):
        parser.error("--m-values must be in [1, 16]")
    if args.split_count < 0 or args.split_count > 128:
        parser.error("--split-count must be in [0, 128]")
    if min(args.warmup, args.iterations, args.idle_samples) <= 0:
        parser.error("warmup, iterations, and idle-samples must be positive")
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT))
    for relative_path in SOURCE_PATHS:
        contents = (REPO_ROOT / relative_path).read_bytes()
        digest.update(str(relative_path).encode())
        digest.update(contents)
    return digest.hexdigest()


def _exclusive_recheck(hardware: dict[str, str]) -> None:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    foreign = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == hardware["uuid"] and int(fields[1]) != os.getpid():
            foreign.append(line)
    if foreign:
        raise RuntimeError(f"foreign compute process appeared before timing: {foreign}")
    print(f"pre-timing exclusivity recheck: uuid={hardware['uuid']} pid={os.getpid()} foreign=0", flush=True)


def _metrics(actual, expected) -> dict[str, float]:
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": difference.norm().div(expected.float().norm().clamp_min(1e-12)).item(),
    }


def _print_table(rows: list[dict]) -> None:
    print("| Rate | Shape | M | K | N | Planar ms | Ampere ms | Speedup | Max abs | Rel L2 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    speedups = []
    for row in rows:
        speedup = row["planar"]["median_ms"] / row["ampere"]["median_ms"]
        speedups.append(speedup)
        print(
            f"| W{row['bits']:g} | {row['shape']} | {row['m']} | {row['k']} | {row['n']} | "
            f"{row['planar']['median_ms']:.6f} | {row['ampere']['median_ms']:.6f} | {speedup:.3f}x | "
            f"{row['ampere_metrics']['max_abs']:.7g} | {row['ampere_metrics']['relative_l2']:.7g} |"
        )
    geomean = math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups))
    print(f"Ampere/planar geometric-mean speedup: {geomean:.3f}x", flush=True)


def _run(args: argparse.Namespace) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    fingerprint = _source_fingerprint()
    hardware = benchmark_utils._idle_preflight(
        args.physical_gpu,
        args.idle_samples,
        args.idle_interval,
        args.idle_memory_tolerance_mib,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = hardware["uuid"]

    import torch

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits, qvq_words_per_tile
    from gptqmodel.utils.cpp import (
        TorchOpsJitExtension,
        default_jit_cflags,
        default_jit_cuda_cflags,
        default_torch_ops_build_root,
    )
    from gptqmodel.utils.qvq_ampere_cuda import (
        _QVQ_AMPERE_EXTENSION,
        qvq_p32_window_ampere,
    )

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(f"P32 Ampere benchmark requires sm_80, got {properties.name} sm_{properties.major}{properties.minor}")
    planar_extension = TorchOpsJitExtension(
        name="gptqmodel_qvq_gemv_only_ops",
        namespace="gptqmodel_qvq",
        required_ops=("gemv",),
        sources=[str(REPO_ROOT / "gptqmodel_ext" / "qvq" / "qvq_gemv_cuda.cu")],
        build_root_env="GPTQMODEL_QVQ_GEMV_ONLY_BUILD_ROOT",
        default_build_root=lambda: default_torch_ops_build_root("qvq_gemv_only"),
        display_name="QVQ planar GEMV benchmark baseline",
        extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
        extra_cuda_cflags=lambda: default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads="2",
            include_split_compile=True,
            include_fast_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        requires_cuda=True,
    )
    if not planar_extension.load():
        raise RuntimeError(planar_extension.last_error_message())
    if not _QVQ_AMPERE_EXTENSION.load():
        raise RuntimeError(_QVQ_AMPERE_EXTENSION.last_error_message())

    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    rows = []
    case_index = 0
    for m in args.m_values:
        for shape_name in args.shapes:
            size_k, size_n = SHAPES[shape_name]
            for bits in args.rates:
                generator = torch.Generator(device="cuda").manual_seed(20261000 + case_index)
                tile_count = (size_k // 16) * (size_n // 16)
                words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
                planar = torch.randint(
                    0,
                    1 << 32,
                    (tile_count, words_per_tile),
                    generator=generator,
                    device="cuda",
                    dtype=torch.int64,
                ).to(torch.int32)
                window = repack_p32_planar_to_window(planar, bits=bits)
                bank_ids = pack_qvq_binary_bank_ids(
                    torch.randint(
                        0,
                        2,
                        (tile_count * 8,),
                        generator=generator,
                        device="cuda",
                        dtype=torch.uint8,
                    )
                )
                x = (torch.randn((m, size_k), generator=generator, device="cuda") * 0.1).half()
                dense = reconstruct_p32_window_inner_weight(
                    window,
                    bits=bits,
                    in_features=size_k,
                    out_features=size_n,
                    bank_ids=bank_ids,
                    bank_alt_id=torch.tensor(3, dtype=torch.uint8, device="cuda"),
                )
                expected = x.float() @ dense

                planar_call = partial(
                    planar_extension.op("gemv"),
                    x,
                    planar,
                    levels,
                    qvq_transition_bits(bits, vector_size=2),
                    size_n,
                    True,
                    bank_ids,
                    3,
                    3,
                )
                ampere_call = partial(
                    qvq_p32_window_ampere,
                    x,
                    window,
                    levels,
                    bank_ids,
                    bits,
                    out_features=size_n,
                    bank_alt_id=3,
                    split_count=args.split_count,
                )

                planar_output = planar_call()
                ampere_output = ampere_call()
                torch.cuda.synchronize()
                planar_metrics = _metrics(planar_output, expected)
                ampere_metrics = _metrics(ampere_output, expected)
                if max(planar_metrics["max_abs"], ampere_metrics["max_abs"]) > 2e-3:
                    raise RuntimeError(
                        f"correctness failed for M{m} W{bits:g} {shape_name}: "
                        f"planar={planar_metrics}, ampere={ampere_metrics}"
                    )
                _exclusive_recheck(hardware)
                planar_timing = benchmark_utils._event_timing(
                    torch, planar_call, warmup=args.warmup, iterations=args.iterations
                )
                ampere_timing = benchmark_utils._event_timing(
                    torch, ampere_call, warmup=args.warmup, iterations=args.iterations
                )
                row = {
                    "bits": bits,
                    "shape": shape_name,
                    "m": m,
                    "k": size_k,
                    "n": size_n,
                    "planar": planar_timing,
                    "ampere": ampere_timing,
                    "planar_metrics": planar_metrics,
                    "ampere_metrics": ampere_metrics,
                }
                rows.append(row)
                print(
                    f"complete M{m} W{bits:g} {shape_name}: planar={planar_timing['median_ms']:.6f} ms "
                    f"ampere={ampere_timing['median_ms']:.6f} ms max_abs={ampere_metrics['max_abs']:.7g}",
                    flush=True,
                )
                case_index += 1
                del planar, window, bank_ids, x, dense, expected, planar_output, ampere_output
                gc.collect()
                torch.cuda.empty_cache()

    if fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark source changed during the run")
    payload = {
        "commit": commit,
        "source_fingerprint": fingerprint,
        "hardware": hardware,
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
            "memory_bytes": properties.total_memory,
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "dtype": "float16 inputs/levels, float32 accumulation/output",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "split_count": args.split_count,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_table(rows)
    print(f"result: {args.output}", flush=True)
    return payload


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
