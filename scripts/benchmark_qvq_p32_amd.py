#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the gfx950 P32 kernel over the requested decode/prefill M sweep."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUESTED_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
P32_RATES = (2.0, 2.5, 3.0, 3.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=P32_RATES)
    parser.add_argument("--m-values", type=int, nargs="+", default=REQUESTED_M)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--fallback-warmup", type=int, default=1)
    parser.add_argument("--fallback-iterations", type=int, default=5)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="Run an explicitly invalidated exploratory benchmark when foreign residency exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qvq_p32_gfx950.json"),
    )
    args = parser.parse_args()
    if any(rate not in P32_RATES for rate in args.rates):
        parser.error("--rates supports only W2, W2.5, W3, and W3.5")
    if any(m not in REQUESTED_M for m in args.m_values):
        parser.error(f"--m-values must be drawn from {REQUESTED_M}")
    if args.k <= 0 or args.n <= 0 or args.k % 16 or args.n % 16:
        parser.error("--k and --n must be positive multiples of 16")
    if min(
        args.warmup,
        args.iterations,
        args.fallback_warmup,
        args.fallback_iterations,
        args.idle_samples,
    ) <= 0:
        parser.error("warmup, iteration, and idle-sample counts must be positive")
    return args


def _rocm_snapshot(physical_gpu: int) -> dict[str, object]:
    command = [
        "rocm-smi",
        "-d",
        str(physical_gpu),
        "--showuse",
        "--showmeminfo",
        "vram",
        "--showpids",
        "--showbus",
        "--showuniqueid",
        "--showdriverversion",
        "--json",
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    card = payload.get(f"card{physical_gpu}")
    if not isinstance(card, dict):
        raise TypeError(f"rocm-smi did not report physical GPU {physical_gpu}: {payload}")
    system = payload.get("system", {})
    process_ids = sorted(
        int(key.removeprefix("PID"))
        for key in system
        if isinstance(key, str) and key.startswith("PID") and key.removeprefix("PID").isdigit()
    )
    return {
        "physical_gpu": physical_gpu,
        "pci_bus_id": card.get("PCI Bus", "unknown"),
        "unique_id": card.get("Unique ID", "unknown"),
        "driver": system.get("Driver version", "unknown"),
        "utilization_percent": int(card["GPU use (%)"]),
        "vram_total_bytes": int(card["VRAM Total Memory (B)"]),
        "vram_used_bytes": int(card["VRAM Total Used Memory (B)"]),
        "process_ids": process_ids,
    }


def _idle_preflight(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    accepted = None
    violations = []
    tolerance_bytes = args.idle_memory_tolerance_mib * 1024 * 1024
    for sample in range(args.idle_samples):
        accepted = _rocm_snapshot(args.physical_gpu)
        sample_violations = []
        if accepted["utilization_percent"] != 0:
            sample_violations.append(f"utilization={accepted['utilization_percent']}%")
        if accepted["vram_used_bytes"] > tolerance_bytes:
            sample_violations.append(
                f"VRAM={accepted['vram_used_bytes'] / 2**20:.1f}MiB>{args.idle_memory_tolerance_mib}MiB"
            )
        if accepted["process_ids"]:
            sample_violations.append(f"foreign_pids={accepted['process_ids']}")
        violations.extend(f"sample {sample + 1}: {item}" for item in sample_violations)
        if sample + 1 < args.idle_samples:
            time.sleep(args.idle_interval)
    assert accepted is not None
    valid = not violations
    print(
        "ROCm idle gate: "
        f"physical={accepted['physical_gpu']} pci={accepted['pci_bus_id']} unique_id={accepted['unique_id']} "
        f"utilization={accepted['utilization_percent']}% vram={accepted['vram_used_bytes'] / 2**20:.1f}MiB "
        f"samples={args.idle_samples} valid={valid}",
        flush=True,
    )
    if violations:
        message = "ROCm idle gate failed: " + "; ".join(violations)
        if not args.allow_busy:
            raise RuntimeError(message)
        print(f"INVALIDATED EXPLORATORY RUN: {message}", flush=True)
    return accepted, valid


def _timing_recheck(
    args: argparse.Namespace,
    permitted_pids: set[int],
) -> tuple[dict[str, object], bool]:
    """Reject activity that appeared after the process initialized ROCm."""

    snapshot = _rocm_snapshot(args.physical_gpu)
    added_pids = set(snapshot["process_ids"]) - permitted_pids
    violations = []
    # Utilization after warmup belongs to this process. PID identity is the
    # stable way to reject a workload that arrived after the idle preflight.
    if added_pids:
        violations.append(f"new_pids={sorted(added_pids)}")
    valid = not violations
    if violations:
        message = "pre-timing ROCm recheck failed: " + "; ".join(violations)
        if not args.allow_busy:
            raise RuntimeError(message)
        print(f"INVALIDATED EXPLORATORY RUN: {message}", flush=True)
    return snapshot, valid


def _timings(torch, fn, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
    }


def _metrics(actual, reference) -> dict[str, float]:
    difference = actual.float() - reference.float()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": difference.norm().div(reference.float().norm().clamp_min(1e-12)).item(),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = (
        "W",
        "M",
        "P32 p50 ms",
        "P32 TF/s",
        "Fallback p50 ms",
        "Fallback/P32",
        "Dense p50 ms",
        "Max abs",
        "Rel L2",
    )
    rendered = []
    for row in rows:
        rendered.append(
            (
                f"{row['bits']:g}",
                str(row["m"]),
                f"{row['p32']['median_ms']:.6f}",
                f"{row['p32_tflops']:.3f}",
                f"{row['fallback']['median_ms']:.6f}",
                f"{row['fallback']['median_ms'] / row['p32']['median_ms']:.2f}x",
                f"{row['dense']['median_ms']:.6f}",
                f"{row['accuracy']['max_abs']:.7g}",
                f"{row['accuracy']['relative_l2']:.7g}",
            )
        )
    widths = [
        max(len(column), *(len(values[index]) for values in rendered))
        for index, column in enumerate(columns)
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    print("| " + " | ".join(column.ljust(width) for column, width in zip(columns, widths)) + " |")
    print(border)
    for values in rendered:
        print("| " + " | ".join(value.rjust(width) for value, width in zip(values, widths)) + " |")
    print(border, flush=True)


def main() -> None:
    args = _parse_args()
    hardware, valid = _idle_preflight(args)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-mi355x")

    import torch

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_amd import qvq_p32_amd, qvq_p32_amd_supported

    if torch.cuda.device_count() != 1 or not qvq_p32_amd_supported("cuda:0"):
        raise RuntimeError("benchmark requires exactly one visible gfx950 ROCm device")
    properties = torch.cuda.get_device_properties(0)
    software = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": getattr(__import__("triton"), "__version__", "unknown"),
    }
    hardware.update(
        {
            "name": properties.name,
            "gcn_arch": properties.gcnArchName,
            "compute_units": properties.multi_processor_count,
            "torch_total_memory": properties.total_memory,
        }
    )
    print(f"hardware={hardware}\nsoftware={software}", flush=True)

    rows = []
    timing_rechecks = []
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to("cuda")
    runtime_context = _rocm_snapshot(args.physical_gpu)
    permitted_pids = set(hardware["process_ids"]) | set(runtime_context["process_ids"])
    for bits in args.rates:
        generator = torch.Generator(device="cuda").manual_seed(950000 + int(bits * 10))
        tile_count = (args.k // 16) * (args.n // 16)
        planar = torch.randint(
            -(1 << 31),
            1 << 31,
            (tile_count, qvq_words_per_tile(bits, vector_size=2)),
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
        bank_ids = pack_qvq_binary_bank_ids(
            torch.randint(0, 2, (tile_count * 8,), dtype=torch.uint8, device="cuda", generator=generator)
        )
        bank_alt_id = torch.tensor([3], dtype=torch.uint8, device="cuda")
        window = repack_p32_planar_to_window(planar, bits=bits)
        dense = reconstruct_qvq_inner_weight(
            planar,
            bits=bits,
            in_features=args.k,
            out_features=args.n,
            bank_ids=bank_ids,
            v2b2_p32=True,
            bank_alt_id=bank_alt_id,
        ).half()
        compressed_bytes = window.numel() * window.element_size() + bank_ids.numel() * bank_ids.element_size()
        dense_bytes = dense.numel() * dense.element_size()
        for m in args.m_values:
            x = (
                torch.randn((m, args.k), dtype=torch.float16, device="cuda", generator=generator)
                * 0.01
            ).contiguous()
            def candidate(
                x=x,
                window=window,
                levels=levels,
                bank_ids=bank_ids,
                bits=bits,
            ):
                return qvq_p32_amd(
                    x,
                    window,
                    levels,
                    bank_ids,
                    bits,
                    out_features=args.n,
                    bank_alt_id=3,
                )

            def fallback(
                x=x,
                planar=planar,
                bank_ids=bank_ids,
                bank_alt_id=bank_alt_id,
                bits=bits,
            ):
                return x @ reconstruct_qvq_inner_weight(
                    planar,
                    bits=bits,
                    in_features=args.k,
                    out_features=args.n,
                    bank_ids=bank_ids,
                    v2b2_p32=True,
                    bank_alt_id=bank_alt_id,
                ).to(dtype=x.dtype)

            def dense_gemm(x=x, dense=dense):
                return x @ dense

            actual = candidate()
            reference = x.float() @ dense.float()
            torch.cuda.synchronize()
            accuracy = _metrics(actual, reference)
            if accuracy["max_abs"] > 2e-3:
                raise RuntimeError(f"W{bits:g} M{m} accuracy failed: {accuracy}")
            timing_snapshot, timing_valid = _timing_recheck(args, permitted_pids)
            timing_rechecks.append(timing_snapshot)
            valid = valid and timing_valid
            p32_timing = _timings(torch, candidate, warmup=args.warmup, iterations=args.iterations)
            fallback_timing = _timings(
                torch,
                fallback,
                warmup=args.fallback_warmup,
                iterations=args.fallback_iterations,
            )
            dense_timing = _timings(torch, dense_gemm, warmup=args.warmup, iterations=args.iterations)
            p32_tflops = 2 * m * args.k * args.n / (p32_timing["median_ms"] * 1e9)
            rows.append(
                {
                    "bits": bits,
                    "m": m,
                    "k": args.k,
                    "n": args.n,
                    "p32": p32_timing,
                    "p32_tflops": p32_tflops,
                    "fallback": fallback_timing,
                    "dense": dense_timing,
                    "storage": {
                        "compressed_bytes": compressed_bytes,
                        "dense_fp16_bytes": dense_bytes,
                        "compression_ratio": dense_bytes / compressed_bytes,
                    },
                    "accuracy": accuracy,
                }
            )
            print(
                f"completed W{bits:g} M{m}: max_abs={accuracy['max_abs']:.7g} "
                f"p32={p32_timing['median_ms']:.6f}ms "
                f"fallback={fallback_timing['median_ms']:.6f}ms",
                flush=True,
            )

    _print_table(rows)

    result = {
        "schema": "qvq_p32_gfx950_benchmark_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": valid,
        "accuracy_gate": 2e-3,
        "hardware": hardware,
        "software": software,
        "runtime_context": runtime_context,
        "timing_rechecks": timing_rechecks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
