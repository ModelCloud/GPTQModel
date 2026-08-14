#!/usr/bin/env python3
"""Benchmark unchanged PGC16 tail-biting Viterbi scheduling on one CUDA GPU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--bits", type=float, nargs="+", default=[rate / 2 for rate in range(2, 17)])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    valid_bits = {rate / 2 for rate in range(2, 17)}
    invalid_bits = sorted(set(args.bits) - valid_bits)
    if invalid_bits:
        parser.error(f"--bits only supports half-step rates from 1 through 8; got {invalid_bits}")
    if any(batch < 1 for batch in args.batch_sizes):
        parser.error("--batch-sizes must contain only positive integers")
    if args.steps < 2:
        parser.error("--steps must be at least two")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    return args


def _idle_preflight(physical_gpu: int, samples: int, interval: float) -> dict[str, str]:
    query = "index,pci.bus_id,uuid,name,memory.used,utilization.gpu"
    accepted = None
    for sample in range(samples):
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        values = [value.strip() for value in output.split(",")]
        if len(values) != 6:
            raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
        accepted = dict(zip(query.split(","), values, strict=True))
        if int(accepted["memory.used"]) != 0 or int(accepted["utilization.gpu"]) != 0:
            raise RuntimeError(
                f"physical GPU {physical_gpu} is not idle: memory={accepted['memory.used']} MiB, "
                f"utilization={accepted['utilization.gpu']}%"
            )
        if sample + 1 < samples:
            time.sleep(interval)
    assert accepted is not None
    return accepted


def _timings(torch, fn, *, warmup: int, iterations: int) -> tuple[dict[str, float], object]:
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for iteration in range(iterations):
        starts[iteration].record()
        result = fn()
        ends[iteration].record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    assert result is not None
    return (
        {
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
        },
        result,
    )


def _print_table(rows: list[dict[str, object]]) -> None:
    columns = (
        ("W", "bits"),
        ("Batch", "batch_size"),
        ("Median ms", "median_ms"),
        ("Tiles/s", "tiles_per_second"),
        ("Peak alloc MiB", "peak_allocated_mib"),
        ("Peak reserve MiB", "peak_reserved_mib"),
        ("Loss abs delta", "loss_abs_delta"),
        ("Path exact", "path_exact"),
        ("Status", "status"),
    )
    rendered = []
    for row in rows:
        rendered.append(
            {
                "bits": f"{row['bits']:g}",
                "batch_size": str(row["batch_size"]),
                "median_ms": "" if row["median_ms"] is None else f"{row['median_ms']:.3f}",
                "tiles_per_second": ("" if row["tiles_per_second"] is None else f"{row['tiles_per_second']:.1f}"),
                "peak_allocated_mib": (
                    "" if row["peak_allocated_mib"] is None else f"{row['peak_allocated_mib']:.1f}"
                ),
                "peak_reserved_mib": ("" if row["peak_reserved_mib"] is None else f"{row['peak_reserved_mib']:.1f}"),
                "loss_abs_delta": ("" if row["loss_abs_delta"] is None else f"{row['loss_abs_delta']:.3e}"),
                "path_exact": "yes" if row["path_exact"] else "no",
                "status": str(row["status"]),
            }
        )
    widths = {key: max(len(title), *(len(row[key]) for row in rendered)) for title, key in columns}
    separator = "+" + "+".join("-" * (widths[key] + 2) for _, key in columns) + "+"
    print(separator)
    print("| " + " | ".join(title.ljust(widths[key]) for title, key in columns) + " |")
    print(separator)
    for row in rendered:
        print("| " + " | ".join(row[key].ljust(widths[key]) for _, key in columns) + " |")
    print(separator)


def main() -> None:
    args = _parse_args()
    hardware = _idle_preflight(args.physical_gpu, args.idle_samples, args.idle_interval)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly the --physical-gpu before running this benchmark")

    import torch

    from gptqmodel.quantization.qvq import tail_biting_viterbi_quantize
    from gptqmodel.quantization.qvq_codecs import pgc16_codebook

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("QVQ Viterbi benchmark requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    visible_uuid = f"GPU-{properties.uuid}"
    if visible_uuid.lower() != hardware["uuid"].lower():
        raise RuntimeError(
            f"visible CUDA UUID {visible_uuid} does not match physical GPU {args.physical_gpu} {hardware['uuid']}"
        )
    device = torch.device("cuda:0")
    codebook = pgc16_codebook(device=device, dtype=torch.float32)
    rows: list[dict[str, object]] = []
    for bits in sorted(set(args.bits)):
        maximum_batch = max(args.batch_sizes)
        generator = torch.Generator(device="cpu").manual_seed(20260811 + int(bits * 2))
        sequences = torch.randn((maximum_batch, args.steps, 2), generator=generator, dtype=torch.float32).to(device)
        reference = tail_biting_viterbi_quantize(sequences[:1], codebook, bits=bits)
        torch.cuda.synchronize()
        for batch_size in sorted(set(args.batch_sizes)):
            batch = sequences[:batch_size]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            try:
                timing, result = _timings(
                    torch,
                    lambda batch=batch, bits=bits: tail_biting_viterbi_quantize(batch, codebook, bits=bits),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                path_exact = torch.equal(result.states[:1], reference.states) and torch.equal(
                    result.values[:1], reference.values
                )
                if not path_exact:
                    raise AssertionError(f"W{bits} batch {batch_size} changed the first tile path or values")
                torch.testing.assert_close(
                    result.squared_error[:1],
                    reference.squared_error,
                    atol=2e-4,
                    rtol=2e-5,
                )
                loss_abs_delta = (result.squared_error[:1] - reference.squared_error).abs().max().item()
                peak_allocated = max(torch.cuda.max_memory_allocated(device) - baseline_allocated, 0)
                peak_reserved = max(torch.cuda.max_memory_reserved(device) - baseline_reserved, 0)
                rows.append(
                    {
                        "bits": bits,
                        "batch_size": batch_size,
                        **timing,
                        "tiles_per_second": batch_size * 1000.0 / timing["median_ms"],
                        "peak_allocated_mib": peak_allocated / 1024**2,
                        "peak_reserved_mib": peak_reserved / 1024**2,
                        "loss_abs_delta": loss_abs_delta,
                        "path_exact": path_exact,
                        "status": "ok",
                    }
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "bits": bits,
                        "batch_size": batch_size,
                        "mean_ms": None,
                        "median_ms": None,
                        "p95_ms": None,
                        "tiles_per_second": None,
                        "peak_allocated_mib": None,
                        "peak_reserved_mib": None,
                        "loss_abs_delta": None,
                        "path_exact": False,
                        "status": "oom",
                    }
                )
        del sequences, reference
        torch.cuda.empty_cache()

    report = {
        "environment": {
            "python": os.sys.version.split()[0],
            "gil_enabled": os.sys._is_gil_enabled(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu": args.physical_gpu,
            "uuid": hardware["uuid"],
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sms": properties.multi_processor_count,
            "total_memory": properties.total_memory,
        },
        "settings": {
            "steps": args.steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "rows": rows,
    }
    _print_table(rows)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
