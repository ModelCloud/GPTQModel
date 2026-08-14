"""Historical learned-PGC16-v2 Apple benchmark retained for research replay.

HYB is an explicit regression baseline, not a loadable checkpoint format.
The W8 Pangolin/int8 row is comparison-only and never assumes QVQ should win.
Production QVQ no longer exposes learned v2, so this script is not part of the
supported benchmark surface and may require its historical source revision.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def request_performance_qos() -> bool:
    """Request Darwin user-interactive QoS so host work uses performance cores."""

    if platform.system() != "Darwin":
        return True
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    set_qos = libsystem.pthread_set_qos_class_self_np
    set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    set_qos.restype = ctypes.c_int
    return set_qos(0x21, 0) == 0


def parse_shape(value: str) -> tuple[int, int]:
    try:
        k_text, n_text = value.lower().split("x", 1)
        k, n = int(k_text), int(n_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shape must be KxN, for example 2048x8192") from exc
    if k <= 0 or n <= 0 or k % 32 or n % 32:
        raise argparse.ArgumentTypeError("K and N must be positive multiples of 32")
    return k, n


def timing_summary(milliseconds: list[float]) -> dict[str, float]:
    ordered = sorted(milliseconds)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def bench_mps(
    fn: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(inner_iterations):
            fn()
        torch.mps.synchronize()
        timings.append((time.perf_counter() - started) * 1000 / inner_iterations)
    return timing_summary(timings)


def bench_mlx(
    fn: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
    inner_iterations: int,
) -> dict[str, float]:
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(fn())
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        outputs = [fn() for _ in range(inner_iterations)]
        mx.eval(*outputs)
        timings.append((time.perf_counter() - started) * 1000 / inner_iterations)
    return timing_summary(timings)


def random_trellis(torch, *, bits: int, k: int, n: int):
    generator = torch.Generator().manual_seed(20262000 + bits * 100 + k + n)
    shape = ((k // 16) * (n // 16), 8 * bits)
    return torch.randint(-(1 << 31), (1 << 31) - 1, shape, generator=generator, dtype=torch.int32)


def pangolin_operands(torch, *, m: int, k: int, n: int):
    generator = torch.Generator().manual_seed(20262100 + m + k + n)
    groups = k // 128
    return (
        torch.randn((m, k), generator=generator).to(torch.float16),
        torch.randint(-(1 << 31), (1 << 31) - 1, (k // 4, n), generator=generator, dtype=torch.int32),
        (torch.rand((groups, n), generator=generator) * 0.02 + 0.005).to(torch.float16),
        torch.randint(-(1 << 31), (1 << 31) - 1, (groups, n // 4), generator=generator, dtype=torch.int32),
        (torch.arange(k, dtype=torch.int32) // 128),
    )


def add_bandwidth(entry: dict[str, float], *, bits: int, k: int, n: int) -> None:
    payload_bytes = k * n * bits / 8
    entry["payload_gbytes_per_second"] = payload_bytes / (entry["median_ms"] / 1000) / 1e9


def qvq_v2_level_bits(torch) -> tuple[int, ...]:
    """Return a deterministic non-Gaussian table for value-independent kernel timing."""

    from gptqmodel.quantization.qvq_codecs import PGC16_LEVEL_COUNT
    from gptqmodel.quantization.qvq_codecs.deprecated import freeze_pgc16_level_bits

    levels = torch.linspace(-3.5, 3.5, PGC16_LEVEL_COUNT)
    return freeze_pgc16_level_bits(levels)


def benchmark_mps(args, report: dict[str, Any]) -> None:
    import torch

    from gptqmodel.quantization.qvq_codecs.deprecated import QVQ_LEARNED_CODEBOOK_VERSION
    from gptqmodel.quantization.qvq_codecs.hyb_reference import canonical_hyb_lut
    from gptqmodel.utils.pangolin_mps import pangolin_mps_gemv, pangolin_mps_supported
    from gptqmodel.utils.qvq_mps import (
        _prepare_qvq_mps_compander,
        qvq_hyb_reference_mps_gemv,
        qvq_mps_gemv,
        qvq_mps_supported,
    )

    if not qvq_mps_supported():
        raise RuntimeError("PyTorch MPS runtime Metal shaders are unavailable")
    device = torch.device("mps")
    compander_bits = qvq_v2_level_bits(torch)
    pgc_compander = _prepare_qvq_mps_compander(device, "pgc16-v1", None)
    qvq_compander = _prepare_qvq_mps_compander(
        device,
        QVQ_LEARNED_CODEBOOK_VERSION,
        compander_bits,
    )
    hyb_lut = canonical_hyb_lut().to(dtype=torch.float16, device=device).contiguous()
    rows = []
    for k, n in args.shape:
        trellises = {bits: random_trellis(torch, bits=bits, k=k, n=n).to(device) for bits in args.bits}
        for m in args.m:
            x = torch.randn((m, k), generator=torch.Generator().manual_seed(20262200 + m + k)).half().to(device)
            for bits in args.bits:
                trellis = trellises[bits]
                pgc = bench_mps(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_mps_gemv(
                        x,
                        trellis,
                        bits,
                        out_features=n,
                        _prepared_compander=pgc_compander,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                qvq = bench_mps(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_mps_gemv(
                        x,
                        trellis,
                        bits,
                        out_features=n,
                        codebook_version=QVQ_LEARNED_CODEBOOK_VERSION,
                        compander_bits=compander_bits,
                        _prepared_compander=qvq_compander,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                hyb = bench_mps(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_hyb_reference_mps_gemv(
                        x,
                        trellis,
                        hyb_lut,
                        bits,
                        out_features=n,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                add_bandwidth(pgc, bits=bits, k=k, n=n)
                add_bandwidth(qvq, bits=bits, k=k, n=n)
                add_bandwidth(hyb, bits=bits, k=k, n=n)
                rows.append(
                    {
                        "backend": "mps",
                        "bits": bits,
                        "m": m,
                        "k": k,
                        "n": n,
                        "pgc16": pgc,
                        "qvq_v2": qvq,
                        "hyb_reference": hyb,
                        "pgc_over_hyb": pgc["median_ms"] / hyb["median_ms"],
                        "qvq_v2_over_v1": qvq["median_ms"] / pgc["median_ms"],
                        "qvq_v2_within_5_percent_of_v1": qvq["median_ms"] <= pgc["median_ms"] * 1.05,
                        "w2_w5_within_5_percent": bits > 5 or pgc["median_ms"] <= hyb["median_ms"] * 1.05,
                    }
                )

            if 8 in args.bits and pangolin_mps_supported():
                operands = tuple(value.to(device) for value in pangolin_operands(torch, m=m, k=k, n=n))
                pangolin = bench_mps(
                    lambda operands=operands: pangolin_mps_gemv(
                        *operands,
                        8,
                        planar=False,
                        _g_idx_validated=True,
                        _g_idx_block_uniform=True,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                add_bandwidth(pangolin, bits=8, k=k, n=n)
                pgc_row = next(
                    row for row in reversed(rows) if row["bits"] == 8 and row["m"] == m and row["k"] == k
                )
                pgc_row["pangolin_int8"] = pangolin
                pgc_row["pgc_over_pangolin"] = pgc_row["pgc16"]["median_ms"] / pangolin["median_ms"]
                pgc_row["pangolin_comparison_only"] = True
    report["rows"].extend(rows)


def benchmark_mlx(args, report: dict[str, Any]) -> None:
    import mlx.core as mx
    import torch

    from gptqmodel.quantization.qvq_codecs.deprecated import QVQ_LEARNED_CODEBOOK_VERSION
    from gptqmodel.quantization.qvq_codecs.hyb_reference import canonical_hyb_lut
    from gptqmodel.utils.pangolin_mlx import pangolin_mlx_gemv
    from gptqmodel.utils.qvq_mlx import (
        _prepare_qvq_mlx_compander,
        qvq_hyb_reference_mlx_gemv,
        qvq_mlx_gemv,
    )

    def mlx(tensor):
        return mx.array(tensor.numpy())

    hyb_lut = mlx(canonical_hyb_lut().to(torch.float16))
    compander_bits = qvq_v2_level_bits(torch)
    pgc_compander = _prepare_qvq_mlx_compander("pgc16-v1", None)
    qvq_compander = _prepare_qvq_mlx_compander(
        QVQ_LEARNED_CODEBOOK_VERSION,
        compander_bits,
    )
    rows = []
    for k, n in args.shape:
        trellises = {bits: mlx(random_trellis(torch, bits=bits, k=k, n=n)) for bits in args.bits}
        for m in args.m:
            x = mlx(torch.randn((m, k), generator=torch.Generator().manual_seed(20262300 + m + k)).half())
            for bits in args.bits:
                trellis = trellises[bits]
                pgc = bench_mlx(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_mlx_gemv(
                        x,
                        trellis,
                        bits,
                        out_features=n,
                        _prepared_compander=pgc_compander,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                qvq = bench_mlx(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_mlx_gemv(
                        x,
                        trellis,
                        bits,
                        out_features=n,
                        codebook_version=QVQ_LEARNED_CODEBOOK_VERSION,
                        compander_bits=compander_bits,
                        _prepared_compander=qvq_compander,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                hyb = bench_mlx(
                    lambda x=x, trellis=trellis, bits=bits, n=n: qvq_hyb_reference_mlx_gemv(
                        x,
                        trellis,
                        hyb_lut,
                        bits,
                        out_features=n,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                add_bandwidth(pgc, bits=bits, k=k, n=n)
                add_bandwidth(qvq, bits=bits, k=k, n=n)
                add_bandwidth(hyb, bits=bits, k=k, n=n)
                rows.append(
                    {
                        "backend": "mlx",
                        "bits": bits,
                        "m": m,
                        "k": k,
                        "n": n,
                        "pgc16": pgc,
                        "qvq_v2": qvq,
                        "hyb_reference": hyb,
                        "pgc_over_hyb": pgc["median_ms"] / hyb["median_ms"],
                        "qvq_v2_over_v1": qvq["median_ms"] / pgc["median_ms"],
                        "qvq_v2_within_5_percent_of_v1": qvq["median_ms"] <= pgc["median_ms"] * 1.05,
                        "w2_w5_within_5_percent": bits > 5 or pgc["median_ms"] <= hyb["median_ms"] * 1.05,
                    }
                )

            if 8 in args.bits:
                operands = tuple(mlx(value) for value in pangolin_operands(torch, m=m, k=k, n=n))
                pangolin = bench_mlx(
                    lambda operands=operands: pangolin_mlx_gemv(
                        *operands,
                        8,
                        planar=False,
                        _g_idx_validated=True,
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    inner_iterations=args.inner_iterations,
                )
                add_bandwidth(pangolin, bits=8, k=k, n=n)
                pgc_row = next(
                    row for row in reversed(rows) if row["bits"] == 8 and row["m"] == m and row["k"] == k
                )
                pgc_row["pangolin_int8"] = pangolin
                pgc_row["pgc_over_pangolin"] = pgc_row["pgc16"]["median_ms"] / pangolin["median_ms"]
                pgc_row["pangolin_comparison_only"] = True
    report["rows"].extend(rows)


def machine_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mps", "mlx", "both"), default="both")
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
        help="KxN; repeat for multiple shapes",
    )
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--bits", type=int, nargs="+", default=list(range(2, 9)))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--inner-iterations", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    args.shape = args.shape or [(2048, 2048), (2048, 8192), (8192, 2048)]

    if any(value <= 0 for value in (*args.m, args.warmup, args.samples, args.inner_iterations)):
        parser.error("M, warmup, samples, and inner iterations must be positive")
    invalid_bits = sorted(set(args.bits) - set(range(2, 9)))
    if invalid_bits:
        parser.error(f"bits must be W2 through W8; got {invalid_bits}")
    if not request_performance_qos():
        raise RuntimeError("failed to request performance-core QoS")

    report: dict[str, Any] = {
        "settings": {
            "backend": args.backend,
            "shapes": args.shape,
            "m": args.m,
            "bits": args.bits,
            "warmup": args.warmup,
            "samples": args.samples,
            "inner_iterations": args.inner_iterations,
            "performance_qos": "user-interactive",
            "machine": machine_name(),
        },
        "rows": [],
    }
    if args.backend in ("mps", "both"):
        benchmark_mps(args, report)
    if args.backend in ("mlx", "both"):
        benchmark_mlx(args, report)

    gated_rows = [row for row in report["rows"] if row["bits"] <= 5]
    report["w2_w5_within_5_percent"] = bool(gated_rows) and all(
        row["w2_w5_within_5_percent"] for row in gated_rows
    )
    report["qvq_v2_within_5_percent_of_v1"] = bool(gated_rows) and all(
        row["qvq_v2_within_5_percent_of_v1"] for row in gated_rows
    )
    print(" backend | bits | M | KxN       | PGC-v1 ms | QVQ-v2 ms | v2/v1 | HYB ms  | gate | Pangolin W8 ms")
    print("---------+------+---+-----------+-----------+-----------+-------+---------+------+----------------")
    for row in report["rows"]:
        pangolin = row.get("pangolin_int8", {}).get("median_ms")
        pangolin_text = "-" if pangolin is None else f"{pangolin:.4f}"
        print(
            f" {row['backend']:>7} | {row['bits']:>4} | {row['m']:>2} | {row['k']}x{row['n']:<5} | "
            f"{row['pgc16']['median_ms']:9.4f} | {row['qvq_v2']['median_ms']:9.4f} | "
            f"{row['qvq_v2_over_v1']:5.3f} | {row['hyb_reference']['median_ms']:7.4f} | "
            f"{'PASS' if row['qvq_v2_within_5_percent_of_v1'] else 'FAIL':>4} | {pangolin_text}"
        )
    print(f"W2-W5 <= 1.05x HYB: {'PASS' if report['w2_w5_within_5_percent'] else 'FAIL'}")
    print(
        "QVQ-v2 W2-W5 <= 1.05x PGC16-v1: "
        f"{'PASS' if report['qvq_v2_within_5_percent_of_v1'] else 'FAIL'}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    gates_pass = report["w2_w5_within_5_percent"] and report["qvq_v2_within_5_percent_of_v1"]
    if not gates_pass and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
