#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Reproducible EXL3 forward benchmark for ROCm / RDNA4 (gfx12xx).

The default grid is the one requested in issue #5710::

    size_m  {1, 8, 32, 128}
    size_k  {2048, 4096, 8192}
    size_n  {2048, 4096, 8192, 16384}
    bpw     {2, 3, 4}

Backends are feature-detected, never hard-coded:

``torch``
    ``ExllamaV3TorchLinear``: torch trellis decode + hipBLAS matmul.  This is
    what AUTO loads for EXL3 on ROCm today.
``reconstruct``
    The explicit reference oracle: ``ext.reconstruct`` to a dense FP16 weight,
    then ``ext.hgemm`` (hipBLAS) and the output Hadamard.  Native kernels must
    beat this path.  It is forced through the ext ops so it stays available
    after a native kernel is wired into ``LinearEXL3.forward``.
``module``
    ``LinearEXL3.forward`` exactly as the product dispatches it.  On today's
    ROCm build that is reconstruct+hgemm; once a native kernel is integrated
    this row automatically becomes the native path.
``native``
    ``module.bc.run``: the packed tensor-core/GEMV entry point.  The probe
    calls it once; today it raises on HIP (``exl3_gemm`` is CUDA-only), so the
    row is skipped with the reason.  When the RDNA4 kernels land it starts
    being timed without a harness change.

Tensors are synthetic packed trellises with the same geometry as a quantized
projection; the benchmark measures inference, not quantization.  Every timed
region is bracketed by ``torch.cuda.synchronize()``, each row is warm, and the
default is 20 timed reps.  A row whose median moves more than 20% between the
first block and a short confirmation block is re-measured before it is
reported, because three other agents share the GPUs.

Example::

    HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD HF_HUB_OFFLINE=1 \
    GPTQMODEL_EXLLAMAV3_BUILD_ROOT=$PWD/.extbuild MAX_JOBS=8 \
    /opt/venvs/rocm101t/bin/python tests/benchmark_exl3_rocm_rdna4.py \
        --device cuda:0 --reps 20 --json /tmp/exl3_rdna4.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache, partial

import torch


DEFAULT_SIZES_M = (1, 8, 32, 128)
DEFAULT_SIZES_K = (2048, 4096, 8192)
DEFAULT_SIZES_N = (2048, 4096, 8192, 16384)
DEFAULT_BITS = (2, 3, 4)
BACKENDS = ("torch", "reconstruct", "module", "native")
CONTENTION_THRESHOLD = 0.20


def _gcn_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName)
    except Exception:
        return ""


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class _ExtensionStatus:
    available: bool
    reason: str


@lru_cache(maxsize=1)
def _extension_status() -> _ExtensionStatus:
    try:
        from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

        if prewarm_exllamav3_extension():
            return _ExtensionStatus(True, "")
        return _ExtensionStatus(False, "prewarm_exllamav3_extension() returned False")
    except Exception as exc:
        return _ExtensionStatus(False, f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}")


def _packed_projection(size_k: int, size_n: int, bits: float, device: torch.device) -> dict:
    """Deterministic packed EXL3 projection with the requested geometry."""

    words = int(16 * bits)
    seed = 1000 + size_k * 7 + size_n * 13 + bits * 31
    trellis = torch.randint(
        -32768,
        32768,
        (size_k // 16, size_n // 16, words),
        dtype=torch.int16,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    suh = (
        torch.randn(size_k, generator=torch.Generator().manual_seed(seed + 1)) * 0.1
    ).half().to(device)
    svh = (
        torch.randn(size_n, generator=torch.Generator().manual_seed(seed + 2)) * 0.1
    ).half().to(device)
    mcg = torch.tensor(0xCBAC1FED, dtype=torch.uint32).view(torch.int32).to(device)
    return {"trellis": trellis, "suh": suh, "svh": svh, "mcg": mcg}


def _input(size_m: int, size_k: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(2000 + size_m * 3 + size_k)
    return (torch.randn(size_m, size_k, device=device, generator=generator) * 0.1).half()


def _reconstruct_forward(inner, x: torch.Tensor) -> torch.Tensor:
    """Force the pre-native oracle: ext.reconstruct + ext.hgemm + Hadamard."""

    from gptqmodel.exllamav3.ext import exllamav3_ext as ext

    x = x.reshape(-1, inner.in_features).contiguous()
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, inner.suh, None, 1.0)
    weight = inner.get_inner_weight_tensor()
    y = torch.empty((x.shape[0], inner.out_features), dtype=torch.float32, device=x.device)
    ext.hgemm(xh, weight, y)
    ext.had_r_128(y, y, None, inner.svh, 1.0)
    if inner.bias is not None:
        y.add_(inner.bias)
    return y


def _module_forward(inner, x: torch.Tensor) -> torch.Tensor:
    return inner.forward(x, {})


def _native_forward(inner, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty((x.shape[0], inner.out_features), dtype=torch.float16, device=x.device)
    inner.bc.run(x, y, include_bias=inner.bias is not None)
    return y


def _probe_native(inner, x: torch.Tensor) -> tuple[bool, str]:
    try:
        _native_forward(inner, x)
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


def _measure(fn, *, warmup: int, reps: int, stability: float = CONTENTION_THRESHOLD) -> dict:
    """Warm, time ``reps`` synchronized regions, and re-check contended rows."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def block(count: int) -> list[float]:
        samples: list[float] = []
        for _ in range(count):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        return samples

    samples = block(reps)
    median_s = statistics.median(samples)
    block_medians = [median_s]
    rounds = 1
    check_count = max(5, reps // 4)
    check = block(check_count)
    check_median_s = statistics.median(check)
    block_medians.append(check_median_s)
    contended = abs(check_median_s - median_s) / max(min(check_median_s, median_s), 1e-12) > stability
    if contended:
        # Re-measure the full block once; report the least-contended median.
        extra = block(reps)
        rounds = 2
        extra_median_s = statistics.median(extra)
        block_medians.append(extra_median_s)
        if extra_median_s <= median_s:
            samples = extra
            median_s = extra_median_s
    spread = (max(block_medians) - min(block_medians)) / max(min(block_medians), 1e-12)
    return {
        "median_ms": median_s * 1e3,
        "min_ms": min(samples) * 1e3,
        "spread_pct": spread * 100.0,
        "reps": len(samples),
        "rounds": rounds,
        "contended": contended,
    }


def _tflops(size_m: int, size_k: int, size_n: int, median_ms: float) -> float:
    return 2.0 * size_m * size_k * size_n / (median_ms * 1e-3) / 1e12


def _print_header(args, device: torch.device) -> None:
    print("# EXL3 ROCm RDNA4 benchmark")
    print()
    print(f"- commit: `{_commit()}`")
    print(f"- torch: `{torch.__version__}` (HIP `{torch.version.hip}`)")
    print(f"- device: `{torch.cuda.get_device_name(device)}` ({_gcn_arch() or 'unknown arch'})")
    print(f"- HIP_VISIBLE_DEVICES: `{os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}`")
    print(f"- grid: m={list(args.sizes_m)} k={list(args.sizes_k)} n={list(args.sizes_n)} bpw={list(args.bits)}")
    print(f"- timing: warmup={args.warmup}, reps={args.reps}, sync around every region")
    print()


def _print_table(rows: list[dict]) -> None:
    print("| m | k | n | bpw | backend | median_ms | min_ms | tflops | spread_pct | rounds |")
    print("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['size_m']} | {row['size_k']} | {row['size_n']} | {row['bits']} | "
            f"{row['backend']} | {row['median_ms']:.4f} | {row['min_ms']:.4f} | "
            f"{row['tflops']:.3f} | {row['spread_pct']:.1f} | {row['rounds']} |"
        )


def _print_pivot(rows: list[dict], backend: str, bits: int) -> None:
    selected = [row for row in rows if row["backend"] == backend and row["bits"] == bits]
    if not selected:
        return
    print()
    print(f"## Pivot: `{backend}`, {bits} bpw (median ms)")
    print()
    columns: list[tuple[int, int]] = []
    for row in selected:
        key = (row["size_k"], row["size_n"])
        if key not in columns:
            columns.append(key)
    print("| m \\ k,n | " + " | ".join(f"{size_k}x{size_n}" for size_k, size_n in columns) + " |")
    print("|---|" + "---:|" * len(columns))
    for size_m in sorted({row["size_m"] for row in selected}):
        cells = []
        for size_k, size_n in columns:
            match = next(
                row
                for row in selected
                if row["size_m"] == size_m and row["size_k"] == size_k and row["size_n"] == size_n
            )
            cells.append(f"{match['median_ms']:.4f}")
        print(f"| {size_m} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="CUDA/HIP device ordinal (default: 0)")
    parser.add_argument("--sizes-m", type=_parse_ints, default=DEFAULT_SIZES_M)
    parser.add_argument("--sizes-k", type=_parse_ints, default=DEFAULT_SIZES_K)
    parser.add_argument("--sizes-n", type=_parse_ints, default=DEFAULT_SIZES_N)
    parser.add_argument("--bits", type=_parse_floats, default=DEFAULT_BITS, help="EXL3 bpw (mcg integer rates)")
    parser.add_argument("--reps", type=int, default=20, help="timed reps per row (minimum 20)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", *BACKENDS),
        help="backend to time; auto feature-detects every available backend",
    )
    parser.add_argument("--json", dest="json_path", default=None, help="write the raw rows and metadata as JSON")
    parser.add_argument("--pivot", action="store_true", help="also print a compact median-ms pivot table")
    parser.add_argument("--pivot-backend", default="reconstruct", choices=BACKENDS)
    parser.add_argument("--pivot-bpw", type=int, default=3)
    args = parser.parse_args(argv)

    if args.reps < 20:
        print("error: --reps must be >= 20 (the harness contract)", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("error: EXL3 ROCm benchmark requires a CUDA/HIP device", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.device)
    if torch.cuda.current_device() != args.device:
        print(f"error: failed to pin device {args.device}", file=sys.stderr)
        return 2
    device = torch.device("cuda", args.device)
    if not _gcn_arch().startswith("gfx12"):
        print(
            f"warning: {_gcn_arch() or 'unknown arch'} is not RDNA4 (gfx12xx); numbers are not RDNA4 numbers",
            file=sys.stderr,
        )

    requested = BACKENDS if args.backend == "auto" else (args.backend,)
    extension = _extension_status()
    available: dict[str, bool] = {"torch": True}
    reasons: dict[str, str] = {}
    if extension.available:
        for name in ("reconstruct", "module", "native"):
            available[name] = name in requested
    else:
        for name in ("reconstruct", "module", "native"):
            available[name] = False
            reasons[name] = f"EXL3 JIT extension unavailable: {extension.reason}"
    if "torch" in requested:
        available["torch"] = True

    _print_header(args, device)
    if not extension.available:
        print(f"> `reconstruct` / `module` / `native` skipped: {extension.reason}")
        print()

    rows: list[dict] = []
    printed_native_skips: set[str] = set()
    for bits in args.bits:
        rate = int(bits) if float(bits).is_integer() else float(bits)
        for size_k in args.sizes_k:
            for size_n in args.sizes_n:
                packed = _packed_projection(size_k, size_n, rate, device)
                torch_module = None
                inner_module = None
                if available.get("torch") or available.get("reconstruct") or available.get("module") or available.get("native"):
                    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

                    torch_module = ExllamaV3TorchLinear.from_tensors(
                        in_features=size_k,
                        out_features=size_n,
                        name="exl3_rdna4_bench",
                        tensors=packed,
                    )
                if extension.available:
                    from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3

                    inner_module = LinearEXL3(
                        config=None,
                        in_features=size_k,
                        out_features=size_n,
                        suh=packed["suh"],
                        svh=packed["svh"],
                        trellis=packed["trellis"],
                        mcg=packed.get("mcg"),
                        bias=None,
                        out_dtype=torch.float16,
                    )

                native_ok = False
                if available.get("native") and inner_module is not None:
                    probe_x = _input(1, size_k, device)
                    native_ok, native_reason = _probe_native(inner_module, probe_x)
                    del probe_x
                    if not native_ok and native_reason not in printed_native_skips:
                        printed_native_skips.add(native_reason)
                        print(f"[bench] native skipped (all shapes): {native_reason}")

                for size_m in args.sizes_m:
                    x = _input(size_m, size_k, device)
                    for backend in BACKENDS:
                        if backend not in requested:
                            continue
                        if backend == "native":
                            if not (available.get("native") and native_ok):
                                continue
                        elif not available.get(backend, False):
                            continue
                        if backend == "torch":
                            fn = partial(torch_module, x)
                        elif backend == "reconstruct":
                            fn = partial(_reconstruct_forward, inner_module, x)
                        elif backend == "module":
                            fn = partial(_module_forward, inner_module, x)
                        elif backend == "native":
                            fn = partial(_native_forward, inner_module, x)
                        else:  # pragma: no cover - guarded by argparse
                            raise AssertionError(backend)

                        timing = _measure(fn, warmup=args.warmup, reps=args.reps)
                        row = {
                            "size_m": size_m,
                            "size_k": size_k,
                            "size_n": size_n,
                            "bits": bits,
                            "backend": backend,
                            **timing,
                            "tflops": _tflops(size_m, size_k, size_n, timing["median_ms"]),
                        }
                        rows.append(row)
                        if timing["contended"]:
                            print(
                                f"[bench] contention: m={size_m} k={size_k} n={size_n} "
                                f"bpw={bits} {backend} re-measured (spread {timing['spread_pct']:.1f}%)"
                            )
                    del x
                del packed, torch_module, inner_module
                torch.cuda.empty_cache()

    print()
    _print_table(rows)
    if args.pivot:
        _print_pivot(rows, args.pivot_backend, args.pivot_bpw)

    if args.json_path:
        payload = {
            "metadata": {
                "commit": _commit(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(device),
                "gcn_arch": _gcn_arch(),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "warmup": args.warmup,
                "reps": args.reps,
                "grid": {
                    "size_m": list(args.sizes_m),
                    "size_k": list(args.sizes_k),
                    "size_n": list(args.sizes_n),
                    "bits": list(args.bits),
                },
                "extension": {
                    "available": extension.available,
                    "reason": extension.reason,
                },
                "backends": {name: available.get(name, False) for name in BACKENDS},
            },
            "rows": rows,
        }
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {len(rows)} rows to `{args.json_path}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
