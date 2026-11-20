# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 format: TurboDerp and ExLlamaV3 contributors, MIT, https://github.com/turboderp-org/exllamav3
"""Benchmark the EXL3 INT8 small-batch (m <= 2) decode path on RDNA4.

Compares the two HIP GEMV kernels of ``bc_linear_exl3_run_hip_gemv`` and the
reconstruct + hgemm fallback for random mul1 (cb 2) trellis weights at integer
2 / 3 / 4 bpw:

``fp16``
    the fp16-activation HIP GEMV (``int8_gemv=0``), landed in #5795.
``int8-res``
    the INT8-activation HIP GEMV in residual mode (``int8_gemv=1``): one int8
    byte plus an error-feedback byte per activation, ~15-16 effective bits.
``int8-plain``
    plain int8 activations (``int8_gemv=2``), one byte per activation.
``reconstruct``
    ``LinearEXL3.forward(..., {"reconstruct": True})``, i.e. dequantize the whole
    weight to FP16 and call hipBLAS.

The default matrix is every ``k`` in 2048 / 4096 / 8192 crossed with every ``n``
in 2048 / 4096 / 8192 and every ``m`` in 1 / 2 / 4 / 8.  The INT8 kernel is only
launched for the rows its dispatch admits (residual m == 1, plain m <= 2); the
other rows report the fp16 GEMV that stays in charge and are marked ``n/a`` for
INT8, so the table doubles as the enablement report.

Two timings are reported per path:

``cold us``
    L2 flushed (160 MiB read) before every launch, CUDA-event timed, median of
    20 launches.  This is the DRAM-bound weight-streaming case of a real model,
    where each layer's trellis is read once per token.
``warm us``
    Steady state: 100 back-to-back launches between one event pair.  The
    trellis stays resident in the 96 MiB L2 of the R9700, so this isolates the
    kernel's compute/fixed cost from DRAM bandwidth.

``GB/s`` is the effective bandwidth of the packed trellis, i.e.
``k * n * bits / 8`` bytes divided by the measured time.

    PYTHONPATH=. HIP_VISIBLE_DEVICES=3 \\
        /opt/venvs/rocm101t/bin/python tests/benchmark_exl3_rdna4_gemv_int8.py

Add ``--check`` to compare every path against the reconstructed FP32 weight and
report the max relative error, and ``--json`` to write the measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3


RATES = (2.0, 3.0, 4.0)
KS = (2048, 4096, 8192)
NS = (2048, 4096, 8192)
SHAPES = tuple((k, n) for k in KS for n in NS)
BATCHES = (1, 2, 4, 8)
WARMUPS = 5
REPS = 20
LOOP = 100
PASSES = 2
FLUSH_BYTES = 160 * 1024 * 1024
FP16_DISPATCH = 1
INT8_DISPATCH = 2


def packed_projection(k: int, n: int, bits: float, seed: int = 37):
    """Random packed mul1 trellis plus the per-channel Hadamard signs (as in the tests)."""
    words = int(16 * bits)
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, words), generator=generator, dtype=torch.int16
    ).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def make_input(m: int, k: int, seed: int = 19) -> torch.Tensor:
    return (torch.randn((m, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def make_module(weights, bits: float) -> LinearEXL3:
    trellis, suh, svh = weights
    marker = torch.ones((), dtype=torch.int32, device=trellis.device)
    return LinearEXL3(None, suh.numel(), svh.numel(), suh=suh, svh=svh, trellis=trellis, mul1=marker)


def _launch(op, module, x, y, int8_gemv: int) -> int:
    return op(module.trellis, module.suh, module.svh, float(module.K), None,
              module.mcg, module.mul1, x, y, False, int8_gemv)


def _reconstruct(module, x):
    return module.forward(x, {"reconstruct": True})


def _ramp(op, module, x, y) -> None:
    """Run enough traffic to leave the idle SM clock behind before the first sample."""
    for _ in range(300):
        _launch(op, module, x, y, 2)
    torch.cuda.synchronize()


def _time_us(fn, flush, sink) -> tuple[float, float]:
    """Return (cold us, warm us); best of two passes."""
    cold_passes, warm_passes = [], []
    for _ in range(PASSES):
        for _ in range(WARMUPS):
            fn()
        torch.cuda.synchronize()

        # Cold: the flush kernel runs ahead of each timed launch, so the launch
        # command is already queued when the start event fires (no host gap).
        pairs = []
        for _ in range(REPS):
            torch.sum(flush, 0, out=sink)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        cold_passes.append(statistics.median([start.elapsed_time(end) * 1_000.0 for start, end in pairs]))

        # Warm: back-to-back launches, one event pair around the whole batch.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(LOOP):
            fn()
        end.record()
        end.synchronize()
        warm_passes.append(start.elapsed_time(end) * 1_000.0 / LOOP)
    return min(cold_passes), min(warm_passes)


def _max_rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-6))


def _gbs(packed_bytes: float, us: float | None) -> float | None:
    return None if us is None else packed_bytes / us / 1e3


def run(args) -> list[dict]:
    if not prewarm_exllamav3_extension():
        raise SystemExit("ExLlamaV3 extension is not available")
    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_hip_gemv
    flush = torch.ones(FLUSH_BYTES // 2, dtype=torch.int16, device="cuda")
    sink = torch.zeros((), dtype=torch.int64, device="cuda")
    rows = []
    for bits in args.bits:
        for k, n in args.shapes:
            weights = packed_projection(k, n, bits)
            module = make_module(weights, bits)
            packed_bytes = k * n * bits / 8.0
            for m in args.batches:
                x = make_input(m, k)
                y = torch.empty((m, n), device="cuda", dtype=torch.float32)
                reference = module.forward(x, {"reconstruct": True}) if args.check else None
                _ramp(op, module, x, y)

                results = {}
                for name, mode in (("fp16", 0), ("int8-res", 1), ("int8-plain", 2)):
                    dispatch = _launch(op, module, x, y, mode)
                    torch.cuda.synchronize()
                    expected = INT8_DISPATCH if mode else FP16_DISPATCH
                    if dispatch != expected:
                        results[name] = {"dispatch": dispatch, "cold_us": None, "warm_us": None, "rel": None}
                        continue
                    cold, warm = _time_us(lambda mode=mode: _launch(op, module, x, y, mode), flush, sink)
                    rel = _max_rel(y.view(m, n).float(), reference) if args.check else None
                    results[name] = {"dispatch": dispatch, "cold_us": cold, "warm_us": warm, "rel": rel}

                rec_cold, rec_warm = _time_us(lambda: _reconstruct(module, x), flush, sink)

                row = {"bits": bits, "k": k, "n": n, "m": m, "packed_bytes": packed_bytes,
                       "reconstruct_cold_us": rec_cold, "reconstruct_warm_us": rec_warm,
                       "reconstruct_cold_gbs": _gbs(packed_bytes, rec_cold),
                       "reconstruct_warm_gbs": _gbs(packed_bytes, rec_warm)}
                for name, entry in results.items():
                    row[f"{name}_dispatch"] = entry["dispatch"]
                    row[f"{name}_cold_us"] = entry["cold_us"]
                    row[f"{name}_warm_us"] = entry["warm_us"]
                    row[f"{name}_cold_gbs"] = _gbs(packed_bytes, entry["cold_us"])
                    row[f"{name}_warm_gbs"] = _gbs(packed_bytes, entry["warm_us"])
                    row[f"{name}_rel"] = entry["rel"]
                rows.append(row)
                print(_format(row), flush=True)
    return rows


def _us(value: float | None) -> str:
    return "     n/a" if value is None else f"{value:8.2f}"


def _format(row: dict) -> str:
    line = (f"{row['bits']:.1f}bpw k={row['k']:5d} n={row['n']:5d} m={row['m']:2d} "
            f"| reconstruct {_us(row['reconstruct_cold_us'])}/{_us(row['reconstruct_warm_us'])}")
    for name in ("fp16", "int8-res", "int8-plain"):
        cold, warm = row[f"{name}_cold_us"], row[f"{name}_warm_us"]
        line += f" | {name:10s} {_us(cold)}/{_us(warm)}"
        if row.get(f"{name}_rel") is not None:
            line += f" rel={row[f'{name}_rel']:.1e}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=float, nargs="+", default=list(RATES), help="packed rates (default: 2 3 4)")
    parser.add_argument("--shapes", type=int, nargs="+", default=None,
                        help="k n pairs to run, e.g. --shapes 4096 4096 (default: the full matrix)")
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES), help="m values (default: 1 2 4 8)")
    parser.add_argument("--check", action="store_true", help="compare every path against the reconstructed weight")
    parser.add_argument("--json", default=None, help="write the measurements to this file")
    args = parser.parse_args()
    if args.shapes:
        args.shapes = tuple(tuple(args.shapes[i:i + 2]) for i in range(0, len(args.shapes), 2))
    else:
        args.shapes = SHAPES
    rows = run(args)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
