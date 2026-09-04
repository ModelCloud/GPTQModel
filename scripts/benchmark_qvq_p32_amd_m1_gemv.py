#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare one-pass ROCm GEMV interfaces with the folded M=1 GEMM path."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmark_qvq_p32_amd import (
    _idle_preflight,
    _rocm_snapshot,
    _timing_recheck,
    _timings,
)
from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES
from benchmark_qvq_p32_amd_fold_ceiling import SHAPE_AXES

REPO_ROOT = Path(__file__).resolve().parents[1]
P32_RATES = (2.0, 2.5, 3.0, 3.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=P32_RATES)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_m1_gemv_ab_gfx950.json"),
    )
    args = parser.parse_args()
    if any(rate not in P32_RATES for rate in args.rates):
        parser.error(f"--rates must be drawn from {P32_RATES}")
    if min(args.warmup, args.iterations, args.idle_samples) <= 0:
        parser.error("warmup, iterations, and idle samples must be positive")
    return args


def _errors(actual, reference) -> dict[str, float]:
    difference = actual.float() - reference.float()
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": difference.norm().div(reference.float().norm().clamp_min(1e-12)).item(),
    }


def main() -> None:
    args = _parse_args()
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-mi355x")

    import torch
    from torch.nn import functional

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_amd import (
        _qvq_p32_folded_execute,
        qvq_p32_amd_folded,
        qvq_p32_amd_folded_shape_supported,
    )
    from gptqmodel.utils.qvq_cuda import _pgc16_levels

    rows = []
    permitted_pids = set(hardware["process_ids"]) | set(_rocm_snapshot(args.physical_gpu)["process_ids"])
    for shape, k, n in QWEN38_27B_SHAPES:
        if not qvq_p32_amd_folded_shape_supported(k, n):
            continue
        input_hadamard, output_hadamard = SHAPE_AXES[shape]
        for bits in args.rates:
            generator = torch.Generator(device="cuda").manual_seed(20260904 + int(bits * 10) + k + n)
            tile_count = (k // 16) * (n // 16)
            planar = torch.randint(
                -(1 << 31),
                1 << 31,
                (tile_count, qvq_words_per_tile(bits, vector_size=2)),
                dtype=torch.int32,
                device="cuda",
                generator=generator,
            )
            bank_ids = pack_qvq_binary_bank_ids(
                torch.randint(
                    0,
                    2,
                    (tile_count * 8,),
                    dtype=torch.uint8,
                    device="cuda",
                    generator=generator,
                )
            )
            layer = QVQLinear(
                bits=bits,
                in_features=k,
                out_features=n,
                bank_count=2,
                v2b2_p32=True,
                input_hadamard=input_hadamard,
                output_hadamard=output_hadamard,
                tensors={
                    "trellis": planar,
                    "SU": torch.ones(k, dtype=torch.float32, device="cuda"),
                    "SV": torch.ones(n, dtype=torch.float32, device="cuda"),
                    "bank_ids": bank_ids,
                    "bank_alt_id": torch.tensor([3], dtype=torch.uint8, device="cuda"),
                },
            ).eval()
            x = (torch.randn((1, k), dtype=torch.float16, device="cuda", generator=generator) * 0.01).contiguous()

            # Build the production cache before timing, then reuse its exact storage
            # through both the row-major N-by-K and transposed K-by-N views.
            layer(x)
            window, prepared_bank_ids, bank_alt_id = layer._prepare_amd_p32_metadata(x.device)
            _, folded_nk, operand_kn = window._qvq_p32_amd_folded_cache
            levels = _pgc16_levels(x.device, layer.codebook_version)
            su = layer._cached_cast("SU", torch.float16)
            sv = layer._cached_cast("SV", torch.float16)
            inner = layer.get_inner_weight_tensor()
            reference = x.float()
            if input_hadamard:
                reference = matmul_hadU(reference)
            reference = reference @ inner
            if output_hadamard:
                reference = matmul_hadU(reference)

            candidates = {
                "production": lambda layer=layer, x=x: layer(x),
                "folded_helper": functools.partial(
                    qvq_p32_amd_folded,
                    x,
                    window,
                    levels,
                    prepared_bank_ids,
                    su,
                    sv,
                    bits,
                    out_features=n,
                    bank_alt_id=bank_alt_id,
                    input_hadamard=input_hadamard,
                    output_hadamard=output_hadamard,
                    output_fp32=False,
                ),
                "folded_execute": functools.partial(
                    _qvq_p32_folded_execute,
                    x,
                    operand_kn,
                    out_features=n,
                    output_fp32=False,
                ),
                "mm_fp16": lambda x=x, operand_kn=operand_kn: torch.mm(
                    x, operand_kn, out_dtype=torch.float16
                ),
                "mv": lambda folded_nk=folded_nk, x=x, n=n: torch.mv(folded_nk, x[0]).view(1, n),
                "matmul": lambda folded_nk=folded_nk, x=x, n=n: torch.matmul(folded_nk, x[0]).view(1, n),
                "linear": lambda x=x, folded_nk=folded_nk: functional.linear(x, folded_nk),
            }
            outputs = {name: candidate() for name, candidate in candidates.items()}
            torch.cuda.synchronize()
            snapshot, timing_valid = _timing_recheck(args, permitted_pids)
            valid = valid and timing_valid
            timings = {
                name: _timings(torch, candidate, warmup=args.warmup, iterations=args.iterations)
                for name, candidate in candidates.items()
            }
            accuracy = {name: _errors(output, reference) for name, output in outputs.items()}
            regression = {name: _errors(output, outputs["production"]) for name, output in outputs.items()}
            production_ms = timings["production"]["median_ms"]
            speedups = {name: production_ms / timing["median_ms"] for name, timing in timings.items()}
            row = {
                "shape": shape,
                "bits": bits,
                "m": 1,
                "k": k,
                "n": n,
                "input_hadamard": input_hadamard,
                "output_hadamard": output_hadamard,
                "timings": timings,
                "speedups_vs_production": speedups,
                "accuracy": accuracy,
                "regression_vs_production": regression,
                "accuracy_pass": {name: error["max_abs"] <= 2e-3 for name, error in accuracy.items()},
                "timing_snapshot": snapshot,
            }
            rows.append(row)
            best = min(timings, key=lambda name: timings[name]["median_ms"])
            print(
                f"[{len(rows):02d}] {shape:12s} W{bits:g} production={production_ms * 1e3:.3f}us "
                f"best={best}:{timings[best]['median_ms'] * 1e3:.3f}us "
                f"speedup={speedups[best]:.3f}x max_abs={accuracy[best]['max_abs']:.7g}",
                flush=True,
            )
            del layer, x, window, folded_nk, operand_kn, inner, reference, outputs, planar, bank_ids
            torch.cuda.empty_cache()

    candidates = tuple(rows[0]["timings"])
    summary = {}
    for candidate in candidates:
        candidate_speedups = [row["speedups_vs_production"][candidate] for row in rows]
        summary[candidate] = {
            "median_us_geometric_mean": 1e3
            * math.exp(statistics.mean(math.log(row["timings"][candidate]["median_ms"]) for row in rows)),
            "speedup_min": min(candidate_speedups),
            "speedup_geometric_mean": math.exp(statistics.mean(math.log(value) for value in candidate_speedups)),
            "speedup_max": max(candidate_speedups),
            "max_abs": max(row["accuracy"][candidate]["max_abs"] for row in rows),
            "accuracy_passes": sum(row["accuracy_pass"][candidate] for row in rows),
        }
    result = {
        "schema": "qvq_p32_amd_m1_gemv_ab_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": valid,
        "accuracy_gate": 2e-3,
        "model": "Qwen/Qwen3.8-27B",
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library()),
        },
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
