#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the complete folded gfx950 QVQ layer against its prior execution path."""

from __future__ import annotations

import argparse
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
from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES, REQUESTED_M
from benchmark_qvq_p32_amd_fold_ceiling import SHAPE_AXES

REPO_ROOT = Path(__file__).resolve().parents[1]
P32_RATES = (2.0, 2.5, 3.0, 3.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=P32_RATES)
    parser.add_argument("--m-values", type=int, nargs="+", default=REQUESTED_M)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_folded_full_gfx950.json"),
    )
    args = parser.parse_args()
    if any(rate not in P32_RATES for rate in args.rates):
        parser.error(f"--rates must be drawn from {P32_RATES}")
    if any(m not in REQUESTED_M for m in args.m_values):
        parser.error(f"--m-values must be drawn from {REQUESTED_M}")
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

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_amd import qvq_p32_amd_folded_case_supported

    rows = []
    permitted_pids = set(hardware["process_ids"]) | set(
        _rocm_snapshot(args.physical_gpu)["process_ids"]
    )
    for shape, k, n in QWEN38_27B_SHAPES:
        input_hadamard, output_hadamard = SHAPE_AXES[shape]
        for bits in args.rates:
            generator = torch.Generator(device="cuda").manual_seed(
                20260904 + int(bits * 10) + k + n
            )
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
            inner = layer.get_inner_weight_tensor()

            for m in args.m_values:
                x = (
                    torch.randn((m, k), dtype=torch.float16, device="cuda", generator=generator)
                    * 0.01
                ).contiguous()

                def prior_path(x=x, layer=layer):
                    output = layer._forward_compute_dtype(x, torch.float16)
                    # Reproduce the public pre-fold forward path, including its
                    # composite-width finite reduction and conditional retry.
                    if not torch.isfinite(output).all():
                        output = layer._forward_compute_dtype(x, torch.bfloat16)
                    return output.to(torch.float16)

                def folded_path(x=x, layer=layer):
                    return layer(x)

                expected_prior = prior_path()
                actual = folded_path()
                reference = x.float()
                if input_hadamard:
                    reference = matmul_hadU(reference)
                reference = reference @ inner
                if output_hadamard:
                    reference = matmul_hadU(reference)
                torch.cuda.synchronize()
                folded_selected = qvq_p32_amd_folded_case_supported(m, k, n)
                canonical_accuracy = _errors(actual, reference)
                prior_regression = _errors(actual, expected_prior)
                accuracy = canonical_accuracy if folded_selected else prior_regression
                accuracy_basis = "canonical_fp32" if folded_selected else "exact_prior_path"
                accuracy_pass = accuracy["max_abs"] <= (2e-3 if folded_selected else 0.0)
                snapshot, timing_valid = _timing_recheck(args, permitted_pids)
                valid = valid and timing_valid
                prior = _timings(torch, prior_path, warmup=args.warmup, iterations=args.iterations)
                folded = _timings(torch, folded_path, warmup=args.warmup, iterations=args.iterations)
                speedup = prior["median_ms"] / folded["median_ms"]
                rows.append(
                    {
                        "shape": shape,
                        "bits": bits,
                        "m": m,
                        "k": k,
                        "n": n,
                        "input_hadamard": input_hadamard,
                        "output_hadamard": output_hadamard,
                        "folded_selected": folded_selected,
                        "prior": prior,
                        "folded": folded,
                        "speedup": speedup,
                        "accuracy": accuracy,
                        "accuracy_basis": accuracy_basis,
                        "canonical_accuracy": canonical_accuracy,
                        "prior_regression": prior_regression,
                        "accuracy_pass": accuracy_pass,
                        "timing_snapshot": snapshot,
                    }
                )
                print(
                    f"[{len(rows):03d}] {shape:12s} W{bits:g} M={m:4d} "
                    f"prior={prior['median_ms']:.6f}ms folded={folded['median_ms']:.6f}ms "
                    f"speedup={speedup:.3f}x max_abs={accuracy['max_abs']:.7g} "
                    f"accuracy_pass={accuracy_pass}",
                    flush=True,
                )

            del layer, inner, planar, bank_ids, x, expected_prior, actual, reference
            torch.cuda.empty_cache()

    speedups = [row["speedup"] for row in rows]
    result = {
        "schema": "qvq_p32_amd_folded_full_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": valid,
        "accuracy_gate": 2e-3,
        "model": "Qwen/Qwen3.8-27B",
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip},
        "summary": {
            "cases": len(rows),
            "speedup_min": min(speedups),
            "speedup_geometric_mean": math.exp(statistics.mean(math.log(value) for value in speedups)),
            "speedup_max": max(speedups),
            "max_abs": max(row["accuracy"]["max_abs"] for row in rows),
            "accuracy_passes": sum(row["accuracy_pass"] for row in rows),
            "accuracy_failures": sum(not row["accuracy_pass"] for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
