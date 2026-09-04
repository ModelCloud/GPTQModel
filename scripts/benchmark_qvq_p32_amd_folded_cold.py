#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure folded-cache first-use cost and amortization on gfx950."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark_qvq_p32_amd import _idle_preflight, _rocm_snapshot, _timing_recheck
from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES
from benchmark_qvq_p32_amd_fold_ceiling import SHAPE_AXES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--bits", type=float, default=3.0, choices=(2.0, 2.5, 3.0, 3.5))
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_folded_full_certified_gfx950.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mi355x_p32/qwen38_27b_folded_cold_gfx950.json"),
    )
    return parser.parse_args()


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
    from gptqmodel.utils.qvq_amd import (
        qvq_p32_amd_folded_case_supported,
        qvq_p32_amd_folded_shape_supported,
    )

    baseline = json.loads(args.baseline.read_text())
    baseline_rows = {
        (row["shape"], row["bits"], row["m"]): row for row in baseline["rows"]
    }
    permitted_pids = set(hardware["process_ids"]) | set(
        _rocm_snapshot(args.physical_gpu)["process_ids"]
    )
    rows = []
    for shape, k, n in QWEN38_27B_SHAPES:
        if not qvq_p32_amd_folded_shape_supported(k, n):
            continue
        if not qvq_p32_amd_folded_case_supported(args.m, k, n):
            continue
        input_hadamard, output_hadamard = SHAPE_AXES[shape]
        generator = torch.Generator(device="cuda").manual_seed(20260904 + int(args.bits * 10) + k + n)
        tile_count = (k // 16) * (n // 16)
        planar = torch.randint(
            -(1 << 31),
            1 << 31,
            (tile_count, qvq_words_per_tile(args.bits, vector_size=2)),
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
            bits=args.bits,
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
        x = (torch.randn((args.m, k), dtype=torch.float16, device="cuda", generator=generator) * 0.01).contiguous()
        snapshot, timing_valid = _timing_recheck(args, permitted_pids)
        valid = valid and timing_valid
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        output = layer(x)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1e3
        device_ms = start_event.elapsed_time(end_event)

        window = layer._qvq_cuda_window_cache[3]
        _, folded, operand, _, _ = window._qvq_p32_amd_folded_cache
        assert operand.untyped_storage().data_ptr() == folded.untyped_storage().data_ptr()
        assert window._qvq_p32_amd_dense_cache is None
        cache_bytes = folded.numel() * folded.element_size()
        del layer, output, window, folded, operand
        torch.cuda.empty_cache()

        # Repeat the same construction in-process to separate one-time Triton
        # compilation from the actual decode/transform/cache cost.
        repeat_layer = QVQLinear(
            bits=args.bits,
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
        torch.cuda.synchronize()
        repeat_wall_start = time.perf_counter()
        repeat_start_event = torch.cuda.Event(enable_timing=True)
        repeat_end_event = torch.cuda.Event(enable_timing=True)
        repeat_start_event.record()
        repeat_output = repeat_layer(x)
        repeat_end_event.record()
        torch.cuda.synchronize()
        process_warm_wall_ms = (time.perf_counter() - repeat_wall_start) * 1e3
        process_warm_device_ms = repeat_start_event.elapsed_time(repeat_end_event)
        repeat_window = repeat_layer._qvq_cuda_window_cache[3]
        assert repeat_window._qvq_p32_amd_dense_cache is None

        measured = baseline_rows[(shape, args.bits, args.m)]
        prior_ms = measured["prior"]["median_ms"]
        hot_ms = measured["folded"]["median_ms"]
        saved_ms = prior_ms - hot_ms
        row = {
            "shape": shape,
            "bits": args.bits,
            "m": args.m,
            "k": k,
            "n": n,
            "cold_wall_ms": wall_ms,
            "cold_device_interval_ms": device_ms,
            "process_warm_build_wall_ms": process_warm_wall_ms,
            "process_warm_build_device_interval_ms": process_warm_device_ms,
            "hot_ms": hot_ms,
            "prior_ms": prior_ms,
            "break_even_calls": max(0.0, wall_ms - hot_ms) / saved_ms,
            "process_warm_break_even_calls": max(0.0, process_warm_wall_ms - hot_ms) / saved_ms,
            "folded_cache_bytes": cache_bytes,
            "dense_intermediate_retained": False,
            "timing_snapshot": snapshot,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del repeat_layer, repeat_output, repeat_window, x, planar, bank_ids
        torch.cuda.empty_cache()

    result = {
        "schema": "qvq_p32_amd_folded_cold_v1",
        "command": " ".join(sys.argv),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "benchmark_valid": valid,
        "model": "Qwen/Qwen3.8-27B",
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip},
        "baseline": str(args.baseline),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
