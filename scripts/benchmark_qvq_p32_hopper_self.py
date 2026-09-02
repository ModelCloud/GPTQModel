#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the exact Hopper P32 TMA/WGMMA kernel against its own baseline.

This harness intentionally has no planar/scalar or Machete timing path.  It is
used to compare a candidate checkout with the checked-out origin/main kernel
using identical random P32 payloads, shapes, and CUDA-event timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("NINJAFLAGS", "-j2")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
os.environ.setdefault("NVCC_THREADS", "2")


@dataclass(frozen=True)
class ShapeCase:
    name: str
    in_features: int
    out_features: int


RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
SHAPES = {
    "mlp_gate_up": ShapeCase("qwen38_mlp_gate_up", 5120, 17408),
    "mlp_down": ShapeCase("qwen38_mlp_down", 17408, 5120),
}
H200_UUID = "GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea"
SOURCE_PATHS = (
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/quantization/qvq.py"),
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_git_commit().encode())
    for relative in SOURCE_PATHS:
        digest.update(str(relative).encode())
        digest.update((REPO_ROOT / relative).read_bytes())
    return digest.hexdigest()


def _idle_h200() -> dict[str, str]:
    query = "index,pci.bus_id,uuid,name,memory.used,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    hardware = dict(zip(query.split(","), values, strict=True))
    if hardware["uuid"] != H200_UUID or "H200" not in hardware["name"]:
        raise RuntimeError(f"physical GPU 0 is not the required H200: {hardware}")
    if int(hardware["memory.used"]) > 8 or int(hardware["utilization.gpu"]) != 0:
        raise RuntimeError(f"H200 is not idle: {hardware}")
    return hardware


def _timing(fn, *, warmup: int, iterations: int) -> dict[str, float]:
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
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="mlp_gate_up")
    parser.add_argument("--bits", nargs="+", type=float, default=[3.0])
    parser.add_argument("--m-values", nargs="+", type=int, default=[16])
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument(
        "--split", type=int, default=0,
        help="Explicit K split count; 0 uses the production H200 shape policy.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(bits not in RATES for bits in args.bits):
        parser.error("--bits supports W2, W2.5, W3, and W3.5")
    if any(m not in M_VALUES for m in args.m_values):
        parser.error("--m-values supports 1, 2, 4, 8, and 16")
    if args.warmup <= 0 or args.iterations <= 0:
        parser.error("warmup and iterations must be positive")
    if args.split < 0 or args.split > 64:
        parser.error("split must be in [0, 64]")
    return args


def main() -> None:
    args = _parse_args()
    hardware = _idle_h200()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"H200 is required, got {properties.name}")

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_wgmma_cuda import qvq_p32_window_wgmma_m16_tma

    shape = SHAPES[args.shape]
    rows: list[dict] = []
    for bits_index, bits in enumerate(args.bits):
        generator = torch.Generator(device="cpu").manual_seed(20260912 + int(bits * 10))
        tile_count = (shape.in_features // 16) * (shape.out_features // 16)
        words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
        planar = torch.randint(
            -(2**31), 2**31 - 1, (tile_count, words_per_tile), generator=generator, dtype=torch.int32
        )
        selectors = torch.randint(0, 2, (tile_count * 8,), generator=generator, dtype=torch.uint8)
        window = repack_p32_planar_to_window(planar, bits=bits).cuda()
        bank_ids = pack_qvq_binary_bank_ids(selectors).cuda()
        levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=shape.in_features,
            out_features=shape.out_features,
            bank_ids=bank_ids,
            bank_alt_id=torch.tensor(3, dtype=torch.uint8, device="cuda"),
        )
        for m_index, m in enumerate(args.m_values):
            x = (torch.randn((m, shape.in_features), generator=generator, dtype=torch.float32) * 0.1).half().cuda()
            padded = torch.zeros((16, shape.in_features), dtype=torch.float16, device="cuda")

            def call():
                if m < 16:
                    padded[:m].copy_(x)
                    return qvq_p32_window_wgmma_m16_tma(
                        padded, window, levels, bank_ids, bits, out_features=shape.out_features,
                        split_count=args.split,
                    )[:m]
                return qvq_p32_window_wgmma_m16_tma(
                    x, window, levels, bank_ids, bits, out_features=shape.out_features,
                    split_count=args.split,
                )

            expected = x.float() @ dense
            actual = call()
            torch.cuda.synchronize()
            max_abs = (actual.float() - expected).abs().max().item()
            if max_abs > 2e-3:
                raise RuntimeError(f"accuracy failure W{bits:g} M{m}: max_abs={max_abs}")
            timing = _timing(call, warmup=args.warmup, iterations=args.iterations)
            row = {
                "bits": bits,
                "m": m,
                "shape": shape.name,
                "k": shape.in_features,
                "n": shape.out_features,
                "max_abs": max_abs,
                **timing,
            }
            rows.append(row)
            print(
                f"complete {shape.name} W{bits:g} M{m}: {timing['median_ms']:.6f} ms "
                f"max_abs={max_abs:.7g}",
                flush=True,
            )
        del dense, window, bank_ids, levels
        torch.cuda.empty_cache()

    payload = {
        "commit": _git_commit(),
        "source_fingerprint": _source_fingerprint(),
        "physical_gpu": 0,
        "hardware": hardware,
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
        },
        "shape": shape.name,
        "bits": args.bits,
        "m_values": args.m_values,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "split": args.split,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
