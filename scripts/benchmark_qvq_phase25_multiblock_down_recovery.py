#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact multiblock split-16 down recovery on a physical H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

M_VALUES = (1, 2, 4, 8, 16)
BITS = (2, 2.5, 3, 3.5)
K = 8192
N = 2048
SPLIT = 16
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_cuda.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_hadamard_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_phase25_multiblock_down_recovery.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", nargs="+", type=float, default=BITS)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/a41_phase25_h100/multiblock_down_recovery_experiment.json"
        ),
    )
    args = parser.parse_args()
    if any(value not in BITS for value in args.bits):
        parser.error("bits must be selected from W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _payload(torch, *, bits: float, generator, device):
    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile

    words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    tile_count = (K // 16) * (N // 16)
    planar = torch.randint(
        0,
        1 << 32,
        (tile_count, words),
        generator=generator,
        device=device,
        dtype=torch.int64,
    ).to(torch.int32)
    window = repack_p32_planar_to_window(planar, bits=bits)
    selectors = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
    )
    return window, selectors


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.utils.qvq_cuda import (
        qvq_cuda_hadamard_ordered_split16_fp32_to_fp16,
    )
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma_ordered_partials,
    )

    device_info = common._assert_h100(torch)
    fingerprint = _source_fingerprint()
    device = torch.device("cuda", 0)
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to(device).contiguous()
    rows = []
    for bits in args.bits:
        generator = torch.Generator(device=device).manual_seed(25000 + int(bits * 10))
        window, selectors = _payload(
            torch, bits=bits, generator=generator, device=device
        )
        for m in args.m_values:
            x = torch.zeros((16, K), dtype=torch.float16, device=device)
            x[:m] = (
                torch.randn(
                    (m, K), generator=generator, device=device, dtype=torch.float16
                )
                * 0.02
            )
            post_scale = (
                torch.randn(
                    (N,), generator=generator, device=device, dtype=torch.float32
                )
                * 0.002
            )
            bias = (
                torch.randn(
                    (N,), generator=generator, device=device, dtype=torch.float32
                )
                * 0.001
            )
            fixed_partials = qvq_p32_window_wgmma_m16_tma_ordered_partials(
                x,
                window,
                levels,
                selectors,
                bits,
                out_features=N,
                bank_alt_id=3,
                split_count=SPLIT,
            )

            def recover(
                multiblock: bool,
                fixed_partials=fixed_partials,
                post_scale=post_scale,
                bias=bias,
                m=m,
            ):
                return qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
                    fixed_partials,
                    post_scale=post_scale,
                    bias=bias,
                    scale_mode=3,
                    logical_rows=m,
                    multiblock=multiblock,
                )

            def site(
                multiblock: bool,
                x=x,
                window=window,
                selectors=selectors,
                bits=bits,
                post_scale=post_scale,
                bias=bias,
                m=m,
            ):
                partials = qvq_p32_window_wgmma_m16_tma_ordered_partials(
                    x,
                    window,
                    levels,
                    selectors,
                    bits,
                    out_features=N,
                    bank_alt_id=3,
                    split_count=SPLIT,
                )
                return qvq_cuda_hadamard_ordered_split16_fp32_to_fp16(
                    partials,
                    post_scale=post_scale,
                    bias=bias,
                    scale_mode=3,
                    logical_rows=m,
                    multiblock=multiblock,
                )

            scalar_recovery, scalar_recovery_outputs = common._graph_timing(
                torch,
                lambda: recover(False),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            multiblock_recovery, multiblock_recovery_outputs = common._graph_timing(
                torch,
                lambda: recover(True),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            scalar_site, scalar_site_outputs = common._graph_timing(
                torch,
                lambda: site(False),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            multiblock_site, multiblock_site_outputs = common._graph_timing(
                torch,
                lambda: site(True),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            bit_exact = all(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                for actual, expected in zip(
                    multiblock_recovery_outputs + multiblock_site_outputs,
                    scalar_recovery_outputs + scalar_site_outputs,
                    strict=True,
                )
            )
            if not bit_exact:
                raise RuntimeError(
                    f"multiblock down recovery changed W{bits:g}/M{m} bits"
                )
            recovery_speedup = (
                scalar_recovery["median_ms"] / multiblock_recovery["median_ms"]
            )
            site_speedup = scalar_site["median_ms"] / multiblock_site["median_ms"]
            row = {
                "bits": bits,
                "m": m,
                "k": K,
                "n": N,
                "split_count": SPLIT,
                "scalar_recovery": scalar_recovery,
                "multiblock_recovery": multiblock_recovery,
                "recovery_speedup": recovery_speedup,
                "scalar_site": scalar_site,
                "multiblock_site": multiblock_site,
                "site_speedup": site_speedup,
                "better_than_phase23_site": site_speedup > 1.0,
                "bit_exact": bit_exact,
                "transient_workspace_bytes": m * N * 4,
                "additional_kernel_launches": 1,
            }
            rows.append(row)
            print(
                f"W{bits:g}/M{m}: recovery "
                f"{scalar_recovery['median_ms'] * 1000:.3f}->"
                f"{multiblock_recovery['median_ms'] * 1000:.3f}us "
                f"({recovery_speedup:.4f}x), site "
                f"{scalar_site['median_ms'] * 1000:.3f}->"
                f"{multiblock_site['median_ms'] * 1000:.3f}us "
                f"({site_speedup:.4f}x), exact={bit_exact}",
                flush=True,
            )

    if fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark sources changed during the H100 matrix")
    recovery_geomean = math.exp(
        sum(math.log(row["recovery_speedup"]) for row in rows) / len(rows)
    )
    site_geomean = math.exp(
        sum(math.log(row["site_speedup"]) for row in rows) / len(rows)
    )
    payload = {
        "git_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "host_launch_gaps_included": False,
        },
        "workload": "H100 Llama 3.2 1B P32 down decode and exact split-16 FP16 recovery",
        "recovery_geomean_speedup": recovery_geomean,
        "site_geomean_speedup": site_geomean,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"geomean recovery={recovery_geomean:.4f}x site={site_geomean:.4f}x",
        flush=True,
    )
    print(f"result: {args.output}")
    return payload


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
