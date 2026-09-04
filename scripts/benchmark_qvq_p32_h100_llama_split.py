#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep Llama 3.2 1B P32 down-projection split-K on the physical H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
LEGAL_SPLITS = (1, 2, 4, 8, 16, 32)
K = 8192
N = 2048
SOURCE_PATHS = (
    Path("gptqmodel/quantization/qvq.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_a41_phase4_production.py"),
    Path("scripts/benchmark_qvq_p32_h100_llama_split.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--splits", nargs="+", type=int, default=(1,))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--repeatability-launches", type=int, default=10)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a41_phase6_h100/llama_down_split_sweep.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if any(split not in LEGAL_SPLITS for split in args.splits):
        parser.error("splits must be selected from 1, 2, 4, 8, 16, or 32")
    if args.splits[0] != 1:
        parser.error("split 1 must be first so every row has a stable baseline")
    if (
        min(
            args.warmup,
            args.samples,
            args.replays_per_sample,
            args.repeatability_launches,
        )
        <= 0
    ):
        parser.error("timing and repeatability counts must be positive")
    if args.idle_samples < 3 or args.idle_interval < 0 or args.idle_memory_mib < 0:
        parser.error(
            "idle gate requires at least three samples and nonnegative thresholds"
        )
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma,
        qvq_p32_window_wgmma_m16_tma_ordered_split,
    )

    source_fingerprint = _source_fingerprint()
    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    rows = []
    for bits in args.rates:
        generator = torch.Generator().manual_seed(20260920 + int(bits * 10))
        tile_count = (K // 16) * (N // 16)
        words_per_tile = qvq_words_per_tile(
            bits,
            weight_count=256,
            vector_size=2,
        )
        planar = torch.randint(
            -(2**31),
            2**31 - 1,
            (tile_count, words_per_tile),
            generator=generator,
            dtype=torch.int32,
        )
        selectors = torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            dtype=torch.uint8,
        )
        window = repack_p32_planar_to_window(planar, bits=bits).to(device)
        bank_ids = pack_qvq_binary_bank_ids(selectors).to(device)
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=K,
            out_features=N,
            bank_ids=bank_ids,
            bank_alt_id=torch.tensor(3, dtype=torch.uint8, device=device),
        )

        for m in args.m_values:
            x = (
                torch.randn(
                    (m, K),
                    generator=torch.Generator(device=device).manual_seed(
                        20261000 + int(bits * 10) * 100 + m
                    ),
                    device=device,
                )
                * 0.02
            ).half()
            padded = torch.zeros((16, K), dtype=torch.float16, device=device)
            padded[:m].copy_(x)
            reference = x.float() @ dense

            split1_ms = None
            for split in args.splits:

                def call(
                    split=split,
                    padded=padded,
                    window=window,
                    levels=levels,
                    bank_ids=bank_ids,
                    bits=bits,
                    m=m,
                ):
                    kernel = (
                        qvq_p32_window_wgmma_m16_tma
                        if split == 1
                        else qvq_p32_window_wgmma_m16_tma_ordered_split
                    )
                    return kernel(
                        padded,
                        window,
                        levels,
                        bank_ids,
                        bits,
                        out_features=N,
                        bank_alt_id=3,
                        split_count=split,
                    )[:m]

                first = call()
                torch.cuda.synchronize(device)
                max_abs = (first.float() - reference).abs().max().item()
                if max_abs > 2e-3:
                    raise RuntimeError(
                        f"dense accuracy failed at W{bits:g} M{m} split{split}: "
                        f"max_abs={max_abs}"
                    )
                repeatable = True
                for _ in range(args.repeatability_launches - 1):
                    repeatable &= torch.equal(call(), first)
                torch.cuda.synchronize(device)
                timing, outputs = common._graph_timing(
                    torch,
                    lambda: (call(),),
                    warmup=args.warmup,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                )
                if not torch.equal(outputs[0], first):
                    raise RuntimeError(
                        f"CUDA Graph output changed at W{bits:g} M{m} split{split}"
                    )
                if split == 1:
                    split1_ms = timing["median_ms"]
                logical_flops = 2 * m * K * N
                executed_flops = 2 * 16 * K * N
                row = {
                    "bits": bits,
                    "m": m,
                    "k": K,
                    "n": N,
                    "split_count": split,
                    "reduction": "direct" if split == 1 else "ordered_fp32",
                    "thread_blocks": (N // 64) * split,
                    "blocks_per_sm": ((N // 64) * split) / device_info["sm_count"],
                    "repeatable": repeatable,
                    "max_abs": max_abs,
                    "timing": timing,
                    "speedup_vs_split1": (
                        None if split1_ms is None else split1_ms / timing["median_ms"]
                    ),
                    "logical_effective_tflops": logical_flops
                    / (timing["median_ms"] * 1e9),
                    "padded_m16_effective_tflops": executed_flops
                    / (timing["median_ms"] * 1e9),
                }
                rows.append(row)
                print(
                    f"W{bits:g} M{m} split{split}: "
                    f"{timing['median_ms'] * 1000:.3f}us "
                    f"blocks={row['thread_blocks']} repeatable={repeatable} "
                    f"max_abs={max_abs:.7g}",
                    flush=True,
                )
            del reference, x, padded
        del dense, window, bank_ids, planar, selectors
        gc.collect()
        torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark sources changed while the H100 sweep was running")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    payload = {
        "git_base_commit": commit,
        "source_fingerprint": source_fingerprint,
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "host_launch_gaps_included": False,
        },
        "shape": "Llama 3.2 1B down projection",
        "m_values": args.m_values,
        "rates": args.rates,
        "splits": args.splits,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)
    return payload


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
