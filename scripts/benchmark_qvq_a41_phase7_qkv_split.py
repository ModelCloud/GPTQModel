#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep deterministic grouped-P32 query/key/value split-K on the H100."""

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
SPLITS = (1, 2, 4, 8)
K = 2048
WIDTHS = (2048, 512, 512)
ALT_IDS = (3, 1, 2)
SOURCE_PATHS = (
    Path("gptqmodel/quantization/qvq.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_a41_phase4_production.py"),
    Path("scripts/benchmark_qvq_a41_phase7_qkv_split.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--q-splits", nargs="+", type=int, default=SPLITS)
    parser.add_argument("--kv-splits", nargs="+", type=int, default=SPLITS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--repeatability-launches", type=int, default=5)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a41_phase7_h100/qkv_ordered_split_sweep.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if any(split not in SPLITS for split in (*args.q_splits, *args.kv_splits)):
        parser.error("query and key/value splits must be 1, 2, 4, or 8")
    if args.q_splits[0] != 1 or args.kv_splits[0] != 1:
        parser.error("both split lists must start with 1")
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


def _payloads(torch, bits: float, device):
    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile

    generator = torch.Generator().manual_seed(20261300 + int(bits * 10))
    words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    windows = []
    selectors = []
    for width in WIDTHS:
        tile_count = (K // 16) * (width // 16)
        planar = torch.randint(
            -(2**31),
            2**31 - 1,
            (tile_count, words_per_tile),
            generator=generator,
            dtype=torch.int32,
        )
        dense_selectors = torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            dtype=torch.uint8,
        )
        windows.append(repack_p32_planar_to_window(planar, bits=bits).to(device))
        selectors.append(pack_qvq_binary_bank_ids(dense_selectors).to(device))
    return tuple(windows), tuple(selectors)


def _run(args: argparse.Namespace) -> dict:
    import torch

    from gptqmodel.quantization.qvq import reconstruct_p32_window_inner_weight
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_group_plan,
        qvq_p32_window_wgmma_grouped_ordered_packed,
        qvq_p32_window_wgmma_grouped_packed,
        qvq_pack_p32_window_hopper_group,
    )

    source_fingerprint = _source_fingerprint()
    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
    rows = []
    for bits in args.rates:
        windows, selectors = _payloads(torch, bits, device)
        dense = tuple(
            reconstruct_p32_window_inner_weight(
                window,
                bits=bits,
                in_features=K,
                out_features=width,
                bank_ids=child_selectors,
                bank_alt_id=torch.tensor(alt_id, dtype=torch.uint8, device=device),
            )
            for window, child_selectors, width, alt_id in zip(
                windows, selectors, WIDTHS, ALT_IDS, strict=True
            )
        )
        baseline_plan = qvq_p32_window_wgmma_group_plan(
            torch.empty((16, K), dtype=torch.float16, device=device),
            windows,
            levels,
            selectors,
            bits,
            out_features=WIDTHS,
            bank_alt_ids=ALT_IDS,
            split_counts=(1, 1, 1),
        )
        baseline_payload = qvq_pack_p32_window_hopper_group(
            windows, selectors, baseline_plan
        )
        for m in args.m_values:
            generator = torch.Generator(device=device).manual_seed(
                20261400 + int(bits * 10) * 100 + m
            )
            input = torch.zeros((16, K), dtype=torch.float16, device=device)
            input[:m] = (
                torch.randn((m, K), generator=generator, device=device) * 0.02
            ).half()
            references = tuple(input[:m].float() @ weight for weight in dense)

            def baseline_call(
                input=input,
                payload=baseline_payload,
                levels=levels,
            ):
                return qvq_p32_window_wgmma_grouped_packed(input, payload, levels)

            baseline_timing, baseline_outputs = common._graph_timing(
                torch,
                baseline_call,
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            for child, reference in zip(baseline_outputs, references, strict=True):
                if (child[:m] - reference).abs().max().item() > 2e-3:
                    raise RuntimeError(
                        f"split-1 dense accuracy failed at W{bits:g} M{m}"
                    )

            for q_split in args.q_splits:
                for kv_split in args.kv_splits:
                    split_counts = (q_split, kv_split, kv_split)
                    plan = qvq_p32_window_wgmma_group_plan(
                        input,
                        windows,
                        levels,
                        selectors,
                        bits,
                        out_features=WIDTHS,
                        bank_alt_ids=ALT_IDS,
                        split_counts=split_counts,
                    )
                    payload = qvq_pack_p32_window_hopper_group(windows, selectors, plan)

                    def call(
                        input=input,
                        payload=payload,
                        levels=levels,
                    ):
                        return qvq_p32_window_wgmma_grouped_ordered_packed(
                            input, payload, levels
                        )

                    first = call()
                    torch.cuda.synchronize(device)
                    max_abs = max(
                        (child[:m] - reference).abs().max().item()
                        for child, reference in zip(first, references, strict=True)
                    )
                    if max_abs > 2e-3:
                        raise RuntimeError(
                            f"dense accuracy failed at W{bits:g} M{m} "
                            f"splits={split_counts}: max_abs={max_abs}"
                        )
                    repeatable = True
                    for _ in range(args.repeatability_launches - 1):
                        repeatable &= all(
                            torch.equal(child, expected)
                            for child, expected in zip(call(), first, strict=True)
                        )
                    torch.cuda.synchronize(device)
                    timing, graph_outputs = common._graph_timing(
                        torch,
                        call,
                        warmup=args.warmup,
                        samples=args.samples,
                        replays_per_sample=args.replays_per_sample,
                    )
                    graph_stable = all(
                        torch.equal(child, expected)
                        for child, expected in zip(graph_outputs, first, strict=True)
                    )
                    if not repeatable or not graph_stable:
                        raise RuntimeError(
                            f"repeatability failed at W{bits:g} M{m} "
                            f"splits={split_counts}"
                        )
                    logical_flops = 2 * m * K * sum(WIDTHS)
                    useful_blocks = sum(
                        (width // 64) * split
                        for width, split in zip(WIDTHS, split_counts, strict=True)
                    )
                    workspace_bytes = sum(
                        split * 16 * width * 4
                        for width, split in zip(WIDTHS, split_counts, strict=True)
                    )
                    row = {
                        "bits": bits,
                        "m": m,
                        "k": K,
                        "child_n": WIDTHS,
                        "split_counts": split_counts,
                        "useful_thread_blocks": useful_blocks,
                        "transient_partial_bytes": workspace_bytes,
                        "repeatable": repeatable,
                        "cuda_graph_stable": graph_stable,
                        "max_abs": max_abs,
                        "timing": timing,
                        "baseline_grouped_split1": baseline_timing,
                        "speedup_vs_grouped_split1": baseline_timing["median_ms"]
                        / timing["median_ms"],
                        "logical_effective_tflops": logical_flops
                        / (timing["median_ms"] * 1e9),
                    }
                    rows.append(row)
                    print(
                        f"W{bits:g} M{m} Q{q_split}/KV{kv_split}: "
                        f"{timing['median_ms'] * 1000:.3f}us "
                        f"speedup={row['speedup_vs_grouped_split1']:.3f}x "
                        f"blocks={useful_blocks}",
                        flush=True,
                    )
            del input, references, baseline_outputs
        del windows, selectors, dense, baseline_payload
        gc.collect()
        torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark sources changed while the H100 sweep was running")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
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
        "shape": "Llama 3.2 1B grouped query/key/value projections",
        "rates": args.rates,
        "m_values": args.m_values,
        "q_splits": args.q_splits,
        "kv_splits": args.kv_splits,
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
