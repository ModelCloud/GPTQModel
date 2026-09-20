#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ordered P32 attention output at Qwen3.8-Flash-Next geometry."""

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

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
SPLITS = {2.0: 12, 2.5: 6, 3.0: 12, 3.5: 24}
IN_FEATURES = 6144
OUT_FEATURES = 2560
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_p32_qwen38_flash_next_h100.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--fp64-columns", type=int, default=256)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen38_flash_next_h100/attention_output_phase1.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample, args.fp64_columns) <= 0:
        parser.error("timing and oracle counts must be positive")
    if args.fp64_columns > OUT_FEATURES:
        parser.error("fp64 column count exceeds the output width")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def _metrics(torch, actual, reference) -> dict[str, float]:
    error = actual.double() - reference.double()
    reference_norm = torch.linalg.vector_norm(reference.double())
    return {
        "mean_abs": float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "max_abs": float(error.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(error).div(reference_norm).item()
        ),
    }


def _main(args: argparse.Namespace) -> None:
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
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.planar_packing import planar_pack_rows
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_p32_window_wgmma_m16_tma,
        qvq_p32_window_wgmma_m16_tma_ordered_split,
    )

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fingerprint = _fingerprint()
    rows = []
    for bits in args.rates:
        payload_generator = torch.Generator().manual_seed(20260920 + int(bits * 10))
        input_generator = torch.Generator(device=device).manual_seed(
            20261920 + int(bits * 10)
        )
        transition_bits = qvq_transition_bits(bits, vector_size=2)
        tile_count = (IN_FEATURES // 16) * (OUT_FEATURES // 16)
        edges = torch.randint(
            0,
            1 << transition_bits,
            (128, tile_count),
            generator=payload_generator,
            dtype=torch.int32,
        )
        planar = planar_pack_rows(edges, transition_bits).T.contiguous()
        window = repack_p32_planar_to_window(planar, bits=bits).to(device)
        selectors = pack_qvq_binary_bank_ids(
            torch.randint(
                0,
                2,
                (tile_count * 8,),
                generator=payload_generator,
                dtype=torch.uint8,
            )
        ).to(device)
        levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().to(device)
        dense = reconstruct_p32_window_inner_weight(
            window,
            bits=bits,
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            bank_ids=selectors,
            bank_alt_id=torch.tensor(3, dtype=torch.uint8, device=device),
        )
        columns = torch.linspace(
            0,
            OUT_FEATURES - 1,
            args.fp64_columns,
            dtype=torch.int64,
            device=device,
        )
        for logical_rows in args.m_values:
            x = (
                torch.randn(
                    (logical_rows, IN_FEATURES),
                    generator=input_generator,
                    dtype=torch.float32,
                    device=device,
                )
                * 0.02
            ).half()
            split = SPLITS[bits]

            def baseline(
                x=x,
                window=window,
                levels=levels,
                selectors=selectors,
                bits=bits,
            ):
                return qvq_p32_window_wgmma_m16_tma(
                    x,
                    window,
                    levels,
                    selectors,
                    bits,
                    out_features=OUT_FEATURES,
                    bank_alt_id=3,
                    split_count=1,
                )

            def candidate(
                x=x,
                window=window,
                levels=levels,
                selectors=selectors,
                bits=bits,
                split=split,
            ):
                return qvq_p32_window_wgmma_m16_tma_ordered_split(
                    x,
                    window,
                    levels,
                    selectors,
                    bits,
                    out_features=OUT_FEATURES,
                    bank_alt_id=3,
                    split_count=split,
                )

            candidate_a, (candidate_output_a,) = common._graph_timing(
                torch,
                lambda: (candidate(),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            baseline_timing, (baseline_output,) = common._graph_timing(
                torch,
                lambda: (baseline(),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            candidate_b, (candidate_output_b,) = common._graph_timing(
                torch,
                lambda: (candidate(),),
                warmup=args.warmup,
                samples=args.samples,
                replays_per_sample=args.replays_per_sample,
            )
            if not torch.equal(candidate_output_a, candidate_output_b):
                raise RuntimeError(f"ordered output is not repeatable at W{bits:g} M{logical_rows}")

            fp64 = x.double() @ dense[:, columns].double()
            fp32 = x.float() @ dense[:, columns].float()
            baseline_sample = baseline_output[:, columns]
            candidate_sample = candidate_output_b[:, columns]
            baseline_error = (baseline_sample.double() - fp64).abs()
            candidate_error = (candidate_sample.double() - fp64).abs()
            local_error = _metrics(torch, candidate_output_b, baseline_output)
            candidate_median = math.sqrt(
                candidate_a["median_ms"] * candidate_b["median_ms"]
            )
            row = {
                "bits": bits,
                "m": logical_rows,
                "mkn": [logical_rows, IN_FEATURES, OUT_FEATURES],
                "ordered_split": split,
                "baseline": baseline_timing,
                "candidate_first": candidate_a,
                "candidate_second": candidate_b,
                "candidate_sandwich_median_ms": candidate_median,
                "speedup": baseline_timing["median_ms"] / candidate_median,
                "candidate_vs_baseline": local_error,
                "baseline_vs_fp64": _metrics(torch, baseline_sample, fp64),
                "candidate_vs_fp64": _metrics(torch, candidate_sample, fp64),
                "fp32_vs_fp64": _metrics(torch, fp32, fp64),
                "fp64_closer_counts": {
                    "candidate": int((candidate_error < baseline_error).sum().item()),
                    "baseline": int((baseline_error < candidate_error).sum().item()),
                    "ties": int((baseline_error == candidate_error).sum().item()),
                },
                "repeatable": True,
            }
            if local_error["mean_abs"] > 4e-3 or local_error["max_abs"] > 0.046875:
                raise RuntimeError(f"local accuracy gate failed: {row}")
            rows.append(row)
            print(
                f"W{bits:g} M{logical_rows}: baseline={baseline_timing['median_ms'] * 1000:.3f}us "
                f"ordered={candidate_median * 1000:.3f}us speedup={row['speedup']:.4f}x "
                f"mae={local_error['mean_abs']:.3g} max={local_error['max_abs']:.3g}",
                flush=True,
            )

    if fingerprint != _fingerprint():
        raise RuntimeError("benchmark sources changed during execution")
    payload = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "checkpoint": "/monster/data/model/Qwen3.8-Flash-Next",
        "tensor": "model.language_model.layers.3.self_attn.o_proj.weight",
        "weight_shape_nk": [OUT_FEATURES, IN_FEATURES],
        "device": device_info,
        "math_policy": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "kernel_input": "FP16",
            "kernel_accumulator": "FP32 WGMMA",
            "split_reduction": "deterministic left-to-right FP32",
            "output": "FP32",
        },
        "timing": {
            "method": "candidate/control/candidate CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
        },
        "oracle": {
            "kind": "sampled FP64 accumulation over full K",
            "columns": args.fp64_columns,
            "synthetic_fixture_scope": "kernel correctness and performance only",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
