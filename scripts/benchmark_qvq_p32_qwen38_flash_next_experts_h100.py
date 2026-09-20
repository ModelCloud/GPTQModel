#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact Qwen3.8-Flash-Next P32 expert projections on H100."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
EXPERT_WIDTH = 640
HIDDEN_SIZE = 2560


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--projections", nargs="+", choices=("gate_up", "down"), default=("gate_up", "down")
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--fp64-columns", type=int, default=256)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen38_flash_next_h100/expert_phase2.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample, args.fp64_columns) <= 0:
        parser.error("timing and oracle counts must be positive")
    if args.fp64_columns > HIDDEN_SIZE:
        parser.error("fp64 column count exceeds the largest expert output")
    return args


def _metrics(actual, reference) -> dict[str, float]:
    import torch

    error = actual.double() - reference.double()
    return {
        "mean_abs": float(error.abs().mean().item()),
        "max_abs": float(error.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(error)
            .div(torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))
            .item()
        ),
    }


def _payload(torch, *, bits: float, k: int, n: int, seed: int):
    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.planar_packing import planar_pack_rows

    generator = torch.Generator().manual_seed(seed)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (128, tile_count),
        generator=generator,
        dtype=torch.int32,
    )
    planar = planar_pack_rows(edges, transition_bits).T.contiguous().cuda()
    window = repack_p32_planar_to_window(planar, bits=bits)
    selectors = pack_qvq_binary_bank_ids(
        torch.randint(
            0,
            2,
            (tile_count * 8,),
            generator=generator,
            dtype=torch.uint8,
        )
    ).cuda()
    dense = reconstruct_p32_window_inner_weight(
        window,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=selectors,
        bank_alt_id=torch.tensor(3, dtype=torch.uint8, device="cuda"),
    )
    return planar, window, selectors, dense


def _timed_sandwich(torch, baseline, candidate, args):
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
        raise RuntimeError("candidate graph replay is not bitwise repeatable")
    candidate_median = math.sqrt(
        candidate_a["median_ms"] * candidate_b["median_ms"]
    )
    return {
        "baseline": baseline_timing,
        "candidate_first": candidate_a,
        "candidate_second": candidate_b,
        "candidate_sandwich_median_ms": candidate_median,
        "speedup": baseline_timing["median_ms"] / candidate_median,
    }, baseline_output, candidate_output_b


def _run(args: argparse.Namespace) -> None:
    os.environ["QVQ_AMPERE_AUTOTUNE"] = "0"

    import torch

    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.utils.qvq_ampere_cuda import (
        qvq_p32_window_ampere,
        qvq_p32_window_ampere_group_plan,
        qvq_p32_window_ampere_grouped_packed,
        qvq_pack_p32_window_ampere_group,
    )
    from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv

    device = common._assert_h100(torch)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    rows = []
    for bits in args.rates:
        gate_payloads = tuple(
            _payload(
                torch,
                bits=bits,
                k=HIDDEN_SIZE,
                n=EXPERT_WIDTH,
                seed=20260921 + int(bits * 10) * 10 + child,
            )
            for child in range(2)
        )
        down_payload = _payload(
            torch,
            bits=bits,
            k=EXPERT_WIDTH,
            n=HIDDEN_SIZE,
            seed=20261921 + int(bits * 10),
        )
        input_generator = torch.Generator(device="cuda").manual_seed(
            20262921 + int(bits * 10)
        )
        for logical_rows in args.m_values:
            if "gate_up" in args.projections:
                x = (
                    torch.randn(
                        (logical_rows, HIDDEN_SIZE),
                        generator=input_generator,
                        device="cuda",
                    )
                    * 0.02
                ).half()
                plan = qvq_p32_window_ampere_group_plan(
                    x,
                    tuple(payload[1] for payload in gate_payloads),
                    levels,
                    tuple(payload[2] for payload in gate_payloads),
                    bits,
                    out_features=(EXPERT_WIDTH, EXPERT_WIDTH),
                    bank_alt_ids=(3, 3),
                    # Split 32 deliberately selects the validated generic
                    # segmented SM90 kernel at every rate.
                    split_counts=(32, 32),
                )
                packed = qvq_pack_p32_window_ampere_group(
                    tuple(payload[1] for payload in gate_payloads),
                    tuple(payload[2] for payload in gate_payloads),
                    plan,
                )

                def gate_baseline(
                    x=x, bits=bits, gate_payloads=gate_payloads
                ):
                    return torch.cat(
                        tuple(
                            qvq_cuda_gemv(
                                x,
                                payload[0],
                                bits,
                                out_features=EXPERT_WIDTH,
                                output_fp32=True,
                                vector_size=2,
                                bank_ids=payload[2],
                                v2b2_p32=True,
                                bank_alt_id=3,
                                _bank_ids_validated=True,
                            )
                            for payload in gate_payloads
                        ),
                        dim=1,
                    )

                def gate_candidate(x=x, packed=packed):
                    return torch.cat(
                        qvq_p32_window_ampere_grouped_packed(x, packed, levels), dim=1
                    )

                timing, baseline, candidate = _timed_sandwich(
                    torch, gate_baseline, gate_candidate, args
                )
                fp64_columns = min(args.fp64_columns, EXPERT_WIDTH)
                columns = torch.linspace(
                    0,
                    EXPERT_WIDTH - 1,
                    fp64_columns,
                    dtype=torch.int64,
                    device="cuda",
                )
                dense = torch.cat(tuple(payload[3] for payload in gate_payloads), dim=1)
                oracle_columns = torch.cat((columns, columns + EXPERT_WIDTH))
                fp64 = x.double() @ dense[:, oracle_columns].double()
                fp32 = x.float() @ dense[:, oracle_columns].float()
                baseline_sample = baseline[:, oracle_columns]
                candidate_sample = candidate[:, oracle_columns]
                local = _metrics(candidate, baseline)
                if local["mean_abs"] > 4e-3 or local["max_abs"] > 0.046875:
                    raise RuntimeError(f"gate/up local accuracy gate failed: {local}")
                rows.append(
                    {
                        "projection": "gate_up",
                        "bits": bits,
                        "mkn": [logical_rows, HIDDEN_SIZE, 2 * EXPERT_WIDTH],
                        "split_counts": [segment.split_count for segment in plan.segments],
                        **timing,
                        "candidate_vs_baseline": local,
                        "baseline_vs_fp64": _metrics(baseline_sample, fp64),
                        "candidate_vs_fp64": _metrics(candidate_sample, fp64),
                        "fp32_vs_fp64": _metrics(fp32, fp64),
                        "repeatable": True,
                    }
                )
                print(
                    f"W{bits:g} gate/up M{logical_rows}: "
                    f"{timing['baseline']['median_ms'] * 1000:.3f}us -> "
                    f"{timing['candidate_sandwich_median_ms'] * 1000:.3f}us "
                    f"({timing['speedup']:.3f}x) splits="
                    f"{tuple(segment.split_count for segment in plan.segments)}",
                    flush=True,
                )

            if "down" in args.projections:
                x = (
                    torch.randn(
                        (logical_rows, EXPERT_WIDTH),
                        generator=input_generator,
                        device="cuda",
                    )
                    * 0.02
                ).half()
                planar, window, selectors, dense = down_payload

                def down_baseline(
                    x=x, planar=planar, bits=bits, selectors=selectors
                ):
                    return qvq_cuda_gemv(
                        x,
                        planar,
                        bits,
                        out_features=HIDDEN_SIZE,
                        output_fp32=True,
                        vector_size=2,
                        bank_ids=selectors,
                        v2b2_p32=True,
                        bank_alt_id=3,
                        _bank_ids_validated=True,
                    )

                def down_candidate(
                    x=x,
                    window=window,
                    selectors=selectors,
                    bits=bits,
                ):
                    return qvq_p32_window_ampere(
                        x,
                        window,
                        levels,
                        selectors,
                        bits,
                        out_features=HIDDEN_SIZE,
                        bank_alt_id=3,
                        split_count=0,
                    )

                timing, baseline, candidate = _timed_sandwich(
                    torch, down_baseline, down_candidate, args
                )
                columns = torch.linspace(
                    0,
                    HIDDEN_SIZE - 1,
                    args.fp64_columns,
                    dtype=torch.int64,
                    device="cuda",
                )
                fp64 = x.double() @ dense[:, columns].double()
                fp32 = x.float() @ dense[:, columns].float()
                baseline_sample = baseline[:, columns]
                candidate_sample = candidate[:, columns]
                local = _metrics(candidate, baseline)
                if local["mean_abs"] > 4e-3 or local["max_abs"] > 0.046875:
                    raise RuntimeError(f"down local accuracy gate failed: {local}")
                rows.append(
                    {
                        "projection": "down",
                        "bits": bits,
                        "mkn": [logical_rows, EXPERT_WIDTH, HIDDEN_SIZE],
                        **timing,
                        "candidate_vs_baseline": local,
                        "baseline_vs_fp64": _metrics(baseline_sample, fp64),
                        "candidate_vs_fp64": _metrics(candidate_sample, fp64),
                        "fp32_vs_fp64": _metrics(fp32, fp64),
                        "repeatable": True,
                    }
                )
                print(
                    f"W{bits:g} down M{logical_rows}: "
                    f"{timing['baseline']['median_ms'] * 1000:.3f}us -> "
                    f"{timing['candidate_sandwich_median_ms'] * 1000:.3f}us "
                    f"({timing['speedup']:.3f}x)",
                    flush=True,
                )

    payload = {
        "checkpoint": "/monster/data/model/Qwen3.8-Flash-Next",
        "device": device,
        "baseline": "current planar qvq_cuda V2B2-P32 path",
        "candidate": "native exact contiguous-window narrow WMMA kernel on SM90",
        "math_policy": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "kernel_input": "FP16",
            "kernel_accumulator": "FP32",
            "output": "FP32",
        },
        "timing": {
            "method": "candidate/control/candidate CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
        },
        "oracle": {
            "kind": "sampled FP32 and FP64 accumulation over full K",
            "columns_per_projection": args.fp64_columns,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
