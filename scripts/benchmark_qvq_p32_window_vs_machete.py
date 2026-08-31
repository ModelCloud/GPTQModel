#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact standard-P32 window WGMMA against planar P32 and Machete."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("NINJAFLAGS", "-j2")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
os.environ.setdefault("NVCC_THREADS", "2")

from scripts import benchmark_qvq_cuda_lr as benchmark_utils
from scripts import benchmark_qvq_lr_vs_gptq_llama32_1b as comparison


RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(case.name for case in comparison.QWEN38_27B_SHAPES),
        default=tuple(case.name for case in comparison.QWEN38_27B_SHAPES),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/h200_p32_window/qwen38_m16_p32_vs_machete.json"),
    )
    args = parser.parse_args()
    if args.physical_gpu != 0:
        parser.error("the standard-P32 benchmark is pinned to physical H200 GPU 0")
    if any(rate not in RATES for rate in args.rates):
        parser.error("--rates supports only 2, 2.5, 3, and 3.5")
    if any(m not in M_VALUES for m in args.m_values):
        parser.error("--m-values supports only 1, 2, 4, 8, and 16")
    if args.warmup <= 0 or args.iterations <= 0 or args.idle_samples <= 0:
        parser.error("warmup, iterations, and idle-samples must be positive")
    return args


def _timed_row(*, kernel: str, timing: dict[str, float], max_abs: float, **fields) -> dict:
    return {"kernel": kernel, **fields, **timing, "max_abs": max_abs}


def _print_table(rows: list[dict]) -> None:
    print("| Rate | Shape | M | K | N | Planar P32 ms | Window WGMMA ms | Speedup | Machete W4 ms | xMachete | Max abs |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ratios = []
    for candidate in (row for row in rows if row["kernel"] == "p32_window_tma_rs_wgmma"):
        matching = [
            row
            for row in rows
            if row["shape"] == candidate["shape"]
            and row["bits"] == candidate["bits"]
            and row["m"] == candidate["m"]
        ]
        planar = next(row for row in matching if row["kernel"] == "planar_p32")
        machete = next(row for row in matching if row["kernel"] == "machete_w4")
        speedup = planar["median_ms"] / candidate["median_ms"]
        ratio = machete["median_ms"] / candidate["median_ms"]
        ratios.append(ratio)
        print(
            f"| W{candidate['bits']:g} | {candidate['shape']} | {candidate['m']} | "
            f"{candidate['k']} | {candidate['n']} | {planar['median_ms']:.5f} | "
            f"{candidate['median_ms']:.5f} | {speedup:.3f}x | {machete['median_ms']:.5f} | "
            f"{ratio:.3f}x | {candidate['max_abs']:.7g} |"
        )
    geomean = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    print(f"combined window-P32 geomean: {geomean:.3f}x Machete", flush=True)


def _run(args: argparse.Namespace) -> dict:
    commit = benchmark_utils._git_commit()
    fingerprint = benchmark_utils._source_fingerprint()
    hardware = benchmark_utils._idle_preflight(
        args.physical_gpu,
        args.idle_samples,
        args.idle_interval,
        8,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = hardware["uuid"]

    import torch

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        reconstruct_p32_window_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.machete import (
        _validate_machete_device_support,
        machete_runtime_error,
        prewarm_machete_extension,
    )
    from gptqmodel.utils.marlin_scalar_type import scalar_types
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv
    from gptqmodel.utils.qvq_wgmma_cuda import qvq_p32_window_wgmma_m16_tma

    comparison._visible_gpu_matches(torch, hardware)
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"this benchmark requires H200 GPU 0, got {properties.name}")
    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")
    if not _validate_machete_device_support() or not prewarm_machete_extension():
        raise RuntimeError(machete_runtime_error())

    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    selected_shapes = [
        comparison._shape_by_name(name, "qwen38_27b") for name in args.shapes
    ]
    rows: list[dict] = []
    benchmark_cases = [
        (m, case)
        for m in args.m_values
        for case in selected_shapes
    ]
    for case_index, (m, case) in enumerate(benchmark_cases):
        generator = torch.Generator().manual_seed(20260904 + case_index)
        x = (torch.randn((m, case.in_features), generator=generator) * 0.1).half().cuda()
        window_input = x if m == 16 else torch.zeros(
            (16, case.in_features), dtype=torch.float16, device="cuda"
        )

        gptq_source = comparison._gptq_source(
            torch,
            case,
            group_size=comparison.GPTQ_GROUP_SIZE,
            seed=20260940 + case_index,
            dtype=torch.float16,
        )
        dense_gptq = comparison._dense_gptq_weight(
            torch,
            gptq_source,
            device=torch.device("cuda:0"),
            weight_type=scalar_types.uint4b8,
        )
        machete, _ = comparison._build_gptq_module(
            torch,
            "gptq_machete",
            case,
            group_size=comparison.GPTQ_GROUP_SIZE,
            source=gptq_source,
            device=torch.device("cuda:0"),
            dtype=torch.float16,
        )
        gptq_reference = x.float() @ dense_gptq.float()

        def machete_call():
            return machete(x)

        machete_output = machete_call()
        torch.cuda.synchronize()
        machete_metrics = comparison._assert_correct(
            kernel="Machete W4",
            actual=machete_output,
            reference=gptq_reference,
            expected_shape=(m, case.out_features),
            expected_dtype=torch.float16,
            atol=2e-2,
            rtol=2e-2,
        )
        machete_timing = comparison._cuda_graph_event_timing(
            torch, machete_call, warmup=args.warmup, iterations=args.iterations
        )
        del dense_gptq, gptq_reference, machete_output

        for rate_index, bits in enumerate(args.rates):
            rate_generator = torch.Generator().manual_seed(
                20261000 + case_index * len(RATES) + rate_index
            )
            tile_count = (case.in_features // 16) * (case.out_features // 16)
            words_per_tile = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
            planar = torch.randint(
                0,
                1 << 32,
                (tile_count, words_per_tile),
                generator=rate_generator,
                dtype=torch.int64,
            ).to(torch.int32)
            selectors = torch.randint(
                0,
                2,
                (tile_count * 8,),
                generator=rate_generator,
                dtype=torch.uint8,
            )
            bank_ids = pack_qvq_binary_bank_ids(selectors).cuda()
            window = repack_p32_planar_to_window(planar, bits=bits).cuda()
            planar = planar.cuda()
            dense_p32 = reconstruct_p32_window_inner_weight(
                window,
                bits=bits,
                in_features=case.in_features,
                out_features=case.out_features,
                bank_ids=bank_ids,
                bank_alt_id=torch.tensor(3, dtype=torch.uint8, device="cuda"),
            )
            p32_reference = x.float() @ dense_p32

            def planar_call():
                return qvq_cuda_gemv(
                    x,
                    planar,
                    bits,
                    out_features=case.out_features,
                    output_fp32=True,
                    bank_ids=bank_ids,
                    v2b2_p32=True,
                    bank_alt_id=3,
                )

            def window_call():
                if m < 16:
                    window_input[:m].copy_(x)
                output = qvq_p32_window_wgmma_m16_tma(
                    window_input,
                    window,
                    levels,
                    bank_ids,
                    bits,
                    out_features=case.out_features,
                    bank_alt_id=3,
                )
                return output if m == 16 else output[:m]

            planar_output = planar_call()
            window_output = window_call()
            torch.cuda.synchronize()
            planar_metrics = comparison._assert_correct(
                kernel=f"planar P32 W{bits:g}",
                actual=planar_output,
                reference=p32_reference,
                expected_shape=(m, case.out_features),
                expected_dtype=torch.float32,
                atol=2e-3,
                rtol=0.0,
            )
            window_metrics = comparison._assert_correct(
                kernel=f"window P32 W{bits:g}",
                actual=window_output,
                reference=p32_reference,
                expected_shape=(m, case.out_features),
                expected_dtype=torch.float32,
                atol=2e-3,
                rtol=0.0,
            )
            fields = {
                "bits": bits,
                "shape": case.name,
                "m": m,
                "k": case.in_features,
                "n": case.out_features,
            }
            rows.append(
                _timed_row(
                    kernel="planar_p32",
                    timing=comparison._cuda_graph_event_timing(
                        torch, planar_call, warmup=args.warmup, iterations=args.iterations
                    ),
                    max_abs=planar_metrics["max_abs"],
                    **fields,
                )
            )
            rows.append(
                _timed_row(
                    kernel="p32_window_tma_rs_wgmma",
                    timing=comparison._cuda_graph_event_timing(
                        torch, window_call, warmup=args.warmup, iterations=args.iterations
                    ),
                    max_abs=window_metrics["max_abs"],
                    **fields,
                )
            )
            rows.append(
                _timed_row(
                    kernel="machete_w4",
                    timing=machete_timing,
                    max_abs=machete_metrics["max_abs"],
                    **fields,
                )
            )
            del planar, selectors, bank_ids, window, dense_p32, p32_reference
            del planar_output, window_output
            gc.collect()
            torch.cuda.empty_cache()
            candidate = rows[-2]
            print(
                f"complete M{m} W{bits:g} {case.name}: window={candidate['median_ms']:.5f} ms "
                f"machete={machete_timing['median_ms']:.5f} ms max_abs={candidate['max_abs']:.7g}",
                flush=True,
            )
        del x, window_input, machete, gptq_source
        gc.collect()
        torch.cuda.empty_cache()

    benchmark_utils._verify_source(commit, fingerprint, phase="after P32 benchmark completed")
    payload = {
        "commit": commit,
        "source_fingerprint": fingerprint,
        "physical_gpu": args.physical_gpu,
        "hardware": hardware,
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
        },
        "model_shapes": "qwen38_27b",
        "m_values": args.m_values,
        "small_m_policy": "copy live rows into persistent zero-padded M16 input inside timed graph",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_table(rows)
    print(f"result: {args.output}", flush=True)
    return payload


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
