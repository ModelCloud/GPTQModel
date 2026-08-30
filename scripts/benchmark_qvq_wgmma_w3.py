#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the Hopper W3 RS-WGMMA prototype against LR and Machete.

The benchmark is intentionally narrow: M=16, Qwen3.8-27B projection shapes,
FP16 input, and a single idle H200.  QVQ candidates share one native W3
local-ring payload; Machete uses its native symmetric group-128 W4 payload.
Every timed candidate is checked against its own dense reference first.
"""

from __future__ import annotations

import argparse
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--shape",
        choices=tuple(case.name for case in comparison.QWEN38_27B_SHAPES),
        default="qwen38_mlp_down",
    )
    parser.add_argument("--splits", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/h200_lr_next4x/w3_rs_wgmma_qwen38.json"),
    )
    args = parser.parse_args()
    if args.physical_gpu < 0:
        parser.error("--physical-gpu must be non-negative")
    if not args.splits or any(split <= 0 for split in args.splits):
        parser.error("--splits must contain positive integers")
    if args.warmup <= 0 or args.iterations <= 0 or args.idle_samples <= 0:
        parser.error("warmup, iterations, and idle-samples must be positive")
    return args


def _row(*, kernel: str, timing: dict[str, float], metrics: dict[str, float], split: int | None = None) -> dict:
    return {
        "kernel": kernel,
        "split": split,
        **timing,
        **metrics,
    }


def _print_rows(rows: list[dict]) -> None:
    machete_ms = next(row["median_ms"] for row in rows if row["kernel"] == "gptq_machete_w4")
    production_ms = next(row["median_ms"] for row in rows if row["kernel"] == "qvq_lr_w3")
    print("| Kernel | Split | Median ms | P95 ms | vs production LR | xMachete | Max abs |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for row in rows:
        print(
            f"| {row['kernel']} | {row['split'] if row['split'] is not None else '-'} | "
            f"{row['median_ms']:.5f} | {row['p95_ms']:.5f} | "
            f"{production_ms / row['median_ms']:.3f}x | "
            f"{machete_ms / row['median_ms']:.3f}x | {row['max_abs']:.6g} |",
            flush=True,
        )


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

    from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
    from gptqmodel.utils.machete import (
        _validate_machete_device_support,
        machete_runtime_error,
        prewarm_machete_extension,
    )
    from gptqmodel.utils.marlin_scalar_type import scalar_types
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv
    from gptqmodel.utils.qvq_wgmma_cuda import qvq_wgmma_w3_m16, qvq_wgmma_w3_m16_tma

    comparison._visible_gpu_matches(torch, hardware)
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (9, 0) or "H200" not in properties.name:
        raise RuntimeError(f"this benchmark requires H200, got {properties.name} CC {properties.major}.{properties.minor}")
    print(
        f"device: physical={args.physical_gpu} uuid={hardware['uuid']} name={properties.name} "
        f"cc={properties.major}.{properties.minor} sms={properties.multi_processor_count}",
        flush=True,
    )

    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")
    if not _validate_machete_device_support() or not prewarm_machete_extension():
        raise RuntimeError(machete_runtime_error())

    case = comparison._shape_by_name(args.shape, "qwen38_27b")
    generator = torch.Generator().manual_seed(20260830)
    x = (torch.randn((16, case.in_features), generator=generator, dtype=torch.float32) * 0.1).half().cuda()
    trellis, bank_ids, dense_qvq, _ = comparison._qvq_payload(
        torch,
        case,
        bits=3.0,
        seed=20260831,
        device=torch.device("cuda:0"),
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    qvq_reference = x.float() @ dense_qvq.float()

    gptq_source = comparison._gptq_source(
        torch,
        case,
        group_size=comparison.GPTQ_GROUP_SIZE,
        seed=20260832,
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

    comparison._pre_timing_exclusivity_gate(
        physical_gpu=args.physical_gpu,
        gpu_uuid=hardware["uuid"],
        samples=args.idle_samples,
        interval=args.idle_interval,
    )

    rows: list[dict] = []

    def production_call():
        return qvq_cuda_gemv(
            x,
            trellis,
            3.0,
            out_features=case.out_features,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
        )

    actual = production_call()
    torch.cuda.synchronize()
    metrics = comparison._assert_correct(
        kernel="QVQ LR W3",
        actual=actual,
        reference=qvq_reference,
        expected_shape=(16, case.out_features),
        expected_dtype=torch.float32,
        atol=2e-3,
        rtol=0.0,
    )
    rows.append(_row(
        kernel="qvq_lr_w3",
        split=None,
        timing=comparison._cuda_graph_event_timing(
            torch, production_call, warmup=args.warmup, iterations=args.iterations
        ),
        metrics=metrics,
    ))

    for split in args.splits:
        def wgmma_call(split=split):
            return qvq_wgmma_w3_m16(
                x,
                trellis,
                levels,
                bank_ids,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=split,
            )

        actual = wgmma_call()
        torch.cuda.synchronize()
        metrics = comparison._assert_correct(
            kernel=f"QVQ RS-WGMMA W3 split{split}",
            actual=actual,
            reference=qvq_reference,
            expected_shape=(16, case.out_features),
            expected_dtype=torch.float32,
            atol=2e-3,
            rtol=0.0,
        )
        rows.append(_row(
            kernel="qvq_rs_wgmma_w3",
            split=split,
            timing=comparison._cuda_graph_event_timing(
                torch, wgmma_call, warmup=args.warmup, iterations=args.iterations
            ),
            metrics=metrics,
        ))

        def tma_wgmma_call(split=split):
            return qvq_wgmma_w3_m16_tma(
                x,
                trellis,
                levels,
                bank_ids,
                out_features=case.out_features,
                bank_alt_id=3,
                split_count=split,
            )

        actual = tma_wgmma_call()
        torch.cuda.synchronize()
        metrics = comparison._assert_correct(
            kernel=f"QVQ TMA RS-WGMMA W3 split{split}",
            actual=actual,
            reference=qvq_reference,
            expected_shape=(16, case.out_features),
            expected_dtype=torch.float32,
            atol=2e-3,
            rtol=0.0,
        )
        rows.append(_row(
            kernel="qvq_tma_rs_wgmma_w3",
            split=split,
            timing=comparison._cuda_graph_event_timing(
                torch, tma_wgmma_call, warmup=args.warmup, iterations=args.iterations
            ),
            metrics=metrics,
        ))

    def machete_call():
        return machete(x)

    actual = machete_call()
    torch.cuda.synchronize()
    metrics = comparison._assert_correct(
        kernel="Machete W4",
        actual=actual,
        reference=gptq_reference,
        expected_shape=(16, case.out_features),
        expected_dtype=torch.float16,
        atol=2e-2,
        rtol=2e-2,
    )
    rows.append(_row(
        kernel="gptq_machete_w4",
        split=None,
        timing=comparison._cuda_graph_event_timing(
            torch, machete_call, warmup=args.warmup, iterations=args.iterations
        ),
        metrics=metrics,
    ))

    benchmark_utils._verify_source(commit, fingerprint, phase="after WGMMA benchmark completed")
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
        "shape": {
            "name": case.name,
            "m": 16,
            "k": case.in_features,
            "n": case.out_features,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_rows(rows)
    print(f"result: {args.output}", flush=True)
    return payload


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
