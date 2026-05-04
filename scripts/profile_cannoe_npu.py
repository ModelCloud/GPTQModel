#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from torch_npu.profiler import (
    AiCMetrics,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    tensorboard_trace_handler,
)

from gptqmodel.nn_modules.qlinear.komodo import KomodoLinear
from gptqmodel.nn_modules.qlinear.cannoe import CannoeLinear, cannoe_plan_asdict
from gptqmodel.utils.torch import HAS_NPU
from scripts.benchmark_komodo_npu_ab import (
    QWEN3_6_27B_GPTQ_CASES,
    _dtype,
    _make_gptq_pair,
    _sync,
)


_KERNEL_METRIC_COLUMNS = (
    "aicore_time(us)",
    "aic_total_cycles",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "cube_utilization(%)",
)


def _case_by_name(name: str):
    for case in QWEN3_6_27B_GPTQ_CASES:
        if case.name == name:
            return case
    valid = ", ".join(case.name for case in QWEN3_6_27B_GPTQ_CASES)
    raise ValueError(f"Unknown case `{name}`. Valid cases: {valid}")


def _aggregate_csv(path: Path, duration_columns: tuple[str, ...]) -> list[dict]:
    totals: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name") or row.get("Type") or "unknown"
            current = totals.setdefault(name, {"name": name, "count": 0})
            current["count"] += 1
            for column in duration_columns:
                raw = (row.get(column) or "0").strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = 0.0
                current[column] = current.get(column, 0.0) + value
    sort_column = duration_columns[-1]
    return sorted(totals.values(), key=lambda item: item.get(sort_column, 0.0), reverse=True)


def _aggregate_kernel_csv(path: Path) -> list[dict]:
    totals: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metric_columns = [column for column in _KERNEL_METRIC_COLUMNS if column in (reader.fieldnames or ())]
        for row in reader:
            name = row.get("Name") or "unknown"
            current = totals.setdefault(name, {"name": name, "count": 0, "Duration(us)": 0.0})
            current["count"] += 1
            try:
                current["Duration(us)"] += float((row.get("Duration(us)") or "0").strip())
            except ValueError:
                pass
            for column in metric_columns:
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                current[column] = current.get(column, 0.0) + value

    for current in totals.values():
        count = current["count"]
        if count <= 0:
            continue
        for column in tuple(current):
            if column.endswith("_ratio") or column.endswith("(%)"):
                current[column] /= count
    return sorted(totals.values(), key=lambda item: item.get("Duration(us)", 0.0), reverse=True)


def _profiler_output_dir(root: Path) -> Path:
    candidates = sorted(root.glob("*/ASCEND_PROFILER_OUTPUT"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No ASCEND_PROFILER_OUTPUT found under {root}")
    return candidates[0]


def _profiler_experimental_config(args):
    if args.profiler_level == "none":
        return _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level0,
            aic_metrics=AiCMetrics.AiCoreNone,
        )

    profiler_level = {
        "level0": ProfilerLevel.Level0,
        "level1": ProfilerLevel.Level1,
        "level2": ProfilerLevel.Level2,
    }[args.profiler_level]
    aic_metrics = getattr(AiCMetrics, args.aic_metrics)
    return _ExperimentalConfig(
        profiler_level=profiler_level,
        aic_metrics=aic_metrics,
        l2_cache=args.l2_cache,
        record_op_args=args.record_op_args,
        op_attr=args.op_attr,
    )


def _profile_case(args) -> dict:
    if not HAS_NPU:
        raise RuntimeError("Ascend NPU is required for Cannoe profiling.")

    mode = args.mode

    if mode == "plain":
        candidate_cls = KomodoLinear
        os.environ["GPTQMODEL_CANNOE_PREFETCH"] = "0"
    elif mode == "cannoe":
        candidate_cls = CannoeLinear
        os.environ["GPTQMODEL_CANNOE_PREFETCH"] = "0"
    elif mode == "cannoe_prefetch":
        candidate_cls = CannoeLinear
        os.environ["GPTQMODEL_CANNOE_PREFETCH"] = "1"
    else:
        raise ValueError(f"Unsupported mode `{args.mode}`.")

    os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "1"
    os.environ["GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"] = "0"
    os.environ["GPTQ_CACHE_DEQUANTIZED_WEIGHTS"] = "0"
    os.environ["GPTQMODEL_KOMODO_PREPACK_TILE_N"] = str(args.tile_n)

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    case = _case_by_name(args.case)
    dtype = _dtype(args.dtype)
    _, candidate = _make_gptq_pair(
        case,
        dtype=dtype,
        device=device,
        seed=args.seed,
        cache_dequantized=False,
        candidate_cls=candidate_cls,
    )
    candidate.eval()
    x = torch.randn(args.tokens or case.tokens, case.in_features, dtype=dtype, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            candidate(x)
        _sync(device)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            on_trace_ready=tensorboard_trace_handler(str(args.output_dir)),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            experimental_config=_profiler_experimental_config(args),
        ):
            for _ in range(args.iters):
                candidate(x)
            _sync(device)

    profiler_dir = _profiler_output_dir(args.output_dir)
    operator_csv = profiler_dir / "operator_details.csv"
    kernel_csv = profiler_dir / "kernel_details.csv"
    operator_top = _aggregate_csv(
        operator_csv,
        ("Host Total Duration(us)", "Device Total Duration(us)"),
    )[: args.top_n]
    kernel_top = _aggregate_kernel_csv(kernel_csv)[: args.top_n]

    payload = {
        "mode": mode,
        "case": case.name,
        "device": str(device),
        "dtype": args.dtype,
        "iters": args.iters,
        "tile_n": args.tile_n,
        "profiler_level": args.profiler_level,
        "aic_metrics": args.aic_metrics if args.profiler_level != "none" else None,
        "l2_cache": args.l2_cache,
        "profiler_dir": str(profiler_dir),
        "operator_top": operator_top,
        "kernel_top": kernel_top,
        "cannoe_path": getattr(candidate, "_last_cann_path", None),
        "cannoe_plan": cannoe_plan_asdict(getattr(candidate, "_last_cann_plan", None)),
    }
    summary_path = args.output_dir / f"{mode}_{case.name}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile plain Komodo vs Cannoe with CANN profiler output.")
    parser.add_argument(
        "--mode",
        choices=("plain", "cannoe", "cannoe_prefetch"),
        default="cannoe",
    )
    parser.add_argument("--case", default="qwen3_6_27b_gptq_down_proj")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--tile-n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profiler-level", choices=("none", "level0", "level1", "level2"), default="none")
    parser.add_argument(
        "--aic-metrics",
        choices=tuple(name for name in dir(AiCMetrics) if not name.startswith("_")),
        default="PipeUtilization",
    )
    parser.add_argument("--l2-cache", action="store_true")
    parser.add_argument("--record-op-args", action="store_true")
    parser.add_argument("--op-attr", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cannoe_profile"))
    args = parser.parse_args()

    payload = _profile_case(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
