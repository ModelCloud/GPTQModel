#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure exact, on-demand FP16, and cached FP8 P32 prefill on H100."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts import benchmark_qvq_hopper_large_m as large_m

RATES = (2.0, 2.5, 3.0, 3.5)
DECODE_M = (1, 2, 4, 8, 16)
PREFILL_M = (8192, 16384)
LLAMA_LAYERS = 16


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument(
        "--prefill-m-values", nargs="+", type=int, default=PREFILL_M
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--replays-per-sample", type=int, default=10)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/qvq_hopper_large_m/llama_qkv_prefill_decode_fp8.json"
        ),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(m < 8192 for m in args.prefill_m_values):
        parser.error("every prefill M must activate the M >= 8192 cache gate")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _median_us(timing: dict) -> float:
    return float(timing["median_us"])


def _time_comparators(torch, inputs, args, device_info, device):
    names, widths, _ = common.GROUPS["qkv"]
    timings = {}
    for kernel in ("marlin", "machete"):
        modules = common._gptq_modules(torch, kernel, names, widths, device)
        for m, x in inputs.items():
            timing, snapshots = large_m._graph_timing(
                torch,
                lambda modules=modules, x=x: tuple(module(x) for module in modules),
                args,
                device_info,
            )
            timings[(kernel, m)] = timing
            del snapshots
        del modules
        gc.collect()
        torch.cuda.empty_cache()
    return timings


def _build_children(torch, bits, device):
    names, widths, alt_ids = common.GROUPS["qkv"]
    shared_su = torch.ones(common.K, device=device, dtype=torch.float32)
    children = tuple(
        common._qvq_child(
            torch,
            name,
            width,
            bits,
            alt_id,
            96000 + int(bits * 10) * 10 + index,
            device,
            shared_su,
        )
        for index, (name, width, alt_id) in enumerate(
            zip(names, widths, alt_ids, strict=True)
        )
    )
    for child in children:
        child.SV.fill_(0.002)
        child.bias = None
        child._dtype_cache_clear()
    return names, widths, children


def _cold_build(torch, call, device) -> dict:
    gc.collect()
    torch.cuda.synchronize(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter_ns()
    outputs = call()
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter_ns() - wall_start) / 1000.0
    cuda_us = start.elapsed_time(end) * 1000.0
    del outputs
    gc.collect()
    torch.cuda.synchronize(device)
    return {
        "cuda_us": cuda_us,
        "wall_us": wall_us,
        "live_allocated_delta_bytes": int(torch.cuda.memory_allocated(device))
        - allocated_before,
        "peak_allocated_delta_bytes": int(torch.cuda.max_memory_allocated(device))
        - allocated_before,
    }


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward
    from gptqmodel.nn_modules.qvq_grouped_runtime import (
        install_qvq_hopper_groups,
        qvq_grouped_runtime_telemetry,
    )

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    inputs = {
        m: (
            torch.randn(
                (m, common.K),
                generator=torch.Generator(device=device).manual_seed(95000 + m),
                device=device,
            )
            * 0.02
        ).half()
        for m in (*DECODE_M, *args.prefill_m_values)
    }
    comparators = _time_comparators(torch, inputs, args, device_info, device)
    rows = []
    memory = []

    for bits in args.rates:
        names, widths, children = _build_children(torch, bits, device)
        parent = common._projection_parent(torch, names, children)
        installed = install_qvq_hopper_groups(parent, qkv=True, gate_up=False)
        if installed != {"qkv": 1}:
            raise RuntimeError(f"failed to install QKV group: {installed}")
        runtime = parent.q_proj._gptqmodel_qvq_grouped_runtime

        def call(m, parent=parent, names=names):
            return common._call_children(parent, names, inputs[m])

        os.environ["QVQ_HOPPER_FP8_PREFILL"] = "0"
        os.environ["QVQ_HOPPER_FP16_PREFILL"] = "0"
        os.environ["QVQ_HOPPER_FP16_PREFILL_NATIVE"] = "0"
        exact_prefill = {}
        for m in args.prefill_m_values:
            exact_prefill[m], snapshots = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            del snapshots
        decode_before = {}
        for m in DECODE_M:
            decode_before[m], snapshots = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            del snapshots

        os.environ["QVQ_HOPPER_FP16_PREFILL"] = "1"
        first_prefill_m = min(args.prefill_m_values)
        fp16_call_memory = _cold_build(
            torch,
            lambda first_prefill_m=first_prefill_m: call(first_prefill_m),
            device,
        )
        fp16_prefill = {}
        fp16_prefill_errors = {}
        for m in args.prefill_m_values:
            launches_before = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp16_prefill_launches"
                ]
            )
            fp16_prefill[m], actual = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            expected = tuple(
                qvq_dense_oracle_forward(child, inputs[m], device=device)
                for child in children
            )
            fp16_prefill_errors[m] = large_m._errors(torch, actual, expected)
            if fp16_prefill_errors[m]["max_abs"] > 2e-3:
                raise RuntimeError(
                    f"W{bits:g} M{m} FP16 prefill oracle failure: "
                    f"{fp16_prefill_errors[m]}"
                )
            launches_after = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp16_prefill_launches"
                ]
            )
            if launches_after <= launches_before:
                raise RuntimeError(f"M{m} did not execute on-demand FP16 prefill")
            del actual, expected
        fp16_temporary_bytes = int(
            qvq_grouped_runtime_telemetry(parent)[0][
                "h100_fp16_prefill_temporary_bytes"
            ]
        )

        os.environ["QVQ_HOPPER_FP16_PREFILL_NATIVE"] = "1"
        native_fp16_call_memory = _cold_build(
            torch,
            lambda first_prefill_m=first_prefill_m: call(first_prefill_m),
            device,
        )
        native_fp16_prefill = {}
        native_fp16_prefill_errors = {}
        for m in args.prefill_m_values:
            launches_before = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp16_prefill_native_launches"
                ]
            )
            native_fp16_prefill[m], actual = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            expected = tuple(
                qvq_dense_oracle_forward(child, inputs[m], device=device)
                for child in children
            )
            native_fp16_prefill_errors[m] = large_m._errors(
                torch, actual, expected
            )
            if native_fp16_prefill_errors[m]["max_abs"] > 2e-3:
                raise RuntimeError(
                    f"W{bits:g} M{m} native FP16 prefill oracle failure: "
                    f"{native_fp16_prefill_errors[m]}"
                )
            launches_after = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp16_prefill_native_launches"
                ]
            )
            if launches_after <= launches_before:
                raise RuntimeError(f"M{m} did not execute native FP16 prefill")
            del actual, expected
        native_fp16_scratch_bytes = int(
            qvq_grouped_runtime_telemetry(parent)[0][
                "h100_fp16_prefill_native_scratch_bytes"
            ]
        )

        os.environ["QVQ_HOPPER_FP16_PREFILL"] = "0"
        os.environ["QVQ_HOPPER_FP16_PREFILL_NATIVE"] = "0"
        os.environ["QVQ_HOPPER_FP8_PREFILL"] = "0"
        os.environ["QVQ_HOPPER_FP8_PREFILL_ON_DEMAND"] = "1"
        ondemand_m_values = tuple(m for m in args.prefill_m_values if m >= 16384)
        ondemand_fp8_call_memory = None
        ondemand_fp8 = {}
        ondemand_fp8_errors = {}
        ondemand_fp8_retained_bytes = 0
        ondemand_fp8_scratch_bytes = 0
        if ondemand_m_values:
            first_ondemand_m = min(ondemand_m_values)
            ondemand_fp8_call_memory = _cold_build(
                torch,
                lambda first_ondemand_m=first_ondemand_m: call(first_ondemand_m),
                device,
            )
            for m in ondemand_m_values:
                launches_before = int(
                    qvq_grouped_runtime_telemetry(parent)[0][
                        "h100_fp8_ondemand_launches"
                    ]
                )
                ondemand_fp8[m], actual = large_m._graph_timing(
                    torch,
                    lambda m=m: call(m),
                    args,
                    device_info,
                )
                expected = tuple(
                    qvq_dense_oracle_forward(child, inputs[m], device=device)
                    for child in children
                )
                ondemand_fp8_errors[m] = large_m._errors(torch, actual, expected)
                if ondemand_fp8_errors[m]["max_abs"] > 2e-3:
                    raise RuntimeError(
                        f"W{bits:g} M{m} on-demand FP8 oracle failure: "
                        f"{ondemand_fp8_errors[m]}"
                    )
                launches_after = int(
                    qvq_grouped_runtime_telemetry(parent)[0][
                        "h100_fp8_ondemand_launches"
                    ]
                )
                if launches_after <= launches_before:
                    raise RuntimeError(f"M{m} did not execute on-demand FP8 prefill")
                del actual, expected
            ondemand_telemetry = qvq_grouped_runtime_telemetry(parent)[0]
            ondemand_fp8_retained_bytes = int(
                ondemand_telemetry["h100_fp8_ondemand_retained_bytes"]
            )
            ondemand_fp8_scratch_bytes = int(
                ondemand_telemetry["h100_fp8_ondemand_scratch_bytes"]
            )

        os.environ["QVQ_HOPPER_FP8_PREFILL_ON_DEMAND"] = "0"
        os.environ["QVQ_HOPPER_FP8_PREFILL"] = "1"
        launches_before = int(
            qvq_grouped_runtime_telemetry(parent)[0]["h100_fp8_prefill_launches"]
        )
        cold = _cold_build(
            torch, lambda first_prefill_m=first_prefill_m: call(first_prefill_m), device
        )
        telemetry = qvq_grouped_runtime_telemetry(parent)[0]
        if int(telemetry["h100_fp8_prefill_launches"]) <= launches_before:
            raise RuntimeError("M >= 8192 did not activate folded FP8 prefill")
        cache_bytes = int(telemetry["h100_fp8_prefill_bytes"])
        if cache_bytes != 6_291_464:
            raise RuntimeError(f"unexpected folded-prefill cache size: {cache_bytes}")

        fast_prefill = {}
        prefill_errors = {}
        for m in args.prefill_m_values:
            launches_before = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp8_prefill_launches"
                ]
            )
            fast_prefill[m], actual = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            expected = tuple(
                qvq_dense_oracle_forward(child, inputs[m], device=device)
                for child in children
            )
            prefill_errors[m] = large_m._errors(torch, actual, expected)
            if prefill_errors[m]["max_abs"] > 2e-3:
                raise RuntimeError(
                    f"W{bits:g} M{m} prefill oracle failure: {prefill_errors[m]}"
                )
            launches_after = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp8_prefill_launches"
                ]
            )
            if launches_after <= launches_before:
                raise RuntimeError(f"M{m} did not execute folded FP8 prefill")
            del actual, expected

        decode_after = {}
        for m in DECODE_M:
            launches_before = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp8_prefill_launches"
                ]
            )
            decode_after[m], snapshots = large_m._graph_timing(
                torch,
                lambda m=m: call(m),
                args,
                device_info,
            )
            launches_after = int(
                qvq_grouped_runtime_telemetry(parent)[0][
                    "h100_fp8_prefill_launches"
                ]
            )
            if launches_after != launches_before:
                raise RuntimeError(f"decode M{m} unexpectedly executed FP8 prefill")
            del snapshots
            row = {
                "phase": "decode_after_prefill",
                "bits": bits,
                "m": m,
                "mkn": [m, common.K, sum(widths)],
                "qvq": decode_after[m],
                "qvq_before_prefill_cache": decode_before[m],
                "marlin_w4": comparators[("marlin", m)],
                "machete_w4": comparators[("machete", m)],
                "speedup_vs_marlin_w4": _median_us(comparators[("marlin", m)])
                / _median_us(decode_after[m]),
                "speedup_vs_machete_w4": _median_us(comparators[("machete", m)])
                / _median_us(decode_after[m]),
                "speedup_vs_before_prefill_cache": _median_us(decode_before[m])
                / _median_us(decode_after[m]),
                "better_than_last_benchmark": _median_us(decode_after[m])
                < _median_us(decode_before[m]),
            }
            rows.append(row)

        for m in args.prefill_m_values:
            rows.append(
                {
                    "phase": "on_demand_fp16_prefill",
                    "bits": bits,
                    "m": m,
                    "mkn": [m, common.K, sum(widths)],
                    "qvq": fp16_prefill[m],
                    "qvq_exact_row_multiplex": exact_prefill[m],
                    "marlin_w4": comparators[("marlin", m)],
                    "machete_w4": comparators[("machete", m)],
                    "speedup_vs_exact_row_multiplex": _median_us(
                        exact_prefill[m]
                    )
                    / _median_us(fp16_prefill[m]),
                    "speedup_vs_marlin_w4": _median_us(
                        comparators[("marlin", m)]
                    )
                    / _median_us(fp16_prefill[m]),
                    "speedup_vs_machete_w4": _median_us(
                        comparators[("machete", m)]
                    )
                    / _median_us(fp16_prefill[m]),
                    "better_than_last_benchmark": _median_us(fp16_prefill[m])
                    < _median_us(exact_prefill[m]),
                    "dense_oracle_error": fp16_prefill_errors[m],
                }
            )
            rows.append(
                {
                    "phase": "native_fp16_prefill",
                    "bits": bits,
                    "m": m,
                    "mkn": [m, common.K, sum(widths)],
                    "qvq": native_fp16_prefill[m],
                    "qvq_phase1_fp16": fp16_prefill[m],
                    "qvq_exact_row_multiplex": exact_prefill[m],
                    "marlin_w4": comparators[("marlin", m)],
                    "machete_w4": comparators[("machete", m)],
                    "speedup_vs_phase1_fp16": _median_us(fp16_prefill[m])
                    / _median_us(native_fp16_prefill[m]),
                    "speedup_vs_exact_row_multiplex": _median_us(
                        exact_prefill[m]
                    )
                    / _median_us(native_fp16_prefill[m]),
                    "speedup_vs_marlin_w4": _median_us(
                        comparators[("marlin", m)]
                    )
                    / _median_us(native_fp16_prefill[m]),
                    "speedup_vs_machete_w4": _median_us(
                        comparators[("machete", m)]
                    )
                    / _median_us(native_fp16_prefill[m]),
                    "better_than_last_benchmark": _median_us(
                        native_fp16_prefill[m]
                    )
                    < _median_us(fp16_prefill[m]),
                    "dense_oracle_error": native_fp16_prefill_errors[m],
                }
            )
            if m in ondemand_fp8:
                rows.append(
                    {
                        "phase": "native_ondemand_fp8_prefill",
                        "bits": bits,
                        "m": m,
                        "mkn": [m, common.K, sum(widths)],
                        "qvq": ondemand_fp8[m],
                        "qvq_phase2_fp16": native_fp16_prefill[m],
                        "marlin_w4": comparators[("marlin", m)],
                        "machete_w4": comparators[("machete", m)],
                        "speedup_vs_phase2_fp16": _median_us(
                            native_fp16_prefill[m]
                        )
                        / _median_us(ondemand_fp8[m]),
                        "speedup_vs_marlin_w4": _median_us(
                            comparators[("marlin", m)]
                        )
                        / _median_us(ondemand_fp8[m]),
                        "speedup_vs_machete_w4": _median_us(
                            comparators[("machete", m)]
                        )
                        / _median_us(ondemand_fp8[m]),
                        "better_than_last_benchmark": _median_us(
                            ondemand_fp8[m]
                        )
                        < _median_us(native_fp16_prefill[m]),
                        "dense_oracle_error": ondemand_fp8_errors[m],
                    }
                )
            rows.append(
                {
                    "phase": "warm_prefill",
                    "bits": bits,
                    "m": m,
                    "mkn": [m, common.K, sum(widths)],
                    "qvq": fast_prefill[m],
                    "qvq_exact_row_multiplex": exact_prefill[m],
                    "marlin_w4": comparators[("marlin", m)],
                    "machete_w4": comparators[("machete", m)],
                    "speedup_vs_exact_row_multiplex": _median_us(exact_prefill[m])
                    / _median_us(fast_prefill[m]),
                    "speedup_vs_marlin_w4": _median_us(comparators[("marlin", m)])
                    / _median_us(fast_prefill[m]),
                    "speedup_vs_machete_w4": _median_us(
                        comparators[("machete", m)]
                    )
                    / _median_us(fast_prefill[m]),
                    "better_than_last_benchmark": _median_us(fast_prefill[m])
                    < _median_us(exact_prefill[m]),
                    "dense_oracle_error": prefill_errors[m],
                }
            )
        memory.append(
            {
                "bits": bits,
                "ideal_packed_source_bytes_per_qkv_group": int(
                    common.K * sum(widths) * bits / 8
                ),
                "cache_bytes_per_qkv_group": cache_bytes,
                "cache_mib_per_qkv_group": cache_bytes / 2**20,
                "cache_mib_for_16_layers": cache_bytes * LLAMA_LAYERS / 2**20,
                "cache_percent_of_device_for_16_layers": (
                    100 * cache_bytes * LLAMA_LAYERS / device_info["memory_bytes"]
                ),
                "cache_to_ideal_packed_source_ratio": cache_bytes
                / (common.K * sum(widths) * bits / 8),
                "on_demand_fp16_temporary_bytes": fp16_temporary_bytes,
                "on_demand_fp16_call": fp16_call_memory,
                "native_fp16_scratch_bytes": native_fp16_scratch_bytes,
                "native_fp16_call": native_fp16_call_memory,
                "ondemand_fp8_retained_bytes": ondemand_fp8_retained_bytes,
                "ondemand_fp8_scratch_bytes": ondemand_fp8_scratch_bytes,
                "ondemand_fp8_call": ondemand_fp8_call_memory,
                "cold_build": cold,
            }
        )
        runtime.invalidate()
        del parent, children, runtime
        gc.collect()
        torch.cuda.empty_cache()

    os.environ.pop("QVQ_HOPPER_FP8_PREFILL", None)
    os.environ.pop("QVQ_HOPPER_FP16_PREFILL", None)
    os.environ.pop("QVQ_HOPPER_FP16_PREFILL_NATIVE", None)
    os.environ.pop("QVQ_HOPPER_FP8_PREFILL_ON_DEMAND", None)
    payload = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "cpu_time_included": False,
        },
        "comparator_contract": "three ordinary Q/K/V projection calls",
        "rows": rows,
        "memory": memory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for row in rows:
        print(
            f"{row['phase']} W{row['bits']:g} MKN={tuple(row['mkn'])}: "
            f"qvq={_median_us(row['qvq']):.3f}us "
            f"marlin={row['speedup_vs_marlin_w4']:.3f}x "
            f"machete={row['speedup_vs_machete_w4']:.3f}x "
            f"better={'Yes' if row['better_than_last_benchmark'] else 'No'}",
            flush=True,
        )
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
