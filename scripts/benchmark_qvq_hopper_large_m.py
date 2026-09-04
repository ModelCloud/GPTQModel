#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark complete large-M grouped P32 sites on the physical H100."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (16, 32, 64, 128, 256)
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_hopper_large_m.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(common.GROUPS), default=tuple(common.GROUPS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qvq_hopper_large_m/llama_sites.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value < 1 or value > 4096 for value in args.m_values):
        parser.error("M must be in [1, 4096]")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    if args.idle_samples < 3 or args.idle_interval < 0 or args.idle_memory_mib < 0:
        parser.error("idle gate requires at least three samples and nonnegative thresholds")
    return args


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def _runtime_idle_gate(args: argparse.Namespace, device_info: dict) -> None:
    """Recheck the selected GPU after setup while allowing this process's context."""

    uuid = device_info["uuid"]
    accepted = 0
    attempts = 0
    max_attempts = max(args.idle_samples, 150)
    while accepted < args.idle_samples and attempts < max_attempts:
        attempts += 1
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        foreign = []
        for line in processes.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
                foreign.append(line)
        state = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        utilization = int(state)
        if foreign:
            raise RuntimeError(
                f"H100 runtime idle gate found foreign processes: {foreign}"
            )
        accepted = accepted + 1 if utilization == 0 else 0
        if accepted < args.idle_samples:
            time.sleep(args.idle_interval)
    if accepted != args.idle_samples:
        raise RuntimeError(
            f"H100 did not return to 0% utilization for {args.idle_samples} "
            f"consecutive samples after {attempts} checks"
        )


def _graph_timing(torch, call, args, device_info):
    with torch.inference_mode():
        for _ in range(3):
            outputs = call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        _runtime_idle_gate(args, device_info)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.samples)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.samples)]
        stream = torch.cuda.current_stream()
        for start, end in zip(starts, ends, strict=True):
            start.record(stream)
            for _ in range(args.replays_per_sample):
                graph.replay()
            end.record(stream)
        ends[-1].synchronize()
        values = sorted(
            start.elapsed_time(end) * 1000 / args.replays_per_sample
            for start, end in zip(starts, ends, strict=True)
        )
        snapshots = tuple(output.detach().clone() for output in outputs)
    return {
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p95_us": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "min_us": values[0],
        "max_us": values[-1],
    }, snapshots


def _errors(torch, actual, expected) -> dict:
    differences = tuple(
        (left.float() - right.float()).reshape(-1)
        for left, right in zip(actual, expected, strict=True)
    )
    absolute_sum = sum(float(value.abs().sum().item()) for value in differences)
    square_sum = sum(float(value.square().sum().item()) for value in differences)
    reference_square_sum = sum(float(value.float().square().sum().item()) for value in expected)
    count = sum(value.numel() for value in differences)
    return {
        "mean_abs": absolute_sum / count,
        "rmse": math.sqrt(square_sum / count),
        "relative_l2": math.sqrt(square_sum / max(reference_square_sum, 1e-30)),
        "max_abs": max(float(value.abs().max().item()) for value in differences),
    }


def _main(args: argparse.Namespace) -> None:
    import torch

    from gptqmodel.nn_modules.qlinear.qvq import qvq_dense_oracle_forward
    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups

    device_info = common._assert_h100(torch)
    device = torch.device("cuda:0")
    source_fingerprint = _source_fingerprint()
    inputs = {
        (group, m): (
            torch.randn(
                (m, common.K),
                generator=torch.Generator(device=device).manual_seed(91000 + m),
                device=device,
            )
            * 0.02
        ).half()
        for group in args.groups
        for m in args.m_values
    }

    comparator_timings = {}
    for group in args.groups:
        names, widths, _ = common.GROUPS[group]
        for kernel in ("marlin", "machete"):
            modules = common._gptq_modules(torch, kernel, names, widths, device)
            for m in args.m_values:
                timing, _ = _graph_timing(
                    torch,
                    lambda modules=modules, x=inputs[(group, m)]: tuple(
                        module(x) for module in modules
                    ),
                    args,
                    device_info,
                )
                comparator_timings[(group, m, kernel)] = timing
            del modules
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    for bits in args.rates:
        for group in args.groups:
            names, widths, alt_ids = common.GROUPS[group]
            shared_su = torch.ones(common.K, device=device, dtype=torch.float32)
            children = tuple(
                common._qvq_child(
                    torch,
                    name,
                    width,
                    bits,
                    alt_id,
                    92000 + int(bits * 10) * 10 + index,
                    device,
                    shared_su,
                )
                for index, (name, width, alt_id) in enumerate(
                    zip(names, widths, alt_ids, strict=True)
                )
            )
            for child in children:
                child.SV.fill_(0.002)
                child._dtype_cache_clear()
            parent = common._projection_parent(torch, names, children)
            plain = {}
            for m in args.m_values:
                plain[m], _ = _graph_timing(
                    torch,
                    lambda parent=parent, names=names, x=inputs[(group, m)]: common._call_children(
                        parent, names, x
                    ),
                    args,
                    device_info,
                )
            installed = install_qvq_hopper_groups(
                parent, qkv=group == "qkv", gate_up=group == "gate_up"
            )
            if installed[group] != 1:
                raise RuntimeError(f"failed to install {group} group: {installed}")
            for m in args.m_values:
                timing, actual = _graph_timing(
                    torch,
                    lambda parent=parent, names=names, x=inputs[(group, m)]: common._call_children(
                        parent, names, x
                    ),
                    args,
                    device_info,
                )
                expected = tuple(
                    qvq_dense_oracle_forward(child, inputs[(group, m)], device=device)
                    for child in children
                )
                error = _errors(torch, actual, expected)
                if error["max_abs"] > 2e-3:
                    raise RuntimeError(
                        f"dense P32 accuracy failed for W{bits:g} {group} M{m}: {error}"
                    )
                marlin = comparator_timings[(group, m, "marlin")]
                machete = comparator_timings[(group, m, "machete")]
                logical_flops = 2 * m * common.K * sum(widths)
                row = {
                    "bits": bits,
                    "group": group,
                    "m": m,
                    "k": common.K,
                    "child_n": list(widths),
                    "aggregate_n": sum(widths),
                    "mkn": [m, common.K, sum(widths)],
                    "qvq": timing,
                    "pre_pr_main": plain[m],
                    "marlin_w4": marlin,
                    "machete_w4": machete,
                    "speedup_vs_pre_pr_main": plain[m]["median_us"] / timing["median_us"],
                    "speedup_vs_marlin_w4": marlin["median_us"] / timing["median_us"],
                    "speedup_vs_machete_w4": machete["median_us"] / timing["median_us"],
                    "effective_tflops": logical_flops / (timing["median_us"] * 1e6),
                    "better_than_last_benchmark": timing["median_us"] < plain[m]["median_us"],
                    "dense_oracle_error": error,
                }
                rows.append(row)
                print(
                    f"W{bits:g} {group} MKN={tuple(row['mkn'])}: "
                    f"qvq={timing['median_us']:.3f}us "
                    f"marlin={marlin['median_us']:.3f}us "
                    f"machete={machete['median_us']:.3f}us "
                    f"better={'Yes' if row['better_than_last_benchmark'] else 'No'}",
                    flush=True,
                )
                del expected
            del parent, children
            gc.collect()
            torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint():
        raise RuntimeError("benchmark sources changed during execution")
    payload = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": source_fingerprint,
        "device": device_info,
        "timing": {
            "method": "warmed CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "cpu_time_included": False,
        },
        "workload": "complete Llama 3.2 1B grouped projection sites",
        "comparison": "current grouped row-grid versus pre-change plain per-child QVQ and W4 baselines",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _main(parsed_args)
