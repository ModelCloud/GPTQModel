#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark production A41/R0 QVQ groups against plain QVQ and W4 GPTQ."""

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
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
GROUPS = {
    "qkv": (("q_proj", "k_proj", "v_proj"), (2048, 512, 512), (1, 2, 3)),
    "gate_up": (("gate_proj", "up_proj"), (8192, 8192), (1, 3)),
}
K = 2048
SOURCE_PATHS = (
    Path("gptqmodel/models/base.py"),
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_a41_phase4_production.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(GROUPS), default=tuple(GROUPS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a41_phase4_h100/production_grouped_vs_baselines.json"),
    )
    parser.add_argument(
        "--previous-git-ref",
        default="HEAD",
        help=(
            "git ref containing the preceding result at --output; pass an empty "
            "string to disable prior-run comparisons"
        ),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _physical_h100() -> tuple[str, str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
    )
    matches = []
    for line in output.splitlines():
        uuid, name = (part.strip() for part in line.split(",", 1))
        if "H100" in name:
            matches.append((uuid, name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one physical H100, found {matches}")
    return matches[0]


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((REPO_ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def _previous_benchmark(git_ref: str, output_path: Path):
    if not git_ref:
        return None, {}
    try:
        relative_path = output_path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            "--output must be inside the repository when --previous-git-ref is set"
        ) from exc
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{git_ref}:{relative_path.as_posix()}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None, {}
    payload = json.loads(raw)
    rows = {
        (
            float(row["bits"]),
            row["group"],
            int(row["m"]),
            int(row["k"]),
            tuple(row["child_n"]),
        ): row
        for row in payload.get("rows", ())
    }
    return payload, rows


def _assert_h100(torch):
    expected_uuid, expected_name = _physical_h100()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"1", expected_uuid}:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=1 so the H200 remains hidden")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA-visible device is required")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    if (
        "H100" not in properties.name
        or (properties.major, properties.minor) != (9, 0)
        or actual_uuid != expected_uuid.removeprefix("GPU-").lower()
    ):
        raise RuntimeError(
            f"expected {expected_name} {expected_uuid}, got {properties}"
        )
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
        if (
            len(fields) >= 2
            and fields[0] == expected_uuid
            and int(fields[1]) != os.getpid()
        ):
            foreign.append(line)
    if foreign:
        raise RuntimeError(f"foreign H100 processes detected: {foreign}")
    return {
        "name": properties.name,
        "uuid": expected_uuid,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }


def _graph_timing(
    torch,
    call: Callable[[], Sequence],
    *,
    warmup: int,
    samples: int,
    replays_per_sample: int,
):
    with torch.inference_mode():
        for _ in range(3):
            outputs = call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = call()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
        stream = torch.cuda.current_stream()
        for start, end in zip(starts, ends, strict=True):
            start.record(stream)
            for _ in range(replays_per_sample):
                graph.replay()
            end.record(stream)
        ends[-1].synchronize()
        values = [
            start.elapsed_time(end) / replays_per_sample
            for start, end in zip(starts, ends, strict=True)
        ]
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        snapshots = tuple(output.detach().clone() for output in outputs)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }, snapshots


def _qvq_child(torch, name, width, bits, alt_id, seed, device, shared_su):
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile

    generator = torch.Generator(device=device).manual_seed(seed)
    tiles = (K // 16) * (width // 16)
    words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    tensors = {
        "trellis": torch.randint(
            0,
            1 << 32,
            (tiles, words),
            generator=generator,
            device=device,
            dtype=torch.int64,
        ).to(torch.int32),
        "SU": shared_su.clone(),
        "SV": torch.ones(width, device=device, dtype=torch.float32),
        "bank_ids": pack_qvq_binary_bank_ids(
            torch.randint(
                0,
                2,
                (tiles * 8,),
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
        ),
        "bank_alt_id": torch.tensor([alt_id], device=device, dtype=torch.uint8),
    }
    return QVQLinear.from_tensors(
        bits=bits,
        in_features=K,
        out_features=width,
        name=name,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
    ).eval()


def _projection_parent(torch, names, children):
    parent = torch.nn.Module()
    for name, child in zip(names, children, strict=True):
        setattr(parent, name, child)
    return parent


def _call_children(parent, names, x):
    return tuple(getattr(parent, name)(x) for name in names)


def _gptq_modules(torch, kernel: str, names, widths, device):
    from scripts import benchmark_qwen3_27b_gptq_fp16 as builder

    cls = builder._resolve_linear_cls("cuda", kernel)
    args = Namespace(
        path="cuda",
        cuda_kernel=kernel,
        group_size=128,
        sym=True,
        desc_act=False,
        bias=False,
    )
    return tuple(
        builder._build_module(
            cls,
            args=args,
            case=builder.LayerCase(name, K, width),
            device=device,
            seed=900 + index,
        )
        for index, (name, width) in enumerate(zip(names, widths, strict=True))
    )


def _run(args):
    import torch

    from gptqmodel.nn_modules.qvq_grouped_runtime import (
        install_qvq_hopper_groups,
        qvq_grouped_runtime_telemetry,
    )

    source_fingerprint = _source_fingerprint()
    previous_payload, previous_rows = _previous_benchmark(
        args.previous_git_ref, args.output
    )
    device_info = _assert_h100(torch)
    device = torch.device("cuda:0")
    inputs = {
        (group, m): (
            torch.randn(
                (m, K),
                generator=torch.Generator(device=device).manual_seed(
                    7000 + 10 * m + list(GROUPS).index(group)
                ),
                device=device,
            )
            * 0.02
        ).half()
        for group in args.groups
        for m in args.m_values
    }

    baseline_timings = {}
    for group_name in args.groups:
        names, widths, _ = GROUPS[group_name]
        for kernel in ("marlin", "machete"):
            modules = _gptq_modules(torch, kernel, names, widths, device)
            for m in args.m_values:
                timing, _ = _graph_timing(
                    torch,
                    lambda modules=modules, x=inputs[(group_name, m)]: tuple(
                        module(x) for module in modules
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                )
                baseline_timings[(group_name, m, kernel)] = timing
            del modules
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    for bits in args.rates:
        for group_name in args.groups:
            names, widths, alt_ids = GROUPS[group_name]
            shared_su = torch.ones(K, device=device, dtype=torch.float32)
            children = tuple(
                _qvq_child(
                    torch,
                    name,
                    width,
                    bits,
                    alt_id,
                    10000 + int(bits * 10) * 100 + index,
                    device,
                    shared_su,
                )
                for index, (name, width, alt_id) in enumerate(
                    zip(names, widths, alt_ids, strict=True)
                )
            )
            parent = _projection_parent(torch, names, children)
            plain = {}
            expected = {}
            for m in args.m_values:
                timing, outputs = _graph_timing(
                    torch,
                    lambda parent=parent, names=names, x=inputs[(group_name, m)]: (
                        _call_children(parent, names, x)
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                )
                plain[m] = timing
                expected[m] = outputs

            counts = install_qvq_hopper_groups(
                parent, qkv=group_name == "qkv", gate_up=group_name == "gate_up"
            )
            if counts[group_name] != 1:
                raise RuntimeError(f"failed to install {group_name}: {counts}")
            for m in args.m_values:
                timing, outputs = _graph_timing(
                    torch,
                    lambda parent=parent, names=names, x=inputs[(group_name, m)]: (
                        _call_children(parent, names, x)
                    ),
                    warmup=args.warmup,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                )
                if not all(
                    torch.equal(actual, reference)
                    for actual, reference in zip(outputs, expected[m], strict=True)
                ):
                    raise RuntimeError(
                        f"grouped production output changed at W{bits:g} {group_name} M{m}"
                    )
                marlin = baseline_timings[(group_name, m, "marlin")]
                machete = baseline_timings[(group_name, m, "machete")]
                logical_flops = 2 * m * K * sum(widths)
                previous = previous_rows.get((bits, group_name, m, K, tuple(widths)))
                previous_median_ms = (
                    previous["grouped_qvq"]["median_ms"]
                    if previous is not None
                    else None
                )
                rows.append(
                    {
                        "bits": bits,
                        "group": group_name,
                        "m": m,
                        "k": K,
                        "child_n": list(widths),
                        "aggregate_n": sum(widths),
                        "plain_qvq": plain[m],
                        "grouped_qvq": timing,
                        "marlin_w4": marlin,
                        "machete_w4": machete,
                        "speedup_vs_plain": plain[m]["median_ms"] / timing["median_ms"],
                        "speedup_vs_marlin_w4": marlin["median_ms"]
                        / timing["median_ms"],
                        "speedup_vs_machete_w4": machete["median_ms"]
                        / timing["median_ms"],
                        "plain_effective_tflops": logical_flops
                        / (plain[m]["median_ms"] * 1e9),
                        "grouped_effective_tflops": logical_flops
                        / (timing["median_ms"] * 1e9),
                        "marlin_effective_tflops": logical_flops
                        / (marlin["median_ms"] * 1e9),
                        "machete_effective_tflops": logical_flops
                        / (machete["median_ms"] * 1e9),
                        "better_than_plain_qvq": timing["median_ms"]
                        < plain[m]["median_ms"],
                        "previous_grouped_qvq_median_ms": previous_median_ms,
                        "better_than_previous_benchmark": (
                            timing["median_ms"] < previous_median_ms
                            if previous_median_ms is not None
                            else None
                        ),
                    }
                )
                print(
                    f"W{bits:g} {group_name} M{m}: grouped={timing['median_ms'] * 1000:.3f}us "
                    f"plain={plain[m]['median_ms'] * 1000:.3f}us "
                    f"marlin={marlin['median_ms'] * 1000:.3f}us machete={machete['median_ms'] * 1000:.3f}us",
                    flush=True,
                )
            telemetry = qvq_grouped_runtime_telemetry(parent)
            if len(telemetry) != 1 or telemetry[0]["plain_fallbacks"]:
                raise RuntimeError(f"unexpected grouped runtime telemetry: {telemetry}")
            del parent, children, expected, plain
            gc.collect()
            torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint():
        raise RuntimeError(
            "benchmark sources changed while the H100 matrix was running"
        )
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
        "workload": "Llama 3.2 1B grouped QKV and gate/up full projection forwards",
        "previous_benchmark": (
            {
                "git_ref": args.previous_git_ref,
                "git_base_commit": previous_payload.get("git_base_commit"),
                "source_fingerprint": previous_payload.get("source_fingerprint"),
            }
            if previous_payload is not None
            else None
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}")
    return payload


if __name__ == "__main__":
    _run(_args())
