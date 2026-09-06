#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Matched complete-window/rank8 benchmark; synthetic inputs prove kernel behavior only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from gpu_idle_preflight import (
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    idle = bootstrap_gpu_idle_preflight()
    parser = argparse.ArgumentParser(description=__doc__)
    add_gpu_idle_preflight_args(parser)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 128, 512, 2048])
    parser.add_argument("--bits", type=float, default=3)
    parser.add_argument("--kernel", default="separate_reference")
    parser.add_argument("--projection", default="separate_reference")
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--block-m", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=0)
    parser.add_argument("--chunk-m", type=int, default=0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--tuning-cache", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--activation-file", type=Path,
                        help="Replay audit_1/audit_2 matrices saved by evaluate_qvq_window_rank8.py")
    parser.add_argument("--profile", choices=["off", "on"])
    parser.add_argument("--profile-repeats", type=int, default=200)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile and len(args.m) != 1:
        parser.error("profiling requires one M and one quality mode per process")
    if args.activation_file and not args.package:
        parser.error("--activation-file requires --package")
    if (
        min(args.k, args.n, *args.m, args.samples, args.replays, args.profile_repeats)
        <= 0
    ):
        parser.error("dimensions and iteration counts must be positive")

    import torch

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq_rank8 import (
        CONTRACT,
        P32WindowConfig,
        _base,
        _digest,
        _encode,
        load_window_package,
        prepare_rank8,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(137)
    props = torch.cuda.get_device_properties(0)
    if idle is not None and str(props.uuid).removeprefix(
        "GPU-"
    ) != idle.uuid.removeprefix("GPU-"):
        raise RuntimeError("physical device mapping mismatch")
    if args.package:
        layer = load_window_package(
            torch.load(args.package, weights_only=True), device="cuda"
        )
    else:
        layer = (
            QVQLinear(
                bits=args.bits,
                in_features=args.k,
                out_features=args.n,
                bank_count=2,
                v2b2_p32=True,
            )
            .eval()
            .cuda()
        )
        layer.trellis.random_(-2147483648, 2147483647)
        # Known algebra fixture, never exported/promoted as a fitted model.
        layer.rank8_A = (torch.randn(args.k, 8, device="cuda") * 0.02).half()
        layer.rank8_B = (torch.randn(8, args.n, device="cuda") * 0.02).half()
        tensors, metadata = _base(layer)
        layer.rank8_metadata = _encode(
            {
                "base_hash": _digest(tensors, metadata),
                "fit_contract": CONTRACT,
                "validated": True,
                "selected": True,
                "fixture": "synthetic kernel algebra",
                "factors_hash": _digest({"A": layer.rank8_A, "B": layer.rank8_B}, {}),
            },
            "cuda",
        )
    layer.post_init()
    import triton

    from gptqmodel.utils.qvq_cuda import _QVQ_CUDA_TORCH_OPS_EXTENSION
    from gptqmodel.utils.qvq_wgmma_cuda import _QVQ_WGMMA_EXTENSION

    build_identity = {
        "qvq": _QVQ_CUDA_TORCH_OPS_EXTENSION.build_root().name,
        "wgmma": _QVQ_WGMMA_EXTENSION.build_root().name,
        "triton": triton.__version__,
        "driver": subprocess.check_output(
            [
                "nvidia-smi",
                "--id=GPU-" + str(props.uuid).removeprefix("GPU-"),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    paths = [
        "gptqmodel/quantization/qvq_rank8.py",
        "gptqmodel/nn_modules/qlinear/qvq.py",
        "gptqmodel/nn_modules/qvq_grouped_runtime.py",
        "gptqmodel/utils/qvq_rank8_triton.py",
        "gptqmodel/utils/qvq_wgmma_cuda.py",
        "gptqmodel/quantization/qvq_window_tuning.py",
        "gptqmodel_ext/qvq/qvq_wgmma_cuda.cu",
    ]
    report = {
        "scope": (
            "real fitted module, cyclic replay of captured activations; not full-model prefill or quality"
            if args.activation_file else "synthetic kernel/performance, not model quality"
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "sources": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in paths
            if (ROOT / p).exists()
        },
        "hardware": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
            "memory": props.total_memory,
            "capability": [props.major, props.minor],
        },
        "software": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
        "build_identity": build_identity,
        "preflight": None if idle is None else idle.as_dict(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    activation_cases = None
    if args.activation_file:
        saved = torch.load(args.activation_file, weights_only=True, map_location="cpu")
        activation_cases = [saved[key] for key in ("audit_1", "audit_2")]
        if any(
            x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != layer.in_features
            or x.dtype != torch.float16 or not torch.isfinite(x).all()
            for x in activation_cases
        ):
            raise ValueError("benchmark audit activations must be finite FP16 [rows,K] matrices")
        report["activation_source"] = {
            "path": str(args.activation_file),
            "sha256": hashlib.sha256(args.activation_file.read_bytes()).hexdigest(),
            "rows": [x.shape[0] for x in activation_cases],
            "expansion": "cyclic row replay to requested M; not a full-model prefill",
        }

    def sample_candidate(fn, activation):
        for _ in range(3):
            expected = fn(activation)
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            candidate_output = fn(activation)
        candidate_graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(candidate_output, expected):
            raise RuntimeError("autotune eager/graph mismatch")
        if idle is not None:
            recheck_gpu_exclusivity(idle)
        measurements = []
        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                candidate_graph.replay()
            end.record()
            end.synchronize()
            measurements.append(start.elapsed_time(end) * 1000 / args.replays)
        return measurements

    print(
        " M      K      N mode kernel                 projection          mean_us median_us   p95_us  MAE       max",
        flush=True,
    )
    for m in args.m:
        if activation_cases is None:
            x = torch.randn(m, layer.in_features, device="cuda", dtype=torch.float16) * 0.02
            validation_x = torch.randn_like(x) * 0.02 if args.autotune else None
        else:
            x, validation_x = [
                rows[torch.arange(m) % rows.shape[0]].contiguous().cuda() for rows in activation_cases
            ]
        # A profiler range owns its process. Do not start another quality
        # mode while profiler replay buffers may remain resident afterward.
        for mode in ((args.profile,) if args.profile else ("off", "on")):
            prepare_rank8(
                layer, P32WindowConfig(algorithm=args.algorithm, recovery_mode=mode)
            )
            reference = layer(x)
            prepare_rank8(
                layer,
                P32WindowConfig(
                    algorithm=args.algorithm,
                    recovery_mode=mode,
                    recovery_kernel=args.kernel,
                    recovery_projection=args.projection,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    chunk_m=args.chunk_m,
                ),
            )
            if args.autotune:
                from gptqmodel.quantization.qvq_window_tuning import tune_window_kernel

                tuning = tune_window_kernel(
                    layer,
                    (x, validation_x),
                    benchmark=sample_candidate,
                    build_id=hashlib.sha256(
                        json.dumps(
                            [report["sources"], build_identity], sort_keys=True
                        ).encode()
                    ).hexdigest(),
                    cache_dir=args.tuning_cache,
                )
                report.setdefault("tuning", []).append(
                    {
                        "m": m,
                        "mode": mode,
                        "cache_hit": tuning.cache_hit,
                        "report": tuning.report,
                    }
                )
                for row in tuning.report["rows"]:
                    if row.get("exception_review_required"):
                        print(
                            "ACCURACY/PERFORMANCE EXCEPTION REVIEW REQUIRED: "
                            + json.dumps(row),
                            flush=True,
                        )
            selected_config = layer._p32_window_config
            for _ in range(10):
                output = layer(x)
            delta = (output.float() - reference.float()).abs()
            mae, maximum = delta.mean().item(), delta.max().item()
            if not torch.isfinite(output).all() or mae > 2e-3 or maximum > 0.046875:
                raise RuntimeError(
                    f"local correctness gate failed: M={m} {mode} MAE={mae} max={maximum}"
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer(x)
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(captured, output):
                raise RuntimeError("eager/graph mismatch")
            if idle is not None:
                recheck_gpu_exclusivity(idle)
            if args.profile == mode:
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(args.profile_repeats):
                    with torch.cuda.nvtx.range("p32_rank8_operator"):
                        layer(x)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            samples = []
            for _ in range(args.samples):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / args.replays)
            record = {
                "m": m,
                "k": layer.in_features,
                "n": layer.out_features,
                "bits": layer.bits,
                "mode": mode,
                "kernel": selected_config.recovery_kernel,
                "projection": selected_config.recovery_projection,
                "algorithm": selected_config.algorithm,
                "block_m": selected_config.block_m,
                "block_n": selected_config.block_n,
                "chunk_m": selected_config.chunk_m,
                "mean_us": statistics.mean(samples),
                "median_us": statistics.median(samples),
                "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
                "mae": mae,
                "max": maximum,
                "samples_us": samples,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            report["rows"].append(record)
            print(
                f"{m:4} {layer.in_features:6} {layer.out_features:6} {mode:4} {record['kernel']:22} "
                f"{record['projection']:19} "
                f"{record['mean_us']:8.3f} {record['median_us']:9.3f} {record['p95_us']:8.3f} "
                f"{mae:.3g} {maximum:.3g}",
                flush=True,
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del graph, captured, reference, output


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
