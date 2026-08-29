# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sweep the native QVQ V2B2-P32-LR CUDA GEMV on physical GPUs.

The worker entrypoint intentionally performs its nvidia-smi idle gate before
importing torch. ``--all`` launches one PCI-ordered, physical-index-pinned
worker per GPU and verifies the resulting CUDA UUID before timing kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SHAPES = (
    ("m1_narrow", 2048, 256),
    ("m1_wide", 2048, 8192),
    ("m4_wide", 2048, 8192),
    ("m16_mid", 8192, 2048),
    ("mlp_down", 4096, 11008),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Launch one UUID-pinned worker for every listed GPU.")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--bits", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), nargs="+", default=["float16", "bfloat16"])
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--split-counts", type=int, nargs="+", default=[0], help="0 selects the automatic split policy.")
    parser.add_argument(
        "--no-non-lr",
        action="store_true",
        help="Skip the legacy V2B2-P32 regression path.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/qvq_cuda_lr_sweep"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.all and args.worker:
        parser.error("--all and --worker are mutually exclusive")
    if not args.all and args.physical_gpu is None:
        parser.error("use --all or --physical-gpu")
    return args


def _query_gpu(physical_gpu: int) -> dict[str, str]:
    fields = "index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={physical_gpu}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != len(fields.split(",")):
        raise RuntimeError(f"unexpected nvidia-smi output for physical GPU {physical_gpu}: {output!r}")
    return dict(zip(fields.split(","), values, strict=True))


def _idle_preflight(physical_gpu: int, samples: int, interval: float, memory_tolerance_mib: int) -> dict[str, str]:
    accepted = None
    for sample in range(samples):
        accepted = _query_gpu(physical_gpu)
        memory_used = int(accepted["memory.used"])
        utilization = int(accepted["utilization.gpu"])
        if memory_used > memory_tolerance_mib or utilization != 0:
            raise RuntimeError(
                f"physical GPU {physical_gpu} failed idle gate: memory={memory_used} MiB "
                f"(tolerance={memory_tolerance_mib}), utilization={utilization}%"
            )
        if sample + 1 < samples:
            time.sleep(interval)
    assert accepted is not None
    print(
        "idle gate: "
        f"physical={accepted['index']} pci={accepted['pci.bus_id']} uuid={accepted['uuid']} "
        f"name={accepted['name']} memory={accepted['memory.used']}MiB utilization={accepted['utilization.gpu']}% "
        f"samples={samples} threshold={memory_tolerance_mib}MiB",
        flush=True,
    )
    return accepted


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _event_timing(torch, fn, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def _rows_for_m(m: int) -> int:
    return 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32


def _split_count(m: int, k: int, n: int, sms: int, major: int) -> int:
    rows = _rows_for_m(m)
    base_blocks = (n // 8) * ((m + rows - 1) // rows)
    k_tiles = k // 32
    if major >= 12:
        if rows == 1:
            split_count = 4 if base_blocks < 64 or base_blocks >= 384 else 16
        elif base_blocks < 64:
            split_count = 32 if rows <= 8 else 8
        elif base_blocks < 384:
            split_count = 16
        else:
            split_count = 1
    elif base_blocks >= 384:
        split_count = 4 if rows == 1 and k_tiles >= 128 else 1
    elif rows == 1:
        split_count = 16 if base_blocks < 64 else 8
    elif base_blocks < 64:
        split_count = 8
    else:
        split_count = 4
    return max(1, min(split_count, k_tiles, 64))


def _legacy_split_count(m: int, k: int, n: int, sms: int) -> int:
    """Mirror the legacy V2B2-P32 launch policy for regression labels."""

    base_blocks = (n // 16) * ((m + 31) // 32)
    k_tiles = k // 16
    if base_blocks >= 384:
        return 1
    return min((sms * 6 + base_blocks - 1) // base_blocks, k_tiles, 64)


def _metrics(actual, reference) -> dict[str, float]:
    delta = actual.float() - reference.float()
    return {
        "mae": delta.abs().mean().item(),
        "mse": delta.square().mean().item(),
        "max_abs": delta.abs().max().item(),
        "rel_l2": (delta.square().sum() / reference.float().square().sum().clamp_min(1e-12)).sqrt().item(),
    }


def _regression_report(rows: list[dict]) -> dict:
    """Pair LR32 and legacy timings using the same shape, data, and device."""

    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (
            row.get("physical_gpu"),
            row.get("shape"),
            row.get("bits"),
            row.get("dtype"),
            row.get("m"),
        )
        grouped.setdefault(key, {})[row.get("path", "")] = row

    regressions = []
    for key, paths in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        lr = paths.get("lr_native_auto")
        legacy = paths.get("non_lr_native")
        if lr is None or legacy is None:
            continue
        speedup = legacy["median_ms"] / lr["median_ms"]
        regressions.append({
            "physical_gpu": lr["physical_gpu"],
            "pci_bus_id": lr["pci_bus_id"],
            "uuid": lr["uuid"],
            "gpu_name": lr["gpu_name"],
            "compute_capability": lr["compute_capability"],
            "sm_count": lr["sm_count"],
            "shape": lr["shape"],
            "k": lr["k"],
            "n": lr["n"],
            "dtype": lr["dtype"],
            "bits": lr["bits"],
            "m": lr["m"],
            "rows": lr["rows"],
            "lr_split_count": lr["split_count"],
            "non_lr_split_count": legacy["split_count"],
            "lr_median_ms": lr["median_ms"],
            "non_lr_median_ms": legacy["median_ms"],
            "speedup_vs_non_lr": speedup,
            "lr_max_abs": lr["max_abs"],
            "non_lr_max_abs": legacy["max_abs"],
            "accuracy_within_2e-3": lr["max_abs"] <= 2e-3 and legacy["max_abs"] <= 2e-3,
            "meets_1.5x": speedup >= 1.5,
            "meets_2x": speedup >= 2.0,
            "meets_4x": speedup >= 4.0,
        })

    speedups = [row["speedup_vs_non_lr"] for row in regressions if row["speedup_vs_non_lr"] > 0]
    summary = {
        "cases": len(regressions),
        "geomean_speedup_vs_non_lr": (
            math.exp(statistics.mean(math.log(speedup) for speedup in speedups)) if speedups else None
        ),
        "min_speedup_vs_non_lr": min(speedups) if speedups else None,
        "max_speedup_vs_non_lr": max(speedups) if speedups else None,
        "cases_at_least_1.5x": sum(row["meets_1.5x"] for row in regressions),
        "cases_at_least_2x": sum(row["meets_2x"] for row in regressions),
        "cases_at_least_4x": sum(row["meets_4x"] for row in regressions),
        "all_accuracy_within_2e-3": all(row["accuracy_within_2e-3"] for row in regressions),
        "max_lr_abs": max((row["lr_max_abs"] for row in regressions), default=None),
        "max_non_lr_abs": max((row["non_lr_max_abs"] for row in regressions), default=None),
    }
    return {"regression_summary": summary, "regressions": regressions}


def _print_regression_summary(report: dict) -> None:
    summary = report["regression_summary"]
    if not summary["cases"]:
        return
    print(
        "regression summary: "
        f"cases={summary['cases']} geomean={summary['geomean_speedup_vs_non_lr']:.3f}x "
        f"min={summary['min_speedup_vs_non_lr']:.3f}x "
        f">=1.5x={summary['cases_at_least_1.5x']} "
        f">=2x={summary['cases_at_least_2x']} "
        f">=4x={summary['cases_at_least_4x']} "
        f"accuracy<=2e-3={summary['all_accuracy_within_2e-3']}",
        flush=True,
    )


def _print_table(rows: list[dict]) -> None:
    columns = (
        ("GPU", "physical_gpu"),
        ("Name", "gpu_name"),
        ("Shape", "shape"),
        ("Dtype", "dtype"),
        ("Bits", "bits"),
        ("M", "m"),
        ("Rows", "rows"),
        ("Splits", "split_count"),
        ("Path", "path"),
        ("Median ms", "median_ms"),
        ("P95 ms", "p95_ms"),
        ("Max abs", "max_abs"),
        ("State", "state"),
    )
    rendered = []
    for row in rows:
        rendered.append({
            "physical_gpu": str(row.get("physical_gpu", "?")),
            "gpu_name": str(row.get("gpu_name", "?")),
            "shape": str(row.get("shape", "?")),
            "dtype": str(row.get("dtype", "?")),
            "bits": str(row.get("bits", "?")),
            "m": str(row.get("m", "?")),
            "rows": str(row.get("rows", "?")),
            "split_count": str(row.get("split_count", "?")),
            "path": str(row.get("path", "?")),
            "median_ms": f"{row['median_ms']:.4f}" if "median_ms" in row else "pending",
            "p95_ms": f"{row['p95_ms']:.4f}" if "p95_ms" in row else "pending",
            "max_abs": f"{row['max_abs']:.3g}" if "max_abs" in row else "pending",
            "state": str(row.get("state", "running")),
        })
    widths = {key: max(len(title), *(len(row[key]) for row in rendered)) for title, key in columns}
    border = "+" + "+".join("-" * (widths[key] + 2) for _, key in columns) + "+"
    print(border)
    print("|" + "|".join(f" {title:<{widths[key]}} " for title, key in columns) + "|")
    print(border)
    for row in rendered:
        print("|" + "|".join(f" {row[key]:<{widths[key]}} " for _, key in columns) + "|")
    print(border, flush=True)


def _worker(args: argparse.Namespace) -> None:
    hardware = _idle_preflight(
        args.physical_gpu, args.idle_samples, args.idle_interval, args.idle_memory_tolerance_mib
    )
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch

    from gptqmodel.quantization.qvq import (
        QVQ_V2B2_P32_LR_RING_STEPS,
        QVQ_V2B2_P32_LR_RINGS_PER_TILE,
        local_ring_states_from_edges,
        pack_local_ring_states,
        pack_qvq_binary_bank_ids,
        reconstruct_local_ring_inner_weight,
        reconstruct_qvq_inner_weight,
    )
    from gptqmodel.utils.planar_packing import planar_pack_rows
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"worker requires exactly one visible CUDA device, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    visible_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    torch_uuid = str(getattr(properties, "uuid", "")).removeprefix("GPU-")
    hardware_uuid = hardware["uuid"].removeprefix("GPU-")
    if torch_uuid != hardware_uuid:
        raise RuntimeError(
            f"visible GPU mapping mismatch: physical={args.physical_gpu} uuid={hardware['uuid']} "
            f"torch_uuid={torch_uuid!r} CUDA_VISIBLE_DEVICES={visible_selector!r}"
        )
    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")

    print(
        f"software: physical={args.physical_gpu} pci={hardware['pci.bus_id']} uuid={hardware['uuid']} "
        f"name={properties.name} cc={properties.major}.{properties.minor} sms={properties.multi_processor_count} "
        f"memory={properties.total_memory} torch={torch.__version__} cuda={torch.version.cuda} "
        f"visible={visible_selector} torch_uuid=GPU-{torch_uuid}",
        flush=True,
    )
    all_rows: list[dict] = []
    progress_path = args.out_dir / f"gpu{args.physical_gpu}.progress.json"
    result_path = args.out_dir / f"gpu{args.physical_gpu}.json"
    _write_json(progress_path, {"physical_gpu": args.physical_gpu, "state": "running", "rows": []})

    for shape_name, k, n in DEFAULT_SHAPES:
        for bits in args.bits:
            transition_bits = round(bits * 2)
            tiles = (k // 32) * (n // 8)
            generator = torch.Generator().manual_seed(20260829 + transition_bits * 100 + k + n)
            edges = torch.randint(
                0,
                1 << transition_bits,
                (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
                generator=generator,
                dtype=torch.int64,
            )
            states = local_ring_states_from_edges(edges, bits=bits)
            trellis_cpu = pack_local_ring_states(states, bits=bits)
            selectors = torch.randint(
                0, 2, (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,), generator=generator, dtype=torch.uint8
            )
            bank_ids_cpu = pack_qvq_binary_bank_ids(selectors)
            dense = reconstruct_local_ring_inner_weight(
                trellis_cpu,
                bits=bits,
                in_features=k,
                out_features=n,
                bank_ids=bank_ids_cpu,
                bank_alt_id=torch.tensor([3], dtype=torch.uint8),
            ).cuda()
            trellis = trellis_cpu.cuda()
            bank_ids = bank_ids_cpu.cuda()
            if args.no_non_lr:
                legacy_trellis = legacy_bank_ids = legacy_dense = None
            else:
                legacy_tiles = (k // 16) * (n // 16)
                legacy_edges = torch.randint(
                    0,
                    1 << transition_bits,
                    (128, legacy_tiles),
                    generator=generator,
                    dtype=torch.int32,
                )
                legacy_trellis_cpu = planar_pack_rows(legacy_edges, transition_bits).T.contiguous()
                legacy_selectors = torch.randint(
                    0, 2, (legacy_tiles * 8,), generator=generator, dtype=torch.uint8
                )
                legacy_bank_ids_cpu = pack_qvq_binary_bank_ids(legacy_selectors)
                legacy_dense = reconstruct_qvq_inner_weight(
                    legacy_trellis_cpu,
                    bits=bits,
                    in_features=k,
                    out_features=n,
                    vector_size=2,
                    trellis_window=16,
                    bank_ids=legacy_bank_ids_cpu,
                    v2b2_p32=True,
                    bank_alt_id=torch.tensor([3], dtype=torch.uint8),
                ).cuda()
                legacy_trellis = legacy_trellis_cpu.cuda()
                legacy_bank_ids = legacy_bank_ids_cpu.cuda()
            for dtype_name in args.dtype:
                dtype = getattr(torch, dtype_name)
                for m in args.m:
                    x = torch.randn((m, k), generator=generator, dtype=torch.float32, device="cpu").to(
                        device="cuda", dtype=dtype
                    )
                    reference = x.float() @ dense.float()
                    automatic_split_count = _split_count(
                        m,
                        k,
                        n,
                        properties.multi_processor_count,
                        properties.major,
                    )
                    common = {
                        "physical_gpu": args.physical_gpu,
                        "pci_bus_id": hardware["pci.bus_id"],
                        "uuid": hardware["uuid"],
                        "gpu_name": properties.name,
                        "compute_capability": f"{properties.major}.{properties.minor}",
                        "sm_count": properties.multi_processor_count,
                        "shape": shape_name,
                        "k": k,
                        "n": n,
                        "dtype": dtype_name,
                        "bits": bits,
                        "m": m,
                        "rows": _rows_for_m(m),
                        "state": "complete",
                    }

                    for requested_split_count in args.split_counts:
                        if requested_split_count < 0:
                            raise ValueError("--split-counts values must be non-negative")

                        def native_call(
                            x=x,
                            trellis=trellis,
                            bits=bits,
                            n=n,
                            bank_ids=bank_ids,
                            requested_split_count=requested_split_count,
                        ):
                            return qvq_cuda_gemv(
                                x,
                                trellis,
                                bits,
                                out_features=n,
                                output_fp32=True,
                                bank_ids=bank_ids,
                                v2b2_p32_lr=True,
                                bank_alt_id=3,
                                lr_split_count=requested_split_count,
                            )

                        actual = native_call()
                        torch.cuda.synchronize()
                        metrics = _metrics(actual, reference)
                        if not math.isfinite(metrics["max_abs"]) or metrics["max_abs"] > 2e-3:
                            raise AssertionError(
                                f"LR32 correctness failed for {shape_name} W{bits} {dtype_name} M{m}: {metrics}"
                            )
                        timing = _event_timing(torch, native_call, warmup=args.warmup, iterations=args.iterations)
                        all_rows.append({
                            **common,
                            "split_count": requested_split_count or automatic_split_count,
                            "requested_split_count": requested_split_count,
                            "path": "lr_native_auto" if requested_split_count == 0 else f"lr_native_s{requested_split_count}",
                            **metrics,
                            **timing,
                        })

                    if not args.no_non_lr:
                        legacy_reference = x.float() @ legacy_dense.float()

                        def legacy_call(
                            x=x,
                            legacy_trellis=legacy_trellis,
                            bits=bits,
                            n=n,
                            legacy_bank_ids=legacy_bank_ids,
                        ):
                            return qvq_cuda_gemv(
                                x,
                                legacy_trellis,
                                bits,
                                out_features=n,
                                output_fp32=True,
                                bank_ids=legacy_bank_ids,
                                v2b2_p32=True,
                                bank_alt_id=3,
                            )

                        legacy_actual = legacy_call()
                        torch.cuda.synchronize()
                        legacy_metrics = _metrics(legacy_actual, legacy_reference)
                        if (
                            not math.isfinite(legacy_metrics["max_abs"])
                            or legacy_metrics["max_abs"] > 2e-3
                        ):
                            raise AssertionError(
                                f"legacy V2B2-P32 correctness failed for {shape_name} W{bits} "
                                f"{dtype_name} M{m}: {legacy_metrics}"
                            )
                        legacy_timing = _event_timing(
                            torch,
                            legacy_call,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                        all_rows.append({
                            **common,
                            "split_count": _legacy_split_count(
                                m, k, n, properties.multi_processor_count
                            ),
                            "requested_split_count": 0,
                            "path": "non_lr_native",
                            **legacy_metrics,
                            **legacy_timing,
                        })

                    def dense_call(x=x, dense=dense):
                        return x.float() @ dense.float()

                    dense_timing = _event_timing(
                        torch, dense_call, warmup=args.warmup, iterations=args.iterations
                    )
                    all_rows.append({
                        **common,
                        "split_count": automatic_split_count,
                        "requested_split_count": 0,
                        "path": "cached_dense",
                        "max_abs": 0.0,
                        "mae": 0.0,
                        "mse": 0.0,
                        "rel_l2": 0.0,
                        **dense_timing,
                    })
                    _write_json(progress_path, {
                        "physical_gpu": args.physical_gpu,
                        "hardware": hardware,
                        "state": "running",
                        "completed_rows": len(all_rows),
                        "rows": all_rows,
                    })
            del trellis, bank_ids, dense
            if not args.no_non_lr:
                del legacy_trellis, legacy_bank_ids, legacy_dense
            torch.cuda.empty_cache()

    payload = {
        "label": "qvq_v2b2_p32_lr_cuda_sweep",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "physical_gpu": args.physical_gpu,
        "hardware": hardware,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": all_rows,
    }
    payload.update(_regression_report(all_rows))
    _write_json(result_path, payload)
    _write_json(progress_path, {**payload, "state": "complete", "completed_rows": len(all_rows)})
    _print_table(all_rows)
    _print_regression_summary(payload)


def _all_workers(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hardware = {gpu: _idle_preflight(gpu, args.idle_samples, args.idle_interval, args.idle_memory_tolerance_mib) for gpu in args.gpus}
    processes = []
    script = Path(__file__).resolve()
    for gpu in args.gpus:
        child_args = [
            sys.executable,
            str(script),
            "--worker",
            "--physical-gpu",
            str(gpu),
            "--bits",
            *(str(bits) for bits in args.bits),
            "--dtype",
            *args.dtype,
            "--m",
            *(str(m) for m in args.m),
            "--split-counts",
            *(str(split_count) for split_count in args.split_counts),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--idle-samples",
            str(args.idle_samples),
            "--idle-interval",
            str(args.idle_interval),
            "--idle-memory-tolerance-mib",
            str(args.idle_memory_tolerance_mib),
            "--out-dir",
            str(args.out_dir),
        ]
        if args.no_non_lr:
            child_args.append("--no-non-lr")
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # This torch build accepts numeric selectors reliably; UUID selectors
        # are rejected as an empty device set. PCI_BUS_ID keeps this selector
        # aligned with nvidia-smi's physical index.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["GPTQMODEL_QVQ_CUDA_BUILD_ROOT"] = f"/tmp/qvq-jit-lr-gpu{gpu}"
        env["MAX_JOBS"] = "8"
        env["NINJAFLAGS"] = "-j8"
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = "8"
        env["NVCC_THREADS"] = "2"
        log_path = args.out_dir / f"gpu{gpu}.log"
        log = log_path.open("w", encoding="utf-8")
        processes.append((gpu, subprocess.Popen(child_args, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    print("launched workers:", ", ".join(f"GPU {gpu} ({hardware[gpu]['uuid']})" for gpu in args.gpus), flush=True)
    while any(process.poll() is None for _, process, _ in processes):
        live_rows = []
        for gpu, _, _ in processes:
            progress_path = args.out_dir / f"gpu{gpu}.progress.json"
            if progress_path.exists():
                try:
                    live_rows.extend(json.loads(progress_path.read_text(encoding="utf-8")).get("rows", []))
                except json.JSONDecodeError:
                    pass
        print(f"live update {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: rows={len(live_rows)}", flush=True)
        if live_rows:
            _print_table(live_rows)
        time.sleep(60)
    failures = []
    for gpu, process, log in processes:
        log.close()
        if process.returncode != 0:
            failures.append((gpu, process.returncode))
    if failures:
        raise RuntimeError(f"GPU workers failed: {failures}")
    final_rows = []
    for gpu in args.gpus:
        result_path = args.out_dir / f"gpu{gpu}.json"
        final_rows.extend(json.loads(result_path.read_text(encoding="utf-8")).get("rows", []))
    _print_table(final_rows)
    report = _regression_report(final_rows)
    _write_json(args.out_dir / "summary.json", {
        "label": "qvq_v2b2_p32_lr_cuda_regression",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "physical_gpus": args.gpus,
        **report,
    })
    _print_regression_summary(report)


def main() -> None:
    args = _parse_args()
    if args.all:
        _all_workers(args)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        os.environ.setdefault("GPTQMODEL_QVQ_CUDA_BUILD_ROOT", f"/tmp/qvq-jit-lr-gpu{args.physical_gpu}")
        _worker(args)


if __name__ == "__main__":
    main()
