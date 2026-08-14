# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark QVQ CUDA direct decode against transient and cached dense references."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time

DEEPSEEK_V4_FLASH_0731_SHAPES = {
    "deepseek-v4-flash-0731-attention": (
        ("q_a_proj", 4096, 1024),
        ("q_b_proj", 1024, 32768),
        ("kv_proj", 4096, 512),
        ("o_a_proj", 4096, 8192),
        ("o_b_proj", 8192, 4096),
    ),
    "deepseek-v4-flash-0731-moe": (
        ("expert_gate_up", 4096, 2048),
        ("expert_down", 2048, 4096),
        ("moe_router", 4096, 256),
    ),
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--bits", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
    parser.add_argument("--vector-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument(
        "--shape-set",
        choices=("custom", *DEEPSEEK_V4_FLASH_0731_SHAPES, "deepseek-v4-flash-0731-all"),
        default="custom",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output-fp32", action="store_true", help="Validate/timing the FP32 accumulator output path.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-iterations", type=int, default=5)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=8)
    parser.add_argument("--max-mse", type=float, default=1e-3)
    parser.add_argument("--max-abs-error", type=float, default=2e-3)
    parser.add_argument("--max-kld", type=float, default=2e-4)
    parser.add_argument("--min-top1", type=float, default=0.96875)
    parser.add_argument("--label", default="run", help="Run label stored in the JSON result.")
    parser.add_argument("--results", type=str, help="Write machine-readable timing and accuracy rows to this JSON file.")
    args = parser.parse_args()
    if args.vector_size == 4 and any(bits > 4 for bits in args.bits):
        parser.error("QVQ V4 supports W1 through W4; use --vector-size 2 for W4.5 through W8")
    return args


def _idle_preflight(physical_gpu: int, samples: int, interval: float, memory_tolerance_mib: int) -> dict[str, str]:
    query = "index,pci.bus_id,uuid,name,memory.used,utilization.gpu"
    accepted = None
    for sample in range(samples):
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={physical_gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        values = [value.strip() for value in output.split(",")]
        if len(values) != 6:
            raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
        accepted = dict(zip(query.split(","), values, strict=True))
        if int(accepted["memory.used"]) > memory_tolerance_mib or int(accepted["utilization.gpu"]) != 0:
            raise RuntimeError(
                f"physical GPU {physical_gpu} is not idle: memory={accepted['memory.used']} MiB "
                f"(tolerance={memory_tolerance_mib} MiB), "
                f"utilization={accepted['utilization.gpu']}%"
            )
        if sample + 1 < samples:
            time.sleep(interval)
    assert accepted is not None
    print(
        "idle gate: "
        f"physical={accepted['index']} pci={accepted['pci.bus_id']} uuid={accepted['uuid']} "
        f"name={accepted['name']} memory={accepted['memory.used']}MiB utilization={accepted['utilization.gpu']}% "
        f"samples={samples}"
    )
    return accepted


def _timings(torch, fn, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for iteration in range(iterations):
        starts[iteration].record()
        fn()
        ends[iteration].record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    values.sort()
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
    }


def _metrics(torch, actual, reference) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    actual_top5 = actual_f.topk(5, dim=-1).indices
    reference_top5 = reference_f.topk(5, dim=-1).indices
    error_energy = delta.square().sum().clamp_min(1e-12)
    reference_energy = reference_f.square().sum().clamp_min(1e-12)
    return {
        "mae": delta.abs().mean().item(),
        "mse": delta.square().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "rel_l2": (error_energy / reference_energy).sqrt().item(),
        "sqnr_db": (10 * torch.log10(reference_energy / error_energy)).item(),
        "cosine": torch.nn.functional.cosine_similarity(actual_f.flatten(), reference_f.flatten(), dim=0).item(),
        "kld": torch.nn.functional.kl_div(
            actual_f.log_softmax(dim=-1), reference_f.softmax(dim=-1), reduction="batchmean"
        ).item(),
        "top1": (actual_f.argmax(dim=-1) == reference_f.argmax(dim=-1)).float().mean().item(),
        "top5": (
            (actual_top5.unsqueeze(-1) == reference_top5.unsqueeze(-2)).any(dim=-1).float().mean().item()
        ),
    }


def _print_table(rows):
    columns = (
        ("Shape", "shape"),
        ("K", "k"),
        ("N", "n"),
        ("Bits", "bits"),
        ("M", "m"),
        ("Path", "path"),
        ("Median ms", "median_ms"),
        ("P95 ms", "p95_ms"),
        ("Speedup", "speedup"),
        ("MAE", "mae"),
        ("MSE", "mse"),
        ("RMSE", "rmse"),
        ("Max abs", "max_abs"),
        ("Rel-L2", "rel_l2"),
        ("SQNR dB", "sqnr_db"),
        ("Cosine", "cosine"),
        ("Fwd KLD", "kld"),
        ("Top-1", "top1"),
        ("Top-5", "top5"),
    )
    rendered = []
    for row in rows:
        rendered.append(
            {
                "shape": row["shape"],
                "k": str(row["k"]),
                "n": str(row["n"]),
                "bits": str(row["bits"]),
                "m": str(row["m"]),
                "path": row["path"],
                "median_ms": f"{row['median_ms']:.4f}",
                "p95_ms": f"{row['p95_ms']:.4f}",
                "speedup": f"{row['speedup']:.2f}x",
                "mae": f"{row['mae']:.6g}",
                "mse": f"{row['mse']:.6g}",
                "rmse": f"{row['rmse']:.6g}",
                "max_abs": f"{row['max_abs']:.6g}",
                "rel_l2": f"{row['rel_l2']:.6g}",
                "sqnr_db": f"{row['sqnr_db']:.3f}",
                "cosine": f"{row['cosine']:.6f}",
                "kld": f"{row['kld']:.6g}",
                "top1": f"{row['top1']:.4f}",
                "top5": f"{row['top5']:.4f}",
            }
        )
    widths = {key: max(len(title), *(len(row[key]) for row in rendered)) for title, key in columns}
    border = "+" + "+".join("-" * (widths[key] + 2) for _, key in columns) + "+"
    print(border)
    print("|" + "|".join(f" {title:<{widths[key]}} " for title, key in columns) + "|")
    print(border)
    for row in rendered:
        print("|" + "|".join(f" {row[key]:<{widths[key]}} " for _, key in columns) + "|")
    print(border)


def main():
    args = _parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one requested physical GPU before launch")
    hardware = _idle_preflight(
        args.physical_gpu, args.idle_samples, args.idle_interval, args.idle_memory_tolerance_mib
    )

    import torch

    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_cuda import prewarm_qvq_cuda, qvq_cuda_gemv

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"benchmark requires exactly one visible CUDA device, got {torch.cuda.device_count()}")
    dtype = getattr(torch, args.dtype)
    properties = torch.cuda.get_device_properties(0)
    print(
        f"software: python={os.sys.version.split()[0]} gil={os.sys._is_gil_enabled()} torch={torch.__version__} "
        f"cuda={torch.version.cuda} device={properties.name} cc={properties.major}.{properties.minor} "
        f"sms={properties.multi_processor_count} memory={properties.total_memory} physical_uuid={hardware['uuid']} "
        f"dtype={dtype} shape_set={args.shape_set}"
    )
    if not prewarm_qvq_cuda():
        raise RuntimeError("QVQ CUDA extension failed to load")

    if args.shape_set == "custom":
        shapes = (("custom", args.k, args.n),)
    elif args.shape_set == "deepseek-v4-flash-0731-all":
        shapes = tuple(shape for shape_set in DEEPSEEK_V4_FLASH_0731_SHAPES.values() for shape in shape_set)
    else:
        shapes = DEEPSEEK_V4_FLASH_0731_SHAPES[args.shape_set]

    rows = []
    for shape, size_k, size_n in shapes:
        for bits in args.bits:
            rate_key = int(round(bits * 2))
            generator = torch.Generator(device="cpu").manual_seed(12000 + rate_key + size_k + size_n)
            tile_count = (size_k // 16) * (size_n // 16)
            words_per_tile = qvq_words_per_tile(bits, vector_size=args.vector_size)
            trellis = torch.randint(
                -(2**31),
                2**31 - 1,
                (tile_count, words_per_tile),
                generator=generator,
                dtype=torch.int32,
                device="cpu",
            ).cuda()
            dense = reconstruct_qvq_inner_weight(
                trellis,
                bits=bits,
                vector_size=args.vector_size,
                in_features=size_k,
                out_features=size_n,
            ).to(dtype)
            for m in args.m:
                x = torch.randn((m, size_k), generator=generator, dtype=torch.float32).to(device="cuda", dtype=dtype)
                reference = x.float() @ dense.float()
                if not args.output_fp32:
                    reference = reference.to(dtype)
                actual = qvq_cuda_gemv(
                    x, trellis, bits, out_features=size_n, vector_size=args.vector_size, output_fp32=args.output_fp32
                )
                metrics = _metrics(torch, actual, reference)
                torch.testing.assert_close(actual, reference, rtol=2e-2, atol=args.max_abs_error)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise AssertionError(f"{shape} W{bits} M{m} produced non-finite accuracy metrics: {metrics}")
                if metrics["mse"] > args.max_mse:
                    raise AssertionError(f"{shape} W{bits} M{m} MSE {metrics['mse']} exceeds {args.max_mse}")
                if metrics["max_abs"] > args.max_abs_error:
                    raise AssertionError(
                        f"{shape} W{bits} M{m} max abs error {metrics['max_abs']} exceeds {args.max_abs_error}"
                    )
                if metrics["kld"] > args.max_kld:
                    raise AssertionError(f"{shape} W{bits} M{m} KLD {metrics['kld']} exceeds {args.max_kld}")
                if metrics["top1"] < args.min_top1:
                    raise AssertionError(f"{shape} W{bits} M{m} top-1 {metrics['top1']} is below {args.min_top1}")

                direct = _timings(
                    torch,
                    lambda x=x, trellis=trellis, bits=bits, size_n=size_n: qvq_cuda_gemv(
                        x,
                        trellis,
                        bits,
                        out_features=size_n,
                        vector_size=args.vector_size,
                        output_fp32=args.output_fp32,
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                cached = _timings(
                    torch,
                    lambda x=x, dense=dense: x @ dense,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                transient = _timings(
                    torch,
                    lambda x=x, trellis=trellis, bits=bits, size_k=size_k, size_n=size_n: x
                    @ reconstruct_qvq_inner_weight(
                        trellis,
                        bits=bits,
                        vector_size=args.vector_size,
                        in_features=size_k,
                        out_features=size_n,
                    ).to(dtype),
                    warmup=1,
                    iterations=args.reference_iterations,
                )
                for path, timing in (
                    ("transient_dense_ref", transient),
                    (f"qvq_cuda_v{args.vector_size}", direct),
                    ("cached_dense_ceiling", cached),
                ):
                    rows.append(
                        {
                            "shape": shape,
                            "k": size_k,
                            "n": size_n,
                            "bits": bits,
                            "m": m,
                            "path": path,
                            **timing,
                            "speedup": transient["median_ms"] / timing["median_ms"],
                            **(metrics if path.startswith("qvq_cuda") else {key: 0.0 for key in metrics}),
                        }
                    )
            del dense, trellis
            torch.cuda.empty_cache()
    _print_table(rows)
    if args.results:
        payload = {
            "label": args.label,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "physical_gpu": hardware,
            "vector_size": args.vector_size,
            "output_fp32": args.output_fp32,
            "dtype": str(dtype),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rows": rows,
        }
        with open(args.results, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
