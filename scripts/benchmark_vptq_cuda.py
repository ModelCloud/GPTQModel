#!/usr/bin/env python3
"""Paired VPTQ_CUDA vs VPTQ_TORCH inference benchmark.

The two forwards use the same VPTQ module state and input tensor. A checkpoint
can be measured directly, or --synthetic can create deterministic packed VPTQ
layers without a model download or quantization run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.vptq import pack_vptq_indices


def _measure_pair(
    cuda_forward: Callable[[], torch.Tensor],
    torch_forward: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        torch_forward()
        cuda_forward()
    torch.cuda.synchronize()

    samples = {"vptq_cuda": [], "vptq_torch": []}
    functions = {"vptq_cuda": cuda_forward, "vptq_torch": torch_forward}
    for iteration in range(repeats):
        # Alternate order to reduce systematic clock/cache bias between paths.
        order = ("vptq_cuda", "vptq_torch") if iteration % 2 == 0 else ("vptq_torch", "vptq_cuda")
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))

    def summarize(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "median_ms": statistics.median(ordered),
            "min_ms": ordered[0],
            "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        }

    return summarize(samples["vptq_cuda"]), summarize(samples["vptq_torch"])


def _synthetic_module(
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    seed: int,
) -> VPTQCudaLinear:
    if in_features % 128:
        raise ValueError(f"Synthetic in_features must be divisible by 128, got {in_features}.")
    module = VPTQCudaLinear(
        bits=4,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
        vector_lens=(-1, 4),
        num_centroids=(-1, 16),
        num_res_centroids=(-1, 16),
        group_size=128,
    ).to(device).eval()

    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        module.centroids.weight.copy_(
            torch.randn(module.centroids.weight.shape, generator=generator, device=device, dtype=dtype)
            / in_features**0.5
        )
        module.res_centroids.weight.copy_(
            (0.1 / in_features**0.5) * torch.randn(
                module.res_centroids.weight.shape, generator=generator, device=device, dtype=dtype
            )
        )
        indices = torch.randint(
            module.num_centroids,
            (module.group_num, module.num_indices, module.group_size),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        residual_indices = torch.randint(
            module.num_res_centroids,
            indices.shape,
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        module.indices.copy_(
            pack_vptq_indices(
                indices,
                module.index_bits,
                res_indices=residual_indices,
                res_bits=module.res_index_bits,
            )
        )
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator, device=device, dtype=dtype))
    return module


def _benchmark_module(
    name: str,
    module: VPTQCudaLinear,
    rows_list: list[int],
    *,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    results = []
    for rows in rows_list:
        generator = torch.Generator(device=device).manual_seed(seed + rows)
        x = torch.randn(
            rows, module.in_features, generator=generator, device=device, dtype=module.centroids.weight.dtype
        )
        cuda_forward = lambda: module(x)
        torch_forward = lambda: VPTQLinear.forward(module, x)

        expected = torch_forward()
        actual = cuda_forward()
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        cuda_stats, torch_stats = _measure_pair(
            cuda_forward, torch_forward, warmup=warmup, repeats=repeats
        )
        results.append(
            {
                "module": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "rows": rows,
                "dtype": dtype_name,
                "vptq_cuda_median_ms": cuda_stats["median_ms"],
                "vptq_torch_median_ms": torch_stats["median_ms"],
                "speedup_cuda_over_torch": torch_stats["median_ms"] / cuda_stats["median_ms"],
                "cuda_min_ms": cuda_stats["min_ms"],
                "cuda_p90_ms": cuda_stats["p90_ms"],
                "torch_min_ms": torch_stats["min_ms"],
                "torch_p90_ms": torch_stats["p90_ms"],
                "max_abs_error": float(error.max().item()),
                "mean_abs_error": float(error.mean().item()),
            }
        )
    return results


def _load_checkpoint_modules(model_path: str, dtype: torch.dtype, device: torch.device, layer: str | None):
    from gptqmodel import GPTQModel

    model = GPTQModel.load(
        model_path,
        backend=BACKEND.VPTQ_CUDA,
        dtype=dtype,
        device_map={"": str(device)},
    ).eval()
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VPTQCudaLinear) and (layer is None or name == layer)
    ]
    if not modules:
        raise RuntimeError(f"No VPTQ CUDA layers found for layer={layer!r}.")
    return model, modules


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers.") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Local VPTQ checkpoint directory")
    parser.add_argument("--synthetic", action="store_true", help="Benchmark generated VPTQ layers instead of a checkpoint.")
    parser.add_argument("--layer", help="Exact VPTQ module name for checkpoint mode; default benchmarks all VPTQ layers")
    parser.add_argument("--shapes", default="2048x2048,4096x4096", help="Synthetic INxOUT shapes, e.g. 2048x2048")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-sizes", default="1,8,16,32,128", help="Activation rows, comma-separated")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()

    if args.synthetic == bool(args.model):
        parser.error("Provide a checkpoint path, or pass --synthetic (but not both).")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats must be positive.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device {device} is not available.")

    dtype = getattr(torch, args.dtype)
    rows_list = _parse_ints(args.batch_sizes, "--batch-sizes")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    results = []
    models = []

    if args.synthetic:
        try:
            shapes = [tuple(int(part) for part in item.lower().split("x")) for item in args.shapes.split(",")]
            if any(len(shape) != 2 or any(dim <= 0 for dim in shape) for shape in shapes):
                raise ValueError
        except ValueError as exc:
            parser.error("--shapes must be comma-separated INxOUT pairs with positive dimensions.")
        for index, (in_features, out_features) in enumerate(shapes):
            module = _synthetic_module(
                in_features, out_features, dtype, device, seed=args.seed + index
            )
            results.extend(
                _benchmark_module(
                    f"synthetic_{in_features}x{out_features}", module, rows_list,
                    dtype_name=args.dtype, device=device, warmup=args.warmup,
                    repeats=args.repeats, seed=args.seed,
                )
            )
            del module
            torch.cuda.empty_cache()
    else:
        model, modules = _load_checkpoint_modules(args.model, dtype, device, args.layer)
        models.append(model)
        for name, module in modules:
            if module.centroids.weight.dtype != dtype:
                raise RuntimeError(
                    f"{name}: codebook dtype {module.centroids.weight.dtype} does not match requested {dtype}."
                )
            results.extend(
                _benchmark_module(
                    name, module, rows_list, dtype_name=args.dtype, device=device,
                    warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                )
            )

    fields = list(results[0])
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "mode": "synthetic" if args.synthetic else "checkpoint",
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "results": results,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    del models
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
