#!/usr/bin/env python3
"""Compare the fused VPTQ CUDA path with the Torch reference on a checkpoint."""

from __future__ import annotations

import argparse
import statistics

import torch

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.vptq import VPTQLinear, _VPTQ_CUDA_MAX_ROWS
from gptqmodel.utils.backend import BACKEND


def _milliseconds(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local VPTQ checkpoint directory")
    parser.add_argument("--layer", help="Exact VPTQ module name; default benchmarks all layers")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(batch <= 0 for batch in batch_sizes):
        raise SystemExit("Batch sizes must be positive integers.")
    if any(batch > _VPTQ_CUDA_MAX_ROWS for batch in batch_sizes):
        raise SystemExit(
            f"Fused CUDA benchmark supports at most {_VPTQ_CUDA_MAX_ROWS} rows; "
            "larger batches use the Torch fallback."
        )

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required.")
    dtype = getattr(torch, args.dtype)
    model = GPTQModel.load(
        args.model,
        backend=BACKEND.VPTQ_CUDA,
        dtype=dtype,
        device_map={"": "cuda:0"},
    ).eval()
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VPTQLinear) and (args.layer is None or name == args.layer)
    ]
    if not layers:
        raise SystemExit(f"No VPTQLinear modules found for layer={args.layer!r}.")

    print("layer,batch,dtype,torch_ms,fused_cuda_ms,speedup,max_abs_error")
    for name, module in layers:
        if module.centroids.weight.dtype != dtype:
            raise SystemExit(
                f"{name}: checkpoint codebook dtype {module.centroids.weight.dtype} does not match {dtype}."
            )
        x_by_batch = {
            batch: torch.randn(batch, module.in_features, device="cuda", dtype=dtype)
            for batch in batch_sizes
        }
        for batch, x in x_by_batch.items():
            module.backend = BACKEND.VPTQ_TORCH
            expected = module(x)
            torch_ms = _milliseconds(lambda: module(x), warmup=args.warmup, repeats=args.repeats)

            module.backend = BACKEND.VPTQ_CUDA
            actual = module(x)
            torch.cuda.synchronize()
            max_abs_error = float((actual.float() - expected.float()).abs().max().item())
            cuda_ms = _milliseconds(lambda: module(x), warmup=args.warmup, repeats=args.repeats)
            print(
                f"{name},{batch},{args.dtype},{torch_ms:.4f},{cuda_ms:.4f},"
                f"{torch_ms / cuda_ms:.2f},{max_abs_error:.6g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
