# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Time full-model MACaQ W4A4 and plain GPTQ W4A16 quantization.

Example:
    python scripts/benchmark_macaq_quantization.py \
        --model /path/to/Llama-3.2-1B-Instruct \
        --calibration /path/to/llm.parquet --order gptq,macaq

This is a workflow timing, including calibration, replay, quantization, and
packing. The methods have different activation precision and are not a
quality-matched comparison. A single round is exploratory; alternate order
over multiple rounds before making a narrow performance claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.quantization import MacaqConfig, QuantizeConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True, help="Parquet calibration dataset")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--concat-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--flip-iters", type=int, default=128)
    parser.add_argument("--pre-alpha", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", default="eager")
    parser.add_argument("--order", choices=("macaq,gptq", "gptq,macaq"), default="macaq,gptq")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.rows < 1 or args.rounds < 1:
        parser.error("--rows and --rounds must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA device is required for both explicit Triton backends")
    return args


def _config(method: str, args: argparse.Namespace):
    if method == "macaq":
        return (
            MacaqConfig(
                n_flip_iters=args.flip_iters,
                pre_alpha=args.pre_alpha,
                offload_to_disk=False,
            ),
            BACKEND.MACAQ_TRITON,
        )
    return (
        QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            offload_to_disk=False,
        ),
        BACKEND.GPTQ_TRITON,
    )


def _run(method: str, calibration, args: argparse.Namespace, round_index: int) -> dict:
    config, backend = _config(method, args)
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    started = time.perf_counter()
    model = GPTQModel.load(
        args.model,
        quantize_config=config,
        dtype=torch.bfloat16,
        device=args.device,
        attn_implementation=args.attention,
    )
    load_seconds = time.perf_counter() - started

    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    model.quantize(
        calibration,
        calibration_concat_size=args.concat_size,
        batch_size=args.batch_size,
        backend=backend,
    )
    torch.cuda.synchronize(args.device)
    quantize_seconds = time.perf_counter() - started

    records = [record for record in model.quant_log if isinstance(record, dict) and record.get("process") == method]
    result = {
        "round": round_index,
        "method": method,
        "backend": backend.value,
        "gpu": torch.cuda.get_device_name(args.device),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "torch": torch.__version__,
        "rows": args.rows,
        "concat_size": args.concat_size,
        "flip_iters": args.flip_iters if method == "macaq" else None,
        "load_seconds": load_seconds,
        "quantize_seconds": quantize_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(args.device) / 2**30,
        "quantized_modules": len(records),
        "logged_module_seconds": sum(float(record.get("time") or 0) for record in records),
        "packed_macaq_modules": sum(isinstance(m, MacaqLinear) for m in model.model.modules()),
    }
    print("BENCH_RESULT", json.dumps(result, sort_keys=True), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _parse_args()
    calibration = load_dataset("parquet", data_files=args.calibration, split="train")
    if args.rows > len(calibration):
        raise ValueError(f"Requested {args.rows} calibration rows; dataset has {len(calibration)}.")
    calibration = calibration.select(range(args.rows))
    order = args.order.split(",")
    results = []
    for round_index in range(args.rounds):
        for method in order if round_index % 2 == 0 else order[::-1]:
            results.append(_run(method, calibration, args, round_index + 1))

    summary = {}
    for method in ("macaq", "gptq"):
        times = [record["quantize_seconds"] for record in results if record["method"] == method]
        summary[method] = {
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
        }
    summary["macaq_over_gptq"] = summary["macaq"]["median_seconds"] / summary["gptq"]["median_seconds"]
    summary["rounds"] = args.rounds
    summary["exploratory"] = args.rounds < 4
    print("BENCH_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
