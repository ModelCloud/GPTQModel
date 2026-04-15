#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any


# Default to the first PCI-order A100-class GPU on this host unless the caller
# already pinned a different device explicitly.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7")

import torch

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.backend import BACKEND


RESULT_PREFIX = "RESULT_JSON\t"
DEFAULT_W4A16_PATH = "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-bit4-g128-symFasle-descFalse"
DEFAULT_W4A8_PATH = "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-W4A8-FP8-Dynamic"
DEFAULT_PROMPT = (
    "What is the mass of the Sun? If you don't know the exact value, at least describe how one would create "
    "that equation and Sun's mass composition. "
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AWQ W4A16 Marlin vs activation-quantized AWQ W4A8 unfused/QDQ on A100-class GPUs."
    )
    parser.add_argument("--w4a16-path", default=DEFAULT_W4A16_PATH, help="Path to the W4A16 AWQ checkpoint.")
    parser.add_argument("--w4a8-path", default=DEFAULT_W4A8_PATH, help="Path to the activation-quantized W4A8 AWQ checkpoint.")
    parser.add_argument("--device", default="cuda:0", help="Torch device inside the current CUDA_VISIBLE_DEVICES mask.")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="Load dtype.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for both prefill and generate benchmarks.")
    parser.add_argument(
        "--prefill-lengths",
        default="128,512",
        help="Comma-separated prefill sequence lengths. Example: 128,512",
    )
    parser.add_argument("--new-tokens", type=int, default=64, help="Number of new tokens for generate().")
    parser.add_argument("--warmup-prefill", type=int, default=4, help="Warmup iterations for prefill.")
    parser.add_argument("--runs-prefill", type=int, default=15, help="Measured iterations for prefill.")
    parser.add_argument("--warmup-generate", type=int, default=1, help="Warmup iterations for generate().")
    parser.add_argument("--runs-generate", type=int, default=5, help="Measured iterations for generate().")
    parser.add_argument("--skip-generate", action="store_true", help="Only benchmark prefill forward passes.")
    parser.add_argument("--quick", action="store_true", help="Reduce iterations for a quicker smoke benchmark.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write the result JSON.")
    return parser.parse_args()


def _dtype_from_arg(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _build_batch(tokenizer, prompt: str, seq_len: int, batch_size: int, device: str) -> dict[str, torch.Tensor]:
    tokens = tokenizer(prompt, add_special_tokens=False).input_ids
    repeated = (tokens * ((seq_len // len(tokens)) + 2))[:seq_len]
    input_ids = torch.tensor([repeated for _ in range(batch_size)], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _bench_prefill(
    model: GPTQModel,
    batch: dict[str, torch.Tensor],
    *,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    times: list[float] = []
    peaks: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model.model(**batch, use_cache=False)
        _sync()
        for _ in range(runs):
            torch.cuda.reset_peak_memory_stats()
            _sync()
            start = time.perf_counter()
            _ = model.model(**batch, use_cache=False)
            _sync()
            times.append((time.perf_counter() - start) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated() / (1024**2))

    latency_ms = mean(times)
    return {
        "latency_ms": latency_ms,
        "tokens_per_s": (batch["input_ids"].numel() * 1000.0) / latency_ms,
        "peak_alloc_mib": mean(peaks),
    }


def _bench_generate(
    model: GPTQModel,
    batch: dict[str, torch.Tensor],
    *,
    pad_token_id: int,
    new_tokens: int,
    warmup: int,
    runs: int,
) -> dict[str, float | int]:
    times: list[float] = []
    peaks: list[float] = []
    outputs = None
    gen_kwargs = dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,
        eos_token_id=None,
    )
    with torch.inference_mode():
        for _ in range(warmup):
            outputs = model.generate(**batch, **gen_kwargs)
        _sync()
        for _ in range(runs):
            torch.cuda.reset_peak_memory_stats()
            _sync()
            start = time.perf_counter()
            outputs = model.generate(**batch, **gen_kwargs)
            _sync()
            times.append((time.perf_counter() - start) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated() / (1024**2))

    assert outputs is not None
    generated = int(outputs.shape[1] - batch["input_ids"].shape[1])
    latency_ms = mean(times)
    return {
        "latency_ms": latency_ms,
        "new_tokens_per_s": (generated * batch["input_ids"].shape[0] * 1000.0) / latency_ms,
        "peak_alloc_mib": mean(peaks),
        "generated_tokens": generated,
    }


def _load_model(path: str, backend: BACKEND, device: str, dtype: torch.dtype) -> GPTQModel:
    model = GPTQModel.load(
        path,
        device=device,
        backend=backend,
        dtype=dtype,
    )
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model


def _kernel_names(model: GPTQModel) -> list[str]:
    return sorted({type(module).__name__ for _, module in model.model.named_modules() if isinstance(module, BaseQuantLinear)})


def _summarize(results: dict[str, Any], keys: list[str]) -> dict[str, dict[str, float]]:
    base = results["w4a16_marlin"]
    comp = results["w4a8_qdq_torch"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        base_rate = base[key].get("tokens_per_s", base[key].get("new_tokens_per_s"))
        comp_rate = comp[key].get("tokens_per_s", comp[key].get("new_tokens_per_s"))
        summary[key] = {
            "w4a16_ms": round(base[key]["latency_ms"], 4),
            "w4a8_ms": round(comp[key]["latency_ms"], 4),
            "slowdown_x": round(comp[key]["latency_ms"] / base[key]["latency_ms"], 4),
            "w4a16_rate": round(base_rate, 2),
            "w4a8_rate": round(comp_rate, 2),
            "throughput_loss_pct": round((1.0 - (comp_rate / base_rate)) * 100.0, 2),
            "w4a16_peak_mib": round(base[key]["peak_alloc_mib"], 2),
            "w4a8_peak_mib": round(comp[key]["peak_alloc_mib"], 2),
        }
    return summary


def _print_summary_table(summary: dict[str, dict[str, float]]) -> None:
    headers = ["Case", "W4A16 ms", "W4A8 ms", "Slowdown x", "W4A16 rate", "W4A8 rate", "Loss %", "W4A16 MiB", "W4A8 MiB"]
    rows = []
    for case, data in summary.items():
        rows.append(
            [
                case,
                f"{data['w4a16_ms']:.4f}",
                f"{data['w4a8_ms']:.4f}",
                f"{data['slowdown_x']:.4f}",
                f"{data['w4a16_rate']:.2f}",
                f"{data['w4a8_rate']:.2f}",
                f"{data['throughput_loss_pct']:.2f}",
                f"{data['w4a16_peak_mib']:.2f}",
                f"{data['w4a8_peak_mib']:.2f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(cols: list[str]) -> str:
        return "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"

    print(border())
    print(render(headers))
    print(border())
    for row in rows:
        print(render(row))
    print(border())


def main() -> int:
    args = _parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prefill_lengths = [int(part.strip()) for part in args.prefill_lengths.split(",") if part.strip()]
    if not prefill_lengths:
        raise ValueError("At least one prefill length is required.")

    if args.quick:
        args.warmup_prefill = min(args.warmup_prefill, 1)
        args.runs_prefill = min(args.runs_prefill, 3)
        args.warmup_generate = min(args.warmup_generate, 1)
        args.runs_generate = min(args.runs_generate, 2)

    dtype = _dtype_from_arg(args.dtype)
    cases = {
        "w4a16_marlin": (args.w4a16_path, BACKEND.AWQ_MARLIN),
        "w4a8_qdq_torch": (args.w4a8_path, BACKEND.AWQ_TORCH),
    }

    print(
        f"Benchmarking on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"with CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')} and device={args.device}"
    )

    results: dict[str, Any] = {}
    for name, (path, backend) in cases.items():
        _clear_mem()
        model = _load_model(path=path, backend=backend, device=args.device, dtype=dtype)
        tokenizer = model.tokenizer

        case_result: dict[str, Any] = {
            "checkpoint": path,
            "backend": str(backend),
            "kernels": _kernel_names(model),
        }
        for seq_len in prefill_lengths:
            batch = _build_batch(tokenizer, DEFAULT_PROMPT, seq_len, args.batch_size, args.device)
            case_result[f"prefill_{seq_len}"] = _bench_prefill(
                model,
                batch,
                warmup=args.warmup_prefill,
                runs=args.runs_prefill,
            )

        if not args.skip_generate:
            batch = _build_batch(tokenizer, DEFAULT_PROMPT, prefill_lengths[0], args.batch_size, args.device)
            case_result[f"generate_{prefill_lengths[0]}_plus_{args.new_tokens}"] = _bench_generate(
                model,
                batch,
                pad_token_id=tokenizer.pad_token_id,
                new_tokens=args.new_tokens,
                warmup=args.warmup_generate,
                runs=args.runs_generate,
            )

        results[name] = case_result
        del model
        _clear_mem()

    summary_keys = [f"prefill_{seq_len}" for seq_len in prefill_lengths]
    if not args.skip_generate:
        summary_keys.append(f"generate_{prefill_lengths[0]}_plus_{args.new_tokens}")
    summary = _summarize(results, summary_keys)

    payload = {
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "device": args.device,
            "dtype": args.dtype,
        },
        "results": results,
        "summary": summary,
    }

    _print_summary_table(summary)
    print(RESULT_PREFIX + json.dumps(payload, indent=None, sort_keys=True))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote JSON to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
