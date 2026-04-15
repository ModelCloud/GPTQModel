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


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)

import torch
from datasets import load_dataset

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.torch import torch_empty_cache


RESULT_PREFIX = "RESULT_JSON\t"
DEFAULT_MODEL_PATH = "/monster/data/model/Llama-3.2-1B-Instruct"
DEFAULT_DATASET_PATH = "/monster/data/model/dataset/c4-train.00000-of-01024.json.gz"
DEFAULT_SAVE_PATH = "/tmp/llama3_2_1b_instruct_qqq_w4a4"
DEFAULT_PROMPT = "The capital city of France is named"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and smoke-test the QQQ W4A4 prototype on Llama 3.2 1B Instruct.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--calibration-concat-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--bench-batch-size", type=int, default=4)
    parser.add_argument("--bench-seq-len", type=int, default=128)
    parser.add_argument("--bench-warmup", type=int, default=1)
    parser.add_argument("--bench-runs", type=int, default=3)
    parser.add_argument("--force-requant", action="store_true")
    return parser.parse_args()


def _clear_mem() -> None:
    gc.collect()
    torch_empty_cache()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_calibration_dataset(path: str, rows: int):
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset.select(range(rows))


def _ensure_quantized_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    save_path = Path(args.save_path)
    qcfg_path = save_path / "quantize_config.json"
    if save_path.exists() and qcfg_path.exists() and not args.force_requant:
        return save_path, {"requantized": False}

    save_path.mkdir(parents=True, exist_ok=True)
    calibration = _load_calibration_dataset(args.dataset_path, args.calibration_rows)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        method=METHOD.QQQ,
        format=FORMAT.QQQ,
        activation_bits=4,
    )

    start = time.perf_counter()
    model = GPTQModel.load(args.model_path, quantize_config=qcfg)
    model.quantize(
        calibration,
        batch_size=args.batch_size,
        calibration_concat_size=args.calibration_concat_size,
    )
    model.save(str(save_path))
    quantize_s = time.perf_counter() - start

    del model
    _clear_mem()
    return save_path, {"requantized": True, "quantize_seconds": round(quantize_s, 4)}


def _collect_qmodules(model: GPTQModel) -> list[QQQLinear]:
    modules = [module for _, module in model.model.named_modules() if isinstance(module, QQQLinear)]
    if not modules:
        raise RuntimeError("Expected at least one QQQLinear module after loading the quantized checkpoint.")
    return modules


def _set_activation_bits(modules: list[QQQLinear], activation_bits: int) -> None:
    for module in modules:
        module.activation_bits = activation_bits


def _build_batch(model: GPTQModel, prompt: str, seq_len: int, batch_size: int, device: str) -> dict[str, torch.Tensor]:
    tokens = model.tokenizer(prompt, add_special_tokens=False).input_ids
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
    latencies_ms: list[float] = []
    peaks_mib: list[float] = []
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
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            peaks_mib.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    latency_ms = mean(latencies_ms)
    return {
        "latency_ms": round(latency_ms, 4),
        "tokens_per_s": round((batch["input_ids"].numel() * 1000.0) / latency_ms, 2),
        "peak_alloc_mib": round(mean(peaks_mib), 2),
    }


def main() -> int:
    args = _parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint_path, quantize_meta = _ensure_quantized_checkpoint(args)

    model = GPTQModel.load(
        str(checkpoint_path),
        device=args.device,
        backend=BACKEND.QQQ,
    )
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    qmodules = _collect_qmodules(model)

    _set_activation_bits(qmodules, 4)
    generate_start = time.perf_counter()
    generated = model.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,
        do_sample=False,
    )[0]
    generate_s = time.perf_counter() - generate_start
    text = model.tokenizer.decode(generated, skip_special_tokens=True)

    batch = _build_batch(
        model=model,
        prompt=args.prompt,
        seq_len=args.bench_seq_len,
        batch_size=args.bench_batch_size,
        device=args.device,
    )

    benchmarks: dict[str, dict[str, float]] = {}
    for activation_bits in (8, 4):
        _set_activation_bits(qmodules, activation_bits)
        _clear_mem()
        benchmarks[f"a{activation_bits}"] = _bench_prefill(
            model,
            batch,
            warmup=args.bench_warmup,
            runs=args.bench_runs,
        )

    a8_ms = benchmarks["a8"]["latency_ms"]
    a4_ms = benchmarks["a4"]["latency_ms"]
    summary = {
        "a4_vs_a8_slowdown_x": round(a4_ms / a8_ms, 4),
        "a4_vs_a8_throughput_delta_pct": round(
            ((benchmarks["a4"]["tokens_per_s"] / benchmarks["a8"]["tokens_per_s"]) - 1.0) * 100.0,
            2,
        ),
    }

    result = {
        "env": {
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
        },
        "model_path": args.model_path,
        "checkpoint_path": str(checkpoint_path),
        "quantize": quantize_meta,
        "qqq_modules": len(qmodules),
        "saved_activation_bits": qmodules[0].activation_bits,
        "generate_seconds": round(generate_s, 4),
        "generated_text": text,
        "benchmarks": benchmarks,
        "summary": summary,
    }
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
