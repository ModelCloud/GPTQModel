#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""WikiText-2 perplexity for the ADMM-Q weight quantizer.

Quantizes one pretrained model with RTN, GPTQ and ADMM-Q under the same
calibration windows, reloads each saved checkpoint, and scores all of them
(plus the unquantized model) on the same WikiText-2 test tokens.

Example:
    python scripts/benchmark_admmq_ppl.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --configs dense,rtn,gptq,admmq

Calibration and evaluation use disjoint splits and a fixed tokenizer, so the
row for each config differs only in the weight solver.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    ADMMQConfig,
    QuantizeConfig,
    RTNConfig,
)
from gptqmodel.utils.backend import BACKEND


WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
BACKENDS = {
    "auto": BACKEND.AUTO,
    "gptq_torch": BACKEND.GPTQ_TORCH,
}


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA or ROCm device.")
    return device


def _load_windows(tokenizer, split: str, rows: int, seq_len: int) -> torch.Tensor:
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=split)
    text = "\n\n".join(row["text"] for row in dataset)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=rows * seq_len,
    ).input_ids
    usable = (tokens.shape[1] // seq_len) * seq_len
    if usable < seq_len:
        raise RuntimeError(f"Only {tokens.shape[1]} tokens available for split {split!r}.")
    return tokens[:, :usable]


def _build_config(name: str, args: argparse.Namespace):
    common = {
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "desc_act": False,
        "format": FORMAT.GPTQ,
        "device": args.device,
        "calibration_data_device": args.device,
    }
    if name == "rtn":
        return RTNConfig(**common)
    if name == "gptq":
        return QuantizeConfig(method=METHOD.GPTQ, **common)
    if name == "admmq" or name.startswith("admmq-"):
        overrides = {}
        if name != "admmq":
            overrides["max_iter"] = int(name.removeprefix("admmq-"))
        return ADMMQConfig(**common, offload_to_disk=False, **overrides)
    raise ValueError(f"Unsupported config: {name}")


def _quantize(name: str, args: argparse.Namespace, calibration: list[str], output_dir: Path):
    quant_config = _build_config(name, args)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    model = GPTQModel.load(args.model, quant_config, dtype=torch.float16)
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=args.batch_size),
        backend=BACKENDS[args.backend],
    )
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(args.device) / 2**30
    saved = output_dir / name
    model.save(str(saved))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return saved, elapsed, peak_gib


@torch.inference_mode()
def _perplexity(model, tokens: torch.Tensor, device: torch.device, seq_len: int) -> float:
    tokens = tokens.to(device)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, tokens.shape[1] - 1, seq_len):
        batch = tokens[:, start : start + seq_len]
        if batch.shape[1] < 2:
            continue
        logits = model(batch, use_cache=False).logits[:, :-1].float()
        labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        total_nll += float(loss)
        total_tokens += labels.numel()
    return math.exp(total_nll / total_tokens)


def _evaluate(name: str, checkpoint: Path, args: argparse.Namespace, eval_tokens):
    model = GPTQModel.load(
        str(checkpoint),
        backend=BACKENDS[args.backend],
        dtype=torch.float16,
    )
    ppl = _perplexity(model.model, eval_tokens, args.device, args.eval_seq_len)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or Hub id of the pretrained model")
    parser.add_argument("--configs", default="dense,rtn,gptq,admmq")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="auto")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/admmq_ppl"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.device = _device(args.device)
    torch.cuda.set_device(args.device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    calibration_ids = _load_windows(
        tokenizer, "train", args.calibration_rows, args.calibration_seq_len
    )
    calibration = [
        tokenizer.decode(row.tolist())
        for row in calibration_ids.reshape(-1, args.calibration_seq_len)
    ]
    eval_tokens = _load_windows(tokenizer, "test", args.eval_rows, args.eval_seq_len)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in [item.strip() for item in args.configs.split(",") if item.strip()]:
        if name == "dense":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map={"": args.device},
                local_files_only=args.local_files_only,
            ).eval()
            ppl = _perplexity(model, eval_tokens, args.device, args.eval_seq_len)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            row = {"config": name, "ppl": ppl}
        else:
            checkpoint, seconds, peak_gib = _quantize(name, args, calibration, args.output_dir)
            ppl = _evaluate(name, checkpoint, args, eval_tokens)
            row = {
                "config": name,
                "ppl": ppl,
                "quant_seconds": round(seconds, 1),
                "peak_gib": round(peak_gib, 2),
            }
        row.update(
            model=args.model,
            calibration_rows=args.calibration_rows,
            calibration_seq_len=args.calibration_seq_len,
            eval_tokens=int(eval_tokens.shape[1]),
            eval_seq_len=args.eval_seq_len,
            bits=args.bits,
            group_size=args.group_size,
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    reference = next((row["ppl"] for row in rows if row["config"] == "gptq"), None)
    if reference:
        for row in rows:
            row["delta_vs_gptq_pct"] = round((row["ppl"] / reference - 1.0) * 100.0, 2)
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
