#!/usr/bin/env python3
"""Short, reproducible VPTQ quality benchmark.

The default run follows the compact comparison used by the project's other
method PRs: one model, deterministic calibration/evaluation windows, and a
WikiText-2 perplexity row for dense, VPTQ, and GPTQ at 2/4/8 bits.  Throughput
and ARC are opt-in because they make a PR benchmark substantially longer while
not changing the quantization-quality comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig, VPTQConfig
from gptqmodel.utils.backend import BACKEND


WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
VPTQ_CONFIGS = {
    # (main index bits + residual index bits) / vector length = nominal bpw.
    "vptq2": dict(vector_lens=[-1, 8], num_centroids=[-1, 4096], num_res_centroids=[-1, 16]),
    "vptq4": dict(vector_lens=[-1, 2], num_centroids=[-1, 256], num_res_centroids=[-1, -1]),
    "vptq8": dict(vector_lens=[-1, 1], num_centroids=[-1, 256], num_res_centroids=[-1, -1]),
}


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device.")
    return device


def _load_calibration(tokenizer, blocks: int, block_size: int):
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
    text = "\n\n".join(row["text"] for row in dataset)
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=blocks * block_size,
    ).input_ids[0]
    return [
        input_ids[start : start + block_size].tolist()
        for start in range(0, min(input_ids.numel(), blocks * block_size), block_size)
        if input_ids[start : start + block_size].numel() == block_size
    ]


def _load_eval_tokens(tokenizer, max_tokens: int):
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="test")
    text = "\n\n".join(row["text"] for row in dataset)
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    ).input_ids


def _build_quant_config(name: str):
    if name in VPTQ_CONFIGS:
        return VPTQConfig(
            **VPTQ_CONFIGS[name],
            kmeans_mode="hessian",
            # Keep the PR matrix bounded; full quality studies can use more
            # Lloyd updates in a dedicated run.
            kmeans_iters=1,
            kmeans_seed=0,
        ), BACKEND.VPTQ_TORCH
    if name.startswith("gptq"):
        bits = int(name.removeprefix("gptq"))
        return (
            QuantizeConfig(
                method=METHOD.GPTQ,
                format=FORMAT.GPTQ,
                bits=bits,
                group_size=128,
                sym=True,
                desc_act=False,
            ),
            BACKEND.TORCH,
        )
    raise ValueError(f"Unsupported quantized configuration: {name}")


def _load_model(model_path: str, name: str, output_path: Path, args, tokenizer, calibration):
    if name == "dense":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": args.device},
        )
        return model.eval(), tokenizer, None

    quant_config, backend = _build_quant_config(name)
    if not output_path.exists() or not (output_path / "config.json").exists():
        model = GPTQModel.load(
            model_path,
            quantize_config=quant_config,
            dtype=torch.float16,
            device_map={"": str(args.device)},
        )
        model.quantize(
            calibration,
            calibration_concat_size=args.calibration_concat_size,
            batch_size=1,
            backend=backend,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    model = GPTQModel.load(
        str(output_path),
        backend=backend,
        dtype=torch.float16,
        device_map={"": str(args.device)},
    )
    return model.eval(), tokenizer, backend


def _model_bytes_per_weight(model) -> float | None:
    model = getattr(model, "model", model)
    replaced_bytes = 0
    replaced_weights = 0
    for module in model.modules():
        if not (hasattr(module, "qweight") or hasattr(module, "indices")):
            continue
        in_features = getattr(module, "infeatures", getattr(module, "in_features", 0))
        out_features = getattr(module, "outfeatures", getattr(module, "out_features", 0))
        if not in_features or not out_features:
            continue
        state_bytes = sum(t.numel() * t.element_size() for t in module.state_dict().values())
        replaced_bytes += state_bytes
        replaced_weights += int(in_features) * int(out_features)
    return 8.0 * replaced_bytes / replaced_weights if replaced_weights else None


@torch.inference_mode()
def _perplexity(model, tokens, args) -> float:
    tokens = tokens.to(args.device)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, tokens.shape[1] - 1, args.eval_seq_len):
        batch = tokens[:, start : start + args.eval_seq_len]
        if batch.shape[1] < 2:
            continue
        output = model(batch, use_cache=False)
        logits = output.logits[:, :-1].float()
        labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tokens += labels.numel()
    return math.exp(total_nll / total_tokens)


@torch.inference_mode()
def _throughput(model, tokenizer, args) -> tuple[float, float, float]:
    prompt = "The capital of Australia is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    warmup = max(1, args.warmup)
    for _ in range(warmup):
        out = model(input_ids, use_cache=True)
        token = out.logits[:, -1:].argmax(dim=-1)
        model(token, past_key_values=out.past_key_values, use_cache=True)
    torch.cuda.synchronize(args.device)

    prefill_values = []
    decode_values = []
    peak_mib = 0.0
    for _ in range(args.trials):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        torch.cuda.synchronize(args.device)
        prefill_values.append(input_ids.shape[1] / (time.perf_counter() - t0))
        past = out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        for _ in range(args.decode_tokens):
            out = model(token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize(args.device)
        decode_values.append(args.decode_tokens / (time.perf_counter() - t0))
        peak_mib = max(peak_mib, torch.cuda.max_memory_allocated(args.device) / 2**20)
    return statistics.median(prefill_values), statistics.median(decode_values), peak_mib


def _arc(model_path: Path, backend, args):
    from tests.eval import evaluate, get_eval_task_results

    result = evaluate(
        model_or_id_or_path=str(model_path),
        backend=backend or BACKEND.AUTO,
        model_args={"dtype": "float16", "device": str(args.device)},
        tasks=args.arc_tasks.split(","),
        batch_size=args.arc_batch_size,
        apply_chat_template=True,
        suite_kwargs={"stream": True},
    )
    return get_eval_task_results(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--configs", default="dense,vptq2,vptq4,vptq8,gptq2,gptq4,gptq8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--calibration-concat-size", type=int, default=512)
    parser.add_argument("--eval-tokens", type=int, default=8192)
    parser.add_argument("--eval-seq-len", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--arc-batch-size", type=int, default=32)
    parser.add_argument("--arc-tasks", default="arc_easy")
    parser.add_argument("--with-speed", action="store_true", help="Also measure short prefill/decode throughput.")
    parser.add_argument("--with-arc", action="store_true", help="Also run the selected ARC task(s).")
    args = parser.parse_args()
    args.device = _device(args.device)
    torch.cuda.set_device(args.device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    calibration = _load_calibration(tokenizer, args.calibration_rows, args.calibration_concat_size)
    eval_tokens = _load_eval_tokens(tokenizer, args.eval_tokens)
    if len(calibration) != args.calibration_rows:
        raise RuntimeError(
            f"Only loaded {len(calibration)} complete calibration rows; "
            f"expected {args.calibration_rows}."
        )

    config_names = [item.strip() for item in args.configs.split(",") if item.strip()]
    results = []
    for name in config_names:
        output_path = args.output_root / name
        model, model_tokenizer, backend = _load_model(args.model, name, output_path, args, tokenizer, calibration)
        ppl = _perplexity(model, eval_tokens, args)
        if args.with_speed:
            prefill, decode, peak_mib = _throughput(model, model_tokenizer, args)
        else:
            prefill = decode = peak_mib = None
        arc = (
            _arc(output_path if name != "dense" else Path(args.model), backend, args)
            if args.with_arc
            else {}
        )
        row = {
            "model": args.model,
            "config": name,
            "calibration_rows": args.calibration_rows,
            "calibration_seq_len": args.calibration_concat_size,
            "eval_tokens": args.eval_tokens,
            "ppl": ppl,
            "arc": arc,
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
            "peak_vram_mib": peak_mib,
            "bpw": _model_bytes_per_weight(model) if name != "dense" else None,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        results.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "results.json" if len(config_names) == 1 else args.output_root / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
