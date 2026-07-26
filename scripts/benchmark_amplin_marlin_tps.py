#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU TPS benchmark: GPTQModel Qwen3 8B with GPTQ_AMPLIN vs GPTQ_MARLIN.

The script includes a pre-CUDA idle GPU preflight, then loads the model twice
(once per backend) and measures split prefill/decode tokens/sec.  Prefill is
restricted to 32 tokens because Amplin dynamic currently supports M <= 32.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, ContinuousBatchingConfig


# Make repo root importable when running the script directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run(cmd: list[str], timeout: int = 30) -> str:
    return subprocess.check_output(cmd, text=True, timeout=timeout)


def _query_gpus() -> list[dict[str, Any]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid,name,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, pci, uuid, name, util_str, mem_str = parts
        util_match = re.search(r"(\d+)", util_str)
        mem_match = re.search(r"(\d+)", mem_str)
        gpus.append(
            {
                "index": int(idx),
                "pci": pci,
                "uuid": uuid,
                "name": name,
                "utilization": int(util_match.group(1)) if util_match else -1,
                "memory_mib": int(mem_match.group(1)) if mem_match else -1,
            }
        )
    return gpus


def _find_idle_gpu(
    *,
    samples: int = 3,
    sample_interval_s: float = 0.5,
    max_utilization: int = 0,
    max_memory_mib: int = 200,
) -> dict[str, Any]:
    candidates = None
    for _ in range(samples):
        gpus = _query_gpus()
        acceptable = [
            g
            for g in gpus
            if g["utilization"] <= max_utilization and g["memory_mib"] <= max_memory_mib
        ]
        if candidates is None:
            candidates = acceptable
        else:
            candidate_uuids = {g["uuid"] for g in candidates}
            sample_uuids = {g["uuid"] for g in acceptable}
            candidate_uuids &= sample_uuids
            candidates = [g for g in candidates if g["uuid"] in candidate_uuids]
        if not candidates:
            raise RuntimeError(f"No GPU stayed idle across samples. Last query: {gpus}")
        if _ < samples - 1:
            time.sleep(sample_interval_s)
    return candidates[0]


def _set_gpu_env(gpu: dict[str, Any]) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])


def _preflight_and_print() -> dict[str, Any]:
    gpu = _find_idle_gpu()
    _set_gpu_env(gpu)
    print(
        f"Preflight passed: physical GPU {gpu['index']} ({gpu['name']}) "
        f"PCI {gpu['pci']} UUID {gpu['uuid']} "
        f"util={gpu['utilization']}% mem={gpu['memory_mib']}MiB"
    )
    return gpu


# Pre-flight is stdlib-only above this line.  Import PyTorch/GPT-QModel below.
gpu_info = _preflight_and_print()

import torch

from gptqmodel import GPTQModel
from gptqmodel.utils.amplin import clear_thread_caches


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _dtype(name: str) -> torch.dtype:
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype `{name}`.")


def _build_prompt(prompt: str, tokenizer, prompt_tokens: int, device: torch.device, batch: int = 1):
    text = prompt
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < prompt_tokens:
        text = f"{text}\n{prompt}"
    encoded = tokenizer(
        [text] * batch,
        padding="max_length",
        truncation=True,
        max_length=prompt_tokens,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_ids = encoded["input_ids"].tolist()
    return inputs, prompt_ids


def _full_prefill_decode_cycle(
    model,
    inputs: dict[str, torch.Tensor],
    *,
    new_tokens: int,
    device: torch.device,
) -> None:
    """Run one full prefill + decode cycle without timing; used for warmup."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        decode_attention_mask = attention_mask
        for _ in range(new_tokens):
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)


def _steady_prefill_decode_sample(
    model,
    inputs: dict[str, torch.Tensor],
    *,
    new_tokens: int,
    device: torch.device,
):
    """Timed prefill + decode sample.

    The first prefill is timed while M=32 layouts are GPU-resident.  The first
    decode token is not timed because it pays the one-time packed-layout switch;
    timing covers the remaining ``new_tokens - 1`` steady decode tokens.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    _sync(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _sync(device)
    prefill_s = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    decode_attention_mask = attention_mask

    # Cold decode token: switches from the M=32 packed layout to the M=1 layout.
    # Do not include it in the timed decode TPS.
    _sync(device)
    with torch.inference_mode():
        decode_attention_mask = torch.cat(
            [
                decode_attention_mask,
                torch.ones(
                    (decode_attention_mask.shape[0], 1),
                    dtype=decode_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=decode_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    _sync(device)

    timed_tokens = max(1, new_tokens - 1)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(timed_tokens):
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    _sync(device)
    decode_s = time.perf_counter() - decode_start

    prompt_tokens = int(input_ids.numel())
    generated_tokens = int(input_ids.shape[0] * timed_tokens)
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tps": prompt_tokens / prefill_s if prefill_s > 0 else float("inf"),
        "decode_tps": generated_tokens / decode_s if decode_s > 0 else float("inf"),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def _load_model(path: str, backend: str, device: str, dtype: torch.dtype, attn: str):
    wrapper = GPTQModel.load(
        path,
        backend=backend,
        device=device,
        dtype=dtype,
        attn_implementation=attn,
    )
    model = getattr(wrapper, "model", wrapper).eval()
    tokenizer = getattr(wrapper, "tokenizer", None) or AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, wrapper


def _generate_batch_sample(
    model,
    prompt_ids: list[list[int]],
    new_tokens: int,
    cb_config,
    device: torch.device,
) -> dict[str, Any]:
    """Run one timed `generate_batch` call and split prefill/decode TPS.

    The continuous batcher processes all requests in parallel.  ``record_timestamps``
    captures the time each generated token is produced; the first timestamp marks
    the end of prefill (time-to-first-token) and the last timestamp marks the
    end of generation.  This gives a clean ``prefill`` (prompt tokens / TTFT) and
    ``decode`` (generated tokens / inter-token wall time) split.
    """
    _sync(device)
    results = model.generate_batch(
        prompt_ids,
        max_new_tokens=new_tokens,
        continuous_batching_config=cb_config,
        progress_bar=False,
        warmup=False,
        persistent_manager=False,
        record_timestamps=True,
    )
    _sync(device)

    created = min(r.created_time for r in results.values())
    first_tokens = [r.timestamps[0] for r in results.values() if r.timestamps]
    last_tokens = [r.timestamps[-1] for r in results.values() if r.timestamps]
    if not first_tokens or not last_tokens:
        # Degenerate case: fall back to lifespan total.
        finish = max(r.lifespan[1] for r in results.values())
        total_s = finish - created
        prefill_tokens = sum(len(r.prompt_ids) for r in results.values())
        generated_tokens = sum(len(r.generated_tokens) for r in results.values())
        return {
            "prefill_s": total_s,
            "decode_s": total_s,
            "prefill_tps": prefill_tokens / total_s,
            "decode_tps": generated_tokens / total_s,
            "prompt_tokens": prefill_tokens,
            "generated_tokens": generated_tokens,
        }

    first_token = min(first_tokens)
    last_token = max(last_tokens)
    prefill_s = first_token - created
    decode_s = last_token - first_token
    prefill_tokens = sum(len(r.prompt_ids) for r in results.values())
    generated_tokens = sum(len(r.generated_tokens) for r in results.values())

    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tps": prefill_tokens / prefill_s if prefill_s > 0 else float("inf"),
        "decode_tps": generated_tokens / decode_s if decode_s > 0 else float("inf"),
        "prompt_tokens": prefill_tokens,
        "generated_tokens": generated_tokens,
    }


def _measure_backend(
    path: str,
    backend: str,
    device: str,
    dtype: torch.dtype,
    attn: str,
    prompt: str,
    prompt_tokens: int,
    new_tokens: int,
    warmup: int,
    runs: int,
    batch: int = 1,
    continuous_batching: bool = False,
) -> dict[str, Any]:
    torch.cuda.set_device(torch.device(device))
    _empty_cache(torch.device(device))
    clear_thread_caches()

    load_start = time.perf_counter()
    model, tokenizer, wrapper = _load_model(path, backend, device, dtype, attn)
    actual_device = next(model.parameters()).device
    _sync(actual_device)
    load_s = time.perf_counter() - load_start

    use_cb = continuous_batching or ("paged" in attn)
    inputs, prompt_ids = _build_prompt(prompt, tokenizer, prompt_tokens, actual_device, batch)

    samples = []
    max_allocated = 0
    max_reserved = 0
    allocated = 0

    if use_cb:
        # Paged Flash Attention requires CUDA graph capture to be disabled in
        # the continuous batching config.
        cb_config = ContinuousBatchingConfig(use_cuda_graph=(False, False))
        # The default num_blocks allocates the entire free GPU memory for the
        # paged KV cache, which makes the peak-VRAM metric dominated by the
        # cache rather than the model weights.  Cap it to the blocks needed for
        # this workload plus a small headroom block.
        tokens_per_request = prompt_tokens + new_tokens
        blocks_per_request = (tokens_per_request + cb_config.block_size - 1) // cb_config.block_size + 1
        cb_config.num_blocks = batch * blocks_per_request

        cfg = model.generation_config
        cfg.do_sample = False
        cfg.temperature = 1.0
        cfg.top_k = 1
        cfg.top_p = 1.0
        cfg.repetition_penalty = 1.0
        cfg.eos_token_id = []

        _sync(actual_device)
        for _ in range(warmup):
            model.generate_batch(
                prompt_ids,
                max_new_tokens=new_tokens,
                continuous_batching_config=cb_config,
                progress_bar=False,
                warmup=False,
                persistent_manager=False,
            )
        _sync(actual_device)
        _empty_cache(actual_device)

        for _ in range(runs):
            _sync(actual_device)
            _empty_cache(actual_device)
            torch.cuda.reset_peak_memory_stats(actual_device)
            sample = _generate_batch_sample(model, prompt_ids, new_tokens, cb_config, actual_device)
            mem = _memory_snapshot(actual_device)
            sample["memory"] = mem
            samples.append(sample)
            max_allocated = max(max_allocated, mem["max_allocated"])
            max_reserved = max(max_reserved, mem["max_reserved"])
            allocated = mem["allocated"]
    else:
        _sync(actual_device)
        for _ in range(warmup):
            _full_prefill_decode_cycle(model, inputs, new_tokens=new_tokens, device=actual_device)

        # Settle packed layouts and reset memory statistics so the timed region
        # reflects steady-state inference rather than warmup/microbenchmark peaks.
        with torch.inference_mode():
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
            )
        _sync(actual_device)
        _empty_cache(actual_device)
        torch.cuda.reset_peak_memory_stats(actual_device)

        for _ in range(runs):
            # Restore prefill packed layouts before the timed prefill.  The first
            # iteration reuses the settle prefill above; subsequent iterations need
            # an explicit restore after the previous timed decode switched layout.
            with torch.inference_mode():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=True,
                )
            _sync(actual_device)
            _empty_cache(actual_device)
            torch.cuda.reset_peak_memory_stats(actual_device)

            sample = _steady_prefill_decode_sample(
                model, inputs, new_tokens=new_tokens, device=actual_device
            )
            mem = _memory_snapshot(actual_device)
            sample["memory"] = mem
            samples.append(sample)
            max_allocated = max(max_allocated, mem["max_allocated"])
            max_reserved = max(max_reserved, mem["max_reserved"])
            allocated = mem["allocated"]

    del model, tokenizer, wrapper
    clear_thread_caches()
    _empty_cache(actual_device)

    prefill_tps = [s["prefill_tps"] for s in samples]
    decode_tps = [s["decode_tps"] for s in samples]
    prefill_s = [s["prefill_s"] for s in samples]
    decode_s = [s["decode_s"] for s in samples]
    return {
        "backend": backend,
        "device": str(actual_device),
        "load_s": load_s,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "batch": batch,
        "prefill_tps_mean": statistics.fmean(prefill_tps),
        "prefill_tps_median": statistics.median(prefill_tps),
        "decode_tps_mean": statistics.fmean(decode_tps),
        "decode_tps_median": statistics.median(decode_tps),
        "prefill_ms_mean": statistics.fmean(prefill_s) * 1000.0,
        "prefill_ms_median": statistics.median(prefill_s) * 1000.0,
        "decode_ms_mean": statistics.fmean(decode_s) * 1000.0,
        "decode_ms_median": statistics.median(decode_s) * 1000.0,
        "max_allocated_gib": max_allocated / (1024 ** 3),
        "max_reserved_gib": max_reserved / (1024 ** 3),
        "allocated_gib": allocated / (1024 ** 3),
        "samples": samples,
    }


def _gsm8k_question() -> str:
    gsm8k_dir = Path("/monster/data/model/dataset/gsm8k/main")
    parquet = list(gsm8k_dir.glob("*.parquet"))
    if parquet:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet[0], columns=["question"])
            return str(table.column("question")[0].as_py())
        except Exception:
            pass
    return (
        "Janet buys 3 pounds of broccoli for $4 a pound and 3 pounds of carrots for $2 a pound. "
        "How many dollars does she spend?"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPTQ_AMPLIN vs GPTQ_MARLIN decode TPS on a single GPU."
    )
    parser.add_argument(
        "--model-path",
        default="/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-cal512",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["gptq_marlin", "gptq_amplin"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--continuous-batching", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    prompt = args.prompt or _gsm8k_question()
    dtype = _dtype(args.dtype)

    results = []
    for backend in args.backends:
        print(f"\n=== Backend: {backend} ===")
        result = _measure_backend(
            args.model_path,
            backend,
            args.device,
            dtype,
            args.attn_implementation,
            prompt,
            args.prompt_tokens,
            args.new_tokens,
            args.warmup,
            args.runs,
            batch=args.batch,
            continuous_batching=args.continuous_batching,
        )
        results.append(result)
        print(
            f"  prefill: {result['prefill_tps_mean']:.2f} tok/s "
            f"({result['prefill_ms_mean']:.2f} ms) | "
            f"decode: {result['decode_tps_mean']:.2f} tok/s "
            f"({result['decode_ms_mean']:.2f} ms) | "
            f"peak mem: {result['max_allocated_gib']:.2f} GiB"
        )

    payload = {
        "gpu": gpu_info,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "continuous_batching": args.continuous_batching,
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "prompt": prompt,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
