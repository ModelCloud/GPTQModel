#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.backend import normalize_backend


DEFAULT_BASELINE_MODEL = "/tmp/gptqmodel_hf/SmolLM2-135M-Instruct"
DEFAULT_KOMODO_MODEL = "/tmp/gptqmodel_quantized/SmolLM2-135M-Instruct-gptq-npu"
DEFAULT_PROMPT = (
    "You are measuring language model inference throughput. Explain the difference between "
    "prefill and decode in one concise paragraph."
)


@dataclass(frozen=True)
class BenchTarget:
    label: str
    path: str
    loader: str
    backend: str | None = None


def _dtype(name: str) -> torch.dtype:
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype `{name}`.")


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _memory_backend(device: torch.device):
    if device.type == "npu":
        return torch.npu
    if device.type == "cuda":
        return torch.cuda
    return None


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    backend = _memory_backend(device)
    if backend is None:
        return {}

    memory = {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        fn = getattr(backend, name, None)
        if fn is None:
            continue
        try:
            memory[name] = int(fn(device))
        except TypeError:
            memory[name] = int(fn())
    return memory


def _reset_peak_memory(device: torch.device) -> None:
    backend = _memory_backend(device)
    if backend is None:
        return
    _sync(device)
    backend.empty_cache()
    _sync(device)
    reset_fn = getattr(backend, "reset_peak_memory_stats", None)
    if reset_fn is None:
        return
    try:
        reset_fn(device)
    except TypeError:
        reset_fn()


def _bytes_to_gib(value: int | None) -> float:
    return float(value or 0) / (1024 ** 3)


def _model_device(model: Any, fallback: torch.device) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def _prepare_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _build_prompt(prompt: str, tokenizer, *, prompt_tokens: int, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    text = prompt
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < prompt_tokens:
        text = f"{text}\n{prompt}"

    encoded = tokenizer(
        [text] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=prompt_tokens,
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def _load_transformers(path: str, *, device: torch.device, dtype: torch.dtype, attn_implementation: str):
    kwargs = {"dtype": dtype}
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    tokenizer = _prepare_tokenizer(path)
    return model, tokenizer, None


def _load_gptqmodel(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    backend: str | None,
):
    kwargs = {"device": str(device), "dtype": dtype}
    if backend:
        kwargs["backend"] = normalize_backend(backend)
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    wrapper = GPTQModel.load(path, **kwargs)
    model = getattr(wrapper, "model", wrapper).eval()
    tokenizer = getattr(wrapper, "tokenizer", None) or _prepare_tokenizer(path)
    return model, tokenizer, wrapper


def _evalution_session_model_tokenizer(session: Any) -> tuple[Any, Any]:
    model = getattr(session, "model", None) or getattr(session, "_model", None)
    tokenizer = getattr(session, "tokenizer", None) or getattr(session, "_tokenizer", None)
    if model is None or tokenizer is None:
        raise RuntimeError(
            "Evalution session did not expose a model/tokenizer pair needed for split prefill/decode timing."
        )
    return model, tokenizer


def _load_evalution_gptqmodel(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    backend: str | None,
):
    try:
        import evalution
        from tests.eval import _build_evalution_runtime
    except Exception as exc:
        raise RuntimeError("Evalution is not installed; run `pip install Evalution` to use this loader.") from exc

    model_args: dict[str, Any] = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "padding_side": "left",
    }
    if attn_implementation != "auto":
        model_args["attn_implementation"] = attn_implementation

    _engine, _model_config, session = _build_evalution_runtime(
        evalution=evalution,
        model_or_id_or_path=path,
        tokenizer=None,
        tasks=["arc_challenge"],
        batch_size=1,
        trust_remote_code=False,
        output_path=None,
        llm_backend="gptqmodel",
        backend=normalize_backend(backend or BACKEND.AUTO),
        model_args=model_args,
    )
    model, tokenizer = _evalution_session_model_tokenizer(session)
    return model.eval(), tokenizer, session


def _load_target(target: BenchTarget, *, device: torch.device, dtype: torch.dtype, attn_implementation: str):
    if target.loader == "transformers":
        return _load_transformers(
            target.path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    if target.loader == "gptqmodel":
        return _load_gptqmodel(
            target.path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            backend=target.backend,
        )
    if target.loader == "evalution-gptqmodel":
        return _load_evalution_gptqmodel(
            target.path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            backend=target.backend,
        )
    raise ValueError(f"Unsupported loader `{target.loader}`.")


def _prefill_decode_once(
    model,
    inputs: dict[str, torch.Tensor],
    *,
    new_tokens: int,
    device: torch.device,
) -> dict[str, float]:
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

    _sync(device)
    decode_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(new_tokens):
            decode_attention_mask = torch.cat(
                [
                    decode_attention_mask,
                    torch.ones((decode_attention_mask.shape[0], 1), dtype=decode_attention_mask.dtype, device=device),
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
    generated_tokens = int(input_ids.shape[0] * new_tokens)
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tps": prompt_tokens / prefill_s if prefill_s > 0 else float("inf"),
        "decode_tps": generated_tokens / decode_s if decode_s > 0 else float("inf"),
    }


def _measure_target(
    target: BenchTarget,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    prompt: str,
    prompt_tokens: int,
    new_tokens: int,
    batch_size: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    _empty_cache(device)
    _reset_peak_memory(device)
    load_start = time.perf_counter()
    model, tokenizer, owner = _load_target(
        target,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    actual_device = _model_device(model, fallback=device)
    _sync(actual_device)
    load_s = time.perf_counter() - load_start
    load_memory = _memory_snapshot(actual_device)

    inputs = _build_prompt(prompt, tokenizer, prompt_tokens=prompt_tokens, batch_size=batch_size, device=actual_device)
    effective_attn = (
        getattr(getattr(model, "config", None), "_attn_implementation", None)
        or getattr(getattr(model, "config", None), "attn_implementation", None)
        or attn_implementation
    )

    _reset_peak_memory(actual_device)
    for _ in range(warmup):
        _prefill_decode_once(model, inputs, new_tokens=new_tokens, device=actual_device)

    samples = [
        _prefill_decode_once(model, inputs, new_tokens=new_tokens, device=actual_device)
        for _ in range(runs)
    ]
    inference_memory = _memory_snapshot(actual_device)
    prefill_tps = [sample["prefill_tps"] for sample in samples]
    decode_tps = [sample["decode_tps"] for sample in samples]
    prefill_s = [sample["prefill_s"] for sample in samples]
    decode_s = [sample["decode_s"] for sample in samples]

    result = {
        "label": target.label,
        "path": target.path,
        "loader": target.loader,
        "backend": target.backend,
        "device": str(actual_device),
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": attn_implementation,
        "effective_attn_implementation": effective_attn,
        "batch_size": batch_size,
        "prompt_tokens_per_request": prompt_tokens,
        "new_tokens_per_request": new_tokens,
        "prompt_tokens_total": batch_size * prompt_tokens,
        "new_tokens_total": batch_size * new_tokens,
        "warmup": warmup,
        "runs": runs,
        "load_s": load_s,
        "prefill_tps_mean": statistics.fmean(prefill_tps),
        "prefill_tps_median": statistics.median(prefill_tps),
        "decode_tps_mean": statistics.fmean(decode_tps),
        "decode_tps_median": statistics.median(decode_tps),
        "prefill_ms_mean": statistics.fmean(prefill_s) * 1000.0,
        "decode_ms_mean": statistics.fmean(decode_s) * 1000.0,
        "load_allocated_gib": _bytes_to_gib(load_memory.get("memory_allocated")),
        "load_reserved_gib": _bytes_to_gib(load_memory.get("memory_reserved")),
        "load_peak_allocated_gib": _bytes_to_gib(load_memory.get("max_memory_allocated")),
        "load_peak_reserved_gib": _bytes_to_gib(load_memory.get("max_memory_reserved")),
        "inference_allocated_gib": _bytes_to_gib(inference_memory.get("memory_allocated")),
        "inference_reserved_gib": _bytes_to_gib(inference_memory.get("memory_reserved")),
        "inference_peak_allocated_gib": _bytes_to_gib(inference_memory.get("max_memory_allocated")),
        "inference_peak_reserved_gib": _bytes_to_gib(inference_memory.get("max_memory_reserved")),
        "samples": samples,
    }

    close = getattr(owner, "close", None)
    if callable(close):
        close()
    del model, tokenizer, owner
    _empty_cache(device)
    return result


def _default_targets(args) -> list[BenchTarget]:
    targets = []
    if not args.skip_baseline and Path(args.baseline_model).exists():
        targets.append(BenchTarget("baseline_dense", args.baseline_model, "transformers", None))
    if not args.skip_komodo and Path(args.komodo_model).exists():
        targets.append(BenchTarget("komodo_gptq", args.komodo_model, args.komodo_loader, args.komodo_backend))
    if not targets:
        raise ValueError("No benchmark targets selected. Provide existing --baseline-model/--komodo-model paths.")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure split prefill and decode/new-token TPS for small GPT-style models."
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--komodo-model", default=DEFAULT_KOMODO_MODEL)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-komodo", action="store_true")
    parser.add_argument(
        "--komodo-loader",
        choices=("gptqmodel", "evalution-gptqmodel"),
        default="gptqmodel",
        help="Use evalution-gptqmodel to load the Komodo case through Evalution.",
    )
    parser.add_argument("--komodo-backend", default="komodo")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--attn-implementations", nargs="+", default=["eager", "flash_attention_2"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    targets = _default_targets(args)

    results = []
    for attn_implementation in args.attn_implementations:
        for target in targets:
            result = _measure_target(
                target,
                device=device,
                dtype=dtype,
                attn_implementation=attn_implementation,
                prompt=args.prompt,
                prompt_tokens=args.prompt_tokens,
                new_tokens=args.new_tokens,
                batch_size=args.batch_size,
                warmup=args.warmup,
                runs=args.runs,
            )
            results.append(result)
            print(
                "{label} loader={loader} backend={backend} attn={effective_attn_implementation} "
                "prefill={prefill_tps_mean:.2f} tok/s decode={decode_tps_mean:.2f} tok/s "
                "prefill_ms={prefill_ms_mean:.3f} decode_ms={decode_ms_mean:.3f} "
                "load_peak={load_peak_allocated_gib:.3f}GiB "
                "infer_peak={inference_peak_allocated_gib:.3f}GiB".format(**result)
            )

    payload = {
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "results": results,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
