#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Collect dense-vs-quantized CUDA KL and Top-1/5/10 metrics for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from scripts.compare_qvq_codecs_llama_qkvo import _device_primary_metrics
from scripts.validate_qvq_lifecycle import DEFAULT_PROMPTS, QVQ_INFERENCE_DTYPE, _masked_logits


def _jsonable(value):
    """Convert nested CUDA scalar tensors returned by diagnostic reductions to JSON values."""

    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(f"Expected scalar diagnostic tensor, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Checkpoint accuracy collection requires a CUDA device.")

    tokenizer = AutoTokenizer.from_pretrained(args.reference_model, local_files_only=True)
    dense = AutoModelForCausalLM.from_pretrained(
        args.reference_model,
        torch_dtype=QVQ_INFERENCE_DTYPE,
        device_map={"": args.device},
        local_files_only=True,
    )
    quantized = GPTQModel.load(
        args.checkpoint,
        backend=BACKEND.QVQ,
        dtype=QVQ_INFERENCE_DTYPE,
        device_map={"": args.device},
        attn_implementation="eager",
    )
    dense_logits, dense_mask = _masked_logits(dense, DEFAULT_PROMPTS, tokenizer=tokenizer)
    quantized_logits, quantized_mask = _masked_logits(quantized, DEFAULT_PROMPTS)
    if not torch.equal(dense_mask, quantized_mask):
        raise AssertionError("Dense and quantized prompt masks differ")
    # Keep the primary reduction and deterministic Top-K selection on CUDA;
    # the CPU helper remains the reference for lifecycle unit tests.
    dense_device = dense_logits.to(args.device)
    quantized_device = quantized_logits.to(args.device)
    device_metrics = _device_primary_metrics(
        dense_device,
        quantized_device,
        normalize_distribution=False,
        include_top10=True,
    )
    metrics = _jsonable(device_metrics)
    payload = {
        "reference_model": args.reference_model,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "prompt_count": len(DEFAULT_PROMPTS),
        "metrics": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
