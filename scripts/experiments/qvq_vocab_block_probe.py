#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure bounded YAQA/GSQ preparation for a QVQ vocabulary head.

Research probe only: it does not save a model or change serving weights.
Run with ``python -m scripts.experiments.qvq_vocab_block_probe`` from repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--block-rows", type=int, default=2048)
    parser.add_argument("--gram-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bits", type=float, default=3.5, choices=[2.5, 3, 3.5, 4])
    parser.add_argument("--gsq-steps", type=int, default=4)
    parser.add_argument("--gsq-candidates", type=int, default=3)
    parser.add_argument("--gsq-coordinate-sweeps", type=int, default=0)
    parser.add_argument("--compare-no-gsq", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--factor-mode", choices=["shared-head", "independent-blocks"],
                        default="shared-head")
    parser.add_argument("--factor-cache", type=Path)
    parser.add_argument("--reuse-factor-cache", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_shared_factor_cache(path: Path, inputs, outputs, metadata: dict[str, str]) -> None:
    """Save the compact full-head YAQA sketch without a dense output Gram."""
    from safetensors.torch import save_file

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite YAQA factor cache: {path}")
    tensors = {}
    for prefix, sketch in (("input", inputs["lm_head"]), ("output", outputs["lm_head"])):
        tensors[f"{prefix}_source"] = sketch.source.contiguous()
        tensors[f"{prefix}_diagonal"] = sketch.diagonal.contiguous()
        tensors[f"{prefix}_source_diagonal"] = sketch.source_diagonal.contiguous()
        metadata[f"{prefix}_normalizer"] = repr(sketch.normalizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_shared_factor_cache(path: Path, expected: dict[str, str]):
    """Load only a factor with a matching model, corpus, and capture policy."""
    from safetensors import safe_open

    from gptqmodel.quantization.qvq_yaqa import YaqaGramSketch

    with safe_open(path, framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("YAQA factor-cache provenance does not match the requested run")
        sketches = {}
        for prefix in ("input", "output"):
            sketches[prefix] = YaqaGramSketch(
                source=stored.get_tensor(f"{prefix}_source"),
                diagonal=stored.get_tensor(f"{prefix}_diagonal"),
                source_diagonal=stored.get_tensor(f"{prefix}_source_diagonal"),
                normalizer=float(metadata[f"{prefix}_normalizer"]),
                seed=int(expected["seed"]),
                _source_diagonal_validated=True,
            )
    return {"lm_head": sketches["input"]}, {"lm_head": sketches["output"]}, {
        "independent_sequences": int(metadata["independent_sequences"]),
        "valid_output_samples": int(metadata["valid_tokens"]),
    }


def main() -> None:
    args = parse_args()
    import torch

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization.config import GSQConfig
    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b
    from gptqmodel.utils.model import untie_word_embeddings
    from optimize._common import load_calibration_data
    from optimize.qvq_vocab_blocks import VocabBlockLinear

    if args.dataset_size < 1 or args.gram_rank < 1 or args.batch_size < 1:
        raise ValueError("dataset size, Gram rank, and batch size must be positive")
    if args.gsq_steps < 1 or args.gsq_candidates < 2:
        raise ValueError("GSQ steps must be positive and candidates must be at least 2")
    if args.gsq_coordinate_sweeps < 0:
        raise ValueError("GSQ coordinate sweeps must be nonnegative")
    if args.reuse_factor_cache and args.factor_cache is None:
        raise ValueError("--reuse-factor-cache requires --factor-cache")
    if args.factor_cache is not None and args.factor_mode != "shared-head":
        raise ValueError("factor caching requires --factor-mode shared-head")
    started = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    model = GPTQModel.load(
        str(args.model_path), device=DEVICE.CUDA, backend=BACKEND.AUTO,
        attn_implementation="eager",
    )
    model.model = untie_word_embeddings(model.model)
    dense_head = model.get_output_embeddings()
    block_count = (dense_head.out_features + args.block_rows - 1) // args.block_rows
    if not 0 <= args.block_index < block_count:
        raise ValueError("block index exceeds head block count")
    if args.factor_mode == "independent-blocks":
        head = VocabBlockLinear(dense_head, args.block_rows).eval()
        model.model.set_output_embeddings(head)
        targets = head.yaqa_targets()
    else:
        head = dense_head
        targets = {"lm_head": dense_head}
    calibration_sha256 = sha256_file(args.calibration_parquet)
    factor_identity = {
        "schema": "qvq.yaqa.shared-head-factor.v2",
        "model_path": str(args.model_path.resolve()),
        "calibration_sha256": calibration_sha256,
        "requested_sequences": str(args.dataset_size),
        "gram_rank": str(args.gram_rank),
        "batch_size": str(args.batch_size),
        "seed": str(args.seed),
        "input_features": str(dense_head.in_features),
        "output_features": str(dense_head.out_features),
    }
    if args.reuse_factor_cache:
        inputs, outputs, stats = load_shared_factor_cache(args.factor_cache, factor_identity)
        capture_seconds = 0.0
    else:
        source = load_calibration_data(parquet_path=str(args.calibration_parquet), dataset_size=args.dataset_size)
        batches = model.prepare_dataset(
            calibration_dataset=source, batch_size=args.batch_size,
            calibration_data_min_length=8,
        )
        capture_start = time.monotonic()
        inputs, outputs, stats = capture_yaqa_sketch_b(
            model.model, batches, targets, device=torch.device("cuda:0"),
            seed=args.seed, minimum_sequences=args.dataset_size, first_decoder_layer=head,
            gram_strategy="streaming_projected", gram_projection_rank=args.gram_rank,
        )
        torch.cuda.synchronize()
        capture_seconds = time.monotonic() - capture_start
        if args.factor_cache is not None:
            save_shared_factor_cache(args.factor_cache, inputs, outputs, {
                **factor_identity,
                "independent_sequences": str(stats["independent_sequences"]),
                "valid_tokens": str(stats["valid_output_samples"]),
            })
    start = args.block_index * args.block_rows
    stop = min(start + args.block_rows, dense_head.out_features)
    if args.factor_mode == "shared-head":
        input_hessian = inputs["lm_head"].materialize(device=torch.device("cuda:0"))
        output_factor = outputs["lm_head"].factor(device=torch.device("cuda:0"))
        block_factor = output_factor[start:stop]
        output_hessian = block_factor @ block_factor.T
        output_hessian = (output_hessian + output_hessian.T) * 0.5
        weight = dense_head.weight.detach()[start:stop]
    else:
        name = f"lm_head.blocks.{args.block_index}"
        input_hessian = inputs[name].materialize(device=torch.device("cuda:0"))
        output_hessian = outputs[name].materialize(device=torch.device("cuda:0"))
        weight = head.blocks[args.block_index].weight.detach()
    # Capture consumes global RNG state; reset at the quantization boundary so
    # a cached-factor sweep is identical to a fresh capture with the same seed.
    torch.manual_seed(args.seed)
    quant_start = time.monotonic()
    result = quantize_qvq_linear(
        weight, input_hessian,
        bits=args.bits, output_hessian=output_hessian,
        seed=args.seed, damp_percent=0.05, rounding="yaqa",
        v2b2_p32=args.bits < 4, bank_count=2 if args.bits < 4 else 1,
        gsq=GSQConfig(enabled=True, steps=args.gsq_steps,
                      candidates=args.gsq_candidates, seed=args.seed,
                      qvq_coordinate_sweeps=args.gsq_coordinate_sweeps),
    )
    torch.cuda.synchronize()
    quant_seconds = time.monotonic() - quant_start
    error = result.weight.float() - weight.float()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    damped_input = input_hessian.clone()
    damped_output = output_hessian.clone()
    input_damping = max(float(input_hessian.diagonal().abs().mean()) * 0.05,
                        torch.finfo(torch.float32).eps)
    output_damping = max(float(output_hessian.diagonal().abs().mean()) * 0.05,
                         torch.finfo(torch.float32).eps)
    damped_input.diagonal().add_(input_damping)
    damped_output.diagonal().add_(output_damping)
    oracle = {}
    damped_oracle = {}
    for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
        e, h, g = error.to(dtype), input_hessian.to(dtype), output_hessian.to(dtype)
        oracle[label] = float(torch.einsum("oi,ij,pj,op->", e, h, e, g).item())
        damped_oracle[label] = float(torch.einsum(
            "oi,ij,pj,op->", e, damped_input.to(dtype), e, damped_output.to(dtype),
        ).item())
    baseline_oracle = None
    baseline_damped_oracle = None
    if args.compare_no_gsq:
        torch.manual_seed(args.seed)
        baseline = quantize_qvq_linear(
            weight, input_hessian, bits=args.bits, output_hessian=output_hessian,
            seed=args.seed, damp_percent=0.05, rounding="yaqa",
            v2b2_p32=args.bits < 4, bank_count=2 if args.bits < 4 else 1,
            gsq=None,
        )
        baseline_error = baseline.weight.float() - weight.float()
        baseline_oracle = {}
        baseline_damped_oracle = {}
        for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            e, h, g = baseline_error.to(dtype), input_hessian.to(dtype), output_hessian.to(dtype)
            baseline_oracle[label] = float(torch.einsum("oi,ij,pj,op->", e, h, e, g).item())
            baseline_damped_oracle[label] = float(torch.einsum(
                "oi,ij,pj,op->", e, damped_input.to(dtype), e, damped_output.to(dtype),
            ).item())
    record = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "calibration_path": str(args.calibration_parquet.resolve()),
        "calibration_sha256": calibration_sha256,
        "factor_cache_path": None if args.factor_cache is None else str(args.factor_cache.resolve()),
        "factor_cache_sha256": None if args.factor_cache is None else sha256_file(args.factor_cache),
        "factor_cache_reused": args.reuse_factor_cache,
        "requested_sequences": args.dataset_size,
        "independent_sequences": stats["independent_sequences"],
        "valid_tokens": stats["valid_output_samples"],
        "block_rows": args.block_rows,
        "block_count": block_count,
        "block_index": args.block_index,
        "factor_mode": args.factor_mode,
        "gram_rank": args.gram_rank,
        "bits": args.bits,
        "format": "qvq_v2b2_p32" if args.bits < 4 else "qvq_planar",
        "gsq_steps_requested": args.gsq_steps,
        "gsq_candidates": args.gsq_candidates,
        "gsq_coordinate_sweeps": args.gsq_coordinate_sweeps,
        "gsq_completed_steps": result.gsq_diagnostics["optimizer"]["completed_steps"],
        "gsq_optimizer_diagnostics": result.gsq_diagnostics["optimizer"],
        "gsq_changed_tiles": result.gsq_diagnostics["changed_tiles"],
        "gsq_fisher_before": result.gsq_diagnostics["before"],
        "gsq_fisher_after": result.gsq_diagnostics["after"],
        "oracle": oracle,
        "source_damped_oracle": damped_oracle,
        "no_gsq_oracle": baseline_oracle,
        "no_gsq_source_damped_oracle": baseline_damped_oracle,
        "source_damping": {"input": input_damping, "output": output_damping},
        "capture_seconds": capture_seconds,
        "quant_seconds": quant_seconds,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "status": "diagnostic_only_no_artifact",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
