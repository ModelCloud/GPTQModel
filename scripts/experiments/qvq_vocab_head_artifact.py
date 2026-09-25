#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Build a bounded, block-quantized QVQ lm_head delta from cached YAQA factors.

The output is an offline delta artifact. Model serving needs a block-head
loader; this script does not claim a ZML checkpoint or downstream score.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-parquet", type=Path, required=True)
    parser.add_argument("--factor-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--gram-rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-rows", type=int, default=2048)
    parser.add_argument("--bits", type=float, required=True, choices=[2.5, 3, 3.5, 4])
    parser.add_argument("--gsq-steps", type=int, default=640)
    parser.add_argument("--gsq-candidates", type=int, default=33)
    parser.add_argument("--gsq-coordinate-sweeps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-blocks", type=int)
    return parser.parse_args()


def _oracle_contribution(error, output_factor, input_hessian, input_damping, *, dtype):
    import torch

    e = error.to(dtype)
    s = output_factor.to(dtype)
    h = input_hessian.to(dtype)
    z = e.T @ s
    hd = h.clone()
    hd.diagonal().add_(input_damping)
    diagonal_penalty = torch.einsum("oi,ij,oj->", e, hd, e)
    return z, diagonal_penalty


def _oracle(z, penalty, input_hessian, input_damping, output_damping, *, dtype):
    import torch

    h = input_hessian.to(dtype)
    undamped = torch.einsum("ir,ij,jr->", z, h, z)
    hd = h.clone()
    hd.diagonal().add_(input_damping)
    damped = torch.einsum("ir,ij,jr->", z, hd, z) + output_damping * penalty
    return float(undamped.item()), float(damped.item())


def main() -> None:
    args = parse_args()
    import torch
    from safetensors.torch import save_file

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization.config import GSQConfig
    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.utils.model import untie_word_embeddings
    from scripts.experiments.qvq_vocab_block_probe import (
        load_shared_factor_cache,
        sha256_file,
    )

    if args.block_rows < 16 or args.block_rows % 16:
        raise ValueError("block rows must be a positive multiple of 16")
    if args.gsq_steps < 1 or args.gsq_candidates < 2 or args.gsq_coordinate_sweeps < 0:
        raise ValueError("invalid GSQ schedule")
    if args.limit_blocks is not None and args.limit_blocks < 1:
        raise ValueError("limit-blocks must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite endpoint delta: {args.output_dir}")
    started = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    model = GPTQModel.load(
        str(args.model_path), device=DEVICE.CUDA, backend=BACKEND.AUTO,
        attn_implementation="eager",
    )
    model.model = untie_word_embeddings(model.model)
    head = model.get_output_embeddings()
    device = head.weight.device
    block_count = (head.out_features + args.block_rows - 1) // args.block_rows
    block_limit = min(block_count, args.limit_blocks or block_count)
    cache_identity = {
        "schema": "qvq.yaqa.shared-head-factor.v2",
        "model_path": str(args.model_path.resolve()),
        "calibration_sha256": sha256_file(args.calibration_parquet),
        "requested_sequences": str(args.dataset_size),
        "gram_rank": str(args.gram_rank),
        "batch_size": str(args.batch_size),
        "seed": str(args.seed),
        "input_features": str(head.in_features),
        "output_features": str(head.out_features),
    }
    inputs, outputs, stats = load_shared_factor_cache(args.factor_cache, cache_identity)
    h = inputs["lm_head"].materialize(device=device)
    s = outputs["lm_head"].factor(device=device)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    input_damping = max(float(h.diagonal().abs().mean()) * 0.05, torch.finfo(torch.float32).eps)
    output_damping = max(float(s.square().sum(1).mean()) * 0.05, torch.finfo(torch.float32).eps)
    z = {arm: {label: torch.zeros((head.in_features, args.gram_rank), device=device, dtype=dtype)
               for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64))}
         for arm in ("baseline", "candidate")}
    penalties = {arm: {label: torch.zeros((), device=device, dtype=dtype)
                       for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64))}
                 for arm in ("baseline", "candidate")}
    tensors = {"baseline": {}, "candidate": {}}
    block_records = []
    for index in range(block_limit):
        start = index * args.block_rows
        stop = min(start + args.block_rows, head.out_features)
        sf = s[start:stop]
        out_h = sf @ sf.T
        out_h = (out_h + out_h.T) * 0.5
        weight = head.weight.detach()[start:stop]
        common = {
            "bits": args.bits, "output_hessian": out_h, "seed": args.seed,
            "damp_percent": 0.05, "rounding": "yaqa",
            "v2b2_p32": args.bits < 4, "bank_count": 2 if args.bits < 4 else 1,
        }
        record = {"index": index, "start": start, "stop": stop}
        for arm, gsq in (
            ("baseline", None),
            ("candidate", GSQConfig(enabled=True, steps=args.gsq_steps,
                                    candidates=args.gsq_candidates, seed=args.seed,
                                    qvq_coordinate_sweeps=args.gsq_coordinate_sweeps)),
        ):
            torch.manual_seed(args.seed)
            begun = time.monotonic()
            result = quantize_qvq_linear(weight, h, gsq=gsq, **common)
            torch.cuda.synchronize()
            record[f"{arm}_seconds"] = time.monotonic() - begun
            if arm == "candidate":
                record["gsq"] = result.gsq_diagnostics
            error = result.weight.float() - weight.float()
            for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
                zi, penalty = _oracle_contribution(error, sf, h, input_damping, dtype=dtype)
                z[arm][label].add_(zi)
                penalties[arm][label].add_(penalty)
            for name, value in result.serialized_tensors().items():
                tensors[arm][f"lm_head.blocks.{index}.{name}"] = value.detach().to("cpu").contiguous()
            del result
        block_records.append(record)
        print(f"block={index + 1}/{block_limit} changed={record['gsq']['changed_tiles']}", flush=True)
    scores = {}
    for arm in ("baseline", "candidate"):
        scores[arm] = {}
        for label, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            undamped, damped = _oracle(
                z[arm][label], penalties[arm][label], h,
                input_damping, output_damping, dtype=dtype,
            )
            scores[arm][label] = {"undamped": undamped, "damped": damped}
    args.output_dir.mkdir(parents=True)
    metadata = {
        "schema": "qvq.vocab-head-delta.v1", "model_path": str(args.model_path.resolve()),
        "bits": str(args.bits), "block_rows": str(args.block_rows),
        "head_rows": str(head.out_features), "head_columns": str(head.in_features),
        "block_count": str(block_limit), "complete_head": str(block_limit == block_count).lower(),
        "factor_cache_sha256": sha256_file(args.factor_cache),
    }
    for arm in ("baseline", "candidate"):
        save_file(tensors[arm], str(args.output_dir / f"{arm}.safetensors"), metadata=metadata)
    manifest = {
        **metadata,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_sha256": cache_identity["calibration_sha256"],
        "independent_sequences": stats["independent_sequences"],
        "valid_tokens": stats["valid_output_samples"],
        "gram_rank": args.gram_rank, "seed": args.seed,
        "gsq_steps": args.gsq_steps, "gsq_candidates": args.gsq_candidates,
        "gsq_coordinate_sweeps": args.gsq_coordinate_sweeps,
        "input_damping": input_damping, "output_damping": output_damping,
        "covered_rows_oracle": scores, "blocks": block_records,
        "status": "offline_delta_no_serving_claim",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"blocks": block_limit, "covered_rows_oracle": scores}, indent=2))


if __name__ == "__main__":
    main()
