#!/usr/bin/env python3
"""Screen offline V4 second-pair mappings on real model weight vectors.

This is a pruning experiment only.  It does not write a checkpoint and never
changes the production ``state ^ 0xA5A5`` decoder mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel.quantization.qvq_v4_candidates import (
    V4Candidate,
    score_v4_candidate,
    select_v4_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vectors", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.vectors < 1:
        raise ValueError("--vectors must be positive")
    with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.tensor).to(dtype=torch.float32)
    flat = weight.flatten()
    usable = min(args.vectors * 4, flat.numel() - flat.numel() % 4)
    target = flat[:usable].reshape(-1, 4)
    target = target / target.square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    target = target.to(args.device)
    candidates = [
        V4Candidate("canonical", 0xA5A5, 1.00),
        V4Candidate("mask-5a5a", 0x5A5A, 1.00),
        V4Candidate("mask-3c3c", 0x3C3C, 1.00),
        V4Candidate("mask-c3c3", 0xC3C3, 1.00),
        V4Candidate("mask-9696", 0x9696, 1.00),
        V4Candidate("canonical-scale-095", 0xA5A5, 0.95),
        V4Candidate("canonical-scale-105", 0xA5A5, 1.05),
    ]
    scores = [score_v4_candidate(target, candidate, chunk_size=256) for candidate in candidates]
    selected = select_v4_candidate(scores, baseline=candidates[0])
    rows = [
        {
            "candidate": score.candidate.name,
            "xor_mask": f"0x{score.candidate.xor_mask:04x}",
            "scale_multiplier": score.candidate.scale_multiplier,
            "mse": score.mse,
            "p95": score.p95,
            "represented_orthants": score.represented_orthants,
        }
        for score in scores
    ]
    result = {"model": str(args.model), "tensor": args.tensor, "vectors": target.shape[0], "selected_local": selected.name, "scores": rows}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
