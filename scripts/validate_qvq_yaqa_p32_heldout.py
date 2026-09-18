#!/usr/bin/env python3
"""Compare two P32 payloads on disjoint dense-model projection activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    reconstruct_qvq_inner_weight,
    rht_reconstruct_weight,
)
from scripts.validate_qvq_yaqa_p32_dual_oracle import _load_module_payload


def _projection_names(layer: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer}"
    return tuple(
        f"{prefix}.{suffix}"
        for suffix in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    )


def _reconstruct(checkpoint: Path, module: str, bits: float, device: torch.device) -> torch.Tensor:
    payload = _load_module_payload(checkpoint, module, device)
    in_features = payload["SU"].numel()
    out_features = payload["SV"].numel()
    inner = reconstruct_qvq_inner_weight(
        payload["trellis"],
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        bank_ids=payload["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=payload["bank_alt_id"],
    )
    return rht_reconstruct_weight(inner, payload["SU"], payload["SV"]).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bits", type=float, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--text-column", default="question")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disjointness-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.batch_size < 1 or args.row_start < 0:
        parser.error("rows and batch-size must be positive; row-start must be nonnegative")
    manifest_bytes = args.disjointness_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "pass":
        raise ValueError("Held-out projection validation requires a passing disjointness manifest.")
    evaluation_splits = manifest.get("evaluation_splits", ())
    if "gsm8k_platinum" not in evaluation_splits:
        raise ValueError("Disjointness manifest does not bind the GSM8K Platinum evaluation split.")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    modules = _projection_names(args.layer)
    weights = {
        arm: {name: _reconstruct(path, name, args.bits, device) for name in modules}
        for arm, path in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    named_modules = dict(model.named_modules())
    missing = set(modules) - set(named_modules)
    if missing:
        raise ValueError(f"Dense model is missing projections: {sorted(missing)}")

    accumulators = {
        arm: {
            name: {
                "squared_error": torch.zeros((), dtype=torch.float64, device=device),
                "dense_energy": torch.zeros((), dtype=torch.float64, device=device),
                "elements": 0,
            }
            for name in modules
        }
        for arm in weights
    }
    active_mask: torch.Tensor | None = None

    def hook(name: str):
        def measure(_module, inputs, output):
            nonlocal active_mask
            if active_mask is None:
                raise RuntimeError("Held-out activation mask is unavailable.")
            source = inputs[0][active_mask].float()
            target = output[active_mask].float()
            for arm in weights:
                predicted = F.linear(source, weights[arm][name])
                error = predicted - target
                stats = accumulators[arm][name]
                stats["squared_error"].add_(error.square().sum(dtype=torch.float64))
                stats["dense_energy"].add_(target.square().sum(dtype=torch.float64))
                stats["elements"] += error.numel()
        return measure

    handles = [named_modules[name].register_forward_hook(hook(name)) for name in modules]
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    stop = args.row_start + args.rows
    if stop > len(dataset):
        raise ValueError(f"Requested rows [{args.row_start}, {stop}) from dataset of length {len(dataset)}.")
    texts = [str(dataset[index][args.text_column]) for index in range(args.row_start, stop)]
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(
                    texts[start : start + args.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                active_mask = encoded["attention_mask"].bool()
                model(**encoded, use_cache=False)
    finally:
        active_mask = None
        for handle in handles:
            handle.remove()

    report_modules = {}
    totals = {arm: {"squared_error": 0.0, "dense_energy": 0.0, "elements": 0} for arm in weights}
    for name in modules:
        row = {}
        for arm in weights:
            raw = accumulators[arm][name]
            squared_error = float(raw["squared_error"].item())
            dense_energy = float(raw["dense_energy"].item())
            elements = int(raw["elements"])
            row[arm] = {
                "mse": squared_error / elements,
                "relative_squared_error": squared_error / dense_energy,
                "elements": elements,
            }
            totals[arm]["squared_error"] += squared_error
            totals[arm]["dense_energy"] += dense_energy
            totals[arm]["elements"] += elements
        row["candidate_vs_baseline_relative_mse"] = row["candidate"]["mse"] / row["baseline"]["mse"] - 1.0
        report_modules[name] = row
    aggregate = {}
    for arm, raw in totals.items():
        aggregate[arm] = {
            "mse": raw["squared_error"] / raw["elements"],
            "relative_squared_error": raw["squared_error"] / raw["dense_energy"],
            "elements": raw["elements"],
        }
    aggregate["candidate_vs_baseline_relative_mse"] = (
        aggregate["candidate"]["mse"] / aggregate["baseline"]["mse"] - 1.0
    )
    report = {
        "schema": "qvq.yaqa-p32-heldout-projection.v1",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "bits": args.bits,
        "layer": args.layer,
        "evaluation": {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "row_start": args.row_start,
            "rows": args.rows,
            "text_column": args.text_column,
            "disjointness_manifest": str(args.disjointness_manifest.resolve()),
            "disjointness_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "modules": report_modules,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
