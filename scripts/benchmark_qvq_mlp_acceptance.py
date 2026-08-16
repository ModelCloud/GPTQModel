# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A/B the full-model and exact suffix MLP acceptance evaluators on a real checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from analyze_gptq_low_bit_grid import load_nm_evaluation_batch
from compare_qvq_codecs_llama_qkvo import (
    _build_mlp_acceptance_evaluator,
    _unpadded_evaluation_rows,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("full", "suffix"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--row-offset", type=int, default=1536)
    parser.add_argument("--proposals-per-layer", type=int, default=7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.num_hidden_layers = args.layers
    dense = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    candidate = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded, dataset_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.row_offset,
        rows=args.rows,
        max_length=None,
    )
    rows = _unpadded_evaluation_rows(encoded, device)
    originals = tuple(
        tuple(projection.weight.detach().clone() for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
        for layer in candidate.model.layers
    )
    evaluator = _build_mlp_acceptance_evaluator(
        dense,
        candidate,
        rows,
        progress_prefix=f"benchmark/{args.execution}",
        execution=args.execution,
        kl_regression_limit=0.05,
        topn_regression_limit=0.05,
    )
    reports = [evaluator("baseline")]
    evaluator.set_acceptance_baseline(reports[0])
    subsets = ((0,), (1,), (0, 1), (2,), (0, 2), (1, 2), (0, 1, 2))
    for layer_index, layer in enumerate(candidate.model.layers):
        evaluator.prepare_layer(layer_index)
        projections = (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
        for proposal_index in range(args.proposals_per_layer):
            selected = subsets[proposal_index % len(subsets)]
            with torch.no_grad():
                for projection_index in selected:
                    projections[projection_index].weight.add_(0.002 * (proposal_index + 1))
            reports.append(evaluator(f"layer_{layer_index}_proposal_{proposal_index}"))
            with torch.no_grad():
                for projection, original in zip(projections, originals[layer_index], strict=True):
                    projection.weight.copy_(original)
    telemetry = evaluator.telemetry()
    payload = {
        "execution": args.execution,
        "model": str(args.model),
        "device": str(device),
        "layers": args.layers,
        "rows": args.rows,
        "sequence_lengths": [int(row["input_ids"].shape[1]) for row in rows],
        "proposals_per_layer": args.proposals_per_layer,
        "dataset": dataset_stats,
        "telemetry": telemetry,
        "metric_fingerprints": [
            {
                "kl": report["kl_forward"]["mean"],
                "top1": report["top1_agreement"],
                "top5": report["top5_overlap"]["mean"],
                "top10": report["top10_overlap"]["mean"],
            }
            for report in reports
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(telemetry, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
