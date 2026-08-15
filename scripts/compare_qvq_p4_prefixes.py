#!/usr/bin/env python3
"""Compare two packed QVQ prefixes against one dense teacher on locked rows."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.analyze_gptq_low_bit_grid import tensor_metrics
from scripts.compare_qvq_codecs_llama_qkvo import _WeightedMetricAccumulator
from scripts.validate_qvq_p4_live_prefix import _install_prefix_artifacts, _load_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--row-offset", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def _metric_value(metrics: dict[str, object], name: str) -> float:
    value = metrics[name]
    if name == "kl_forward":
        return float(value["mean"])
    if name in ("top5_overlap", "top10_overlap"):
        return float(value["mean"])
    return float(value)


@torch.inference_mode()
def _compare_locked_rows(
    teacher: torch.nn.Module,
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    rows: tuple[dict[str, torch.Tensor], ...],
) -> tuple[dict[str, object], dict[str, object]]:
    accumulators = (_WeightedMetricAccumulator(), _WeightedMetricAccumulator())
    for row_index, row in enumerate(rows):
        dense_logits = teacher(**row).logits[:, :-1].detach().float().cpu()
        dense_flat = dense_logits.reshape(-1, dense_logits.shape[-1])
        for model, accumulator in zip((baseline, candidate), accumulators, strict=True):
            student_logits = model(**row).logits[:, :-1].detach().float().cpu()
            student_flat = student_logits.reshape(-1, student_logits.shape[-1])
            accumulator.add(
                tensor_metrics(dense_flat, student_flat, normalize_distribution=False, include_top10=True),
                rows=dense_flat.shape[0],
            )
        print(f"locked rows complete: {row_index + 1}/{len(rows)}", flush=True)
    return accumulators[0].result(), accumulators[1].result()


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 2:
        raise ValueError("locked prefix comparison requires at least two decoder layers")
    if args.row_offset < 0 or args.rows < 1:
        raise ValueError("locked row offset/count must be nonnegative/positive")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_model() -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval().to(device)

    print("Loading dense teacher and two packed-prefix students", flush=True)
    teacher, baseline, candidate = load_model(), load_model(), load_model()
    baseline_manifests, baseline_modules = _install_prefix_artifacts(
        baseline,
        paths=args.baseline_prefix,
        model_path=args.model,
    )
    candidate_manifests, candidate_modules = _install_prefix_artifacts(
        candidate,
        paths=args.candidate_prefix,
        model_path=args.model,
    )
    if not set(baseline_modules).issubset(candidate_modules):
        raise ValueError("candidate prefix must contain every baseline module")

    locked_rows, row_stats = _load_rows(
        tokenizer,
        dataset=args.dataset,
        offset=args.row_offset,
        rows=args.rows,
        max_length=args.max_length,
        device=device,
    )
    started = time.perf_counter()
    baseline_metrics, candidate_metrics = _compare_locked_rows(teacher, baseline, candidate, locked_rows)
    elapsed = time.perf_counter() - started
    names = ("kl_forward", "top1_agreement", "top5_overlap", "top10_overlap")
    deltas = {name: _metric_value(candidate_metrics, name) - _metric_value(baseline_metrics, name) for name in names}
    report = {
        "settings": {
            "model": str(args.model.resolve()),
            "dataset": str(args.dataset.resolve()),
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "baseline_prefix": [str(path.resolve()) for path in args.baseline_prefix],
            "candidate_prefix": [str(path.resolve()) for path in args.candidate_prefix],
            "baseline_manifest_count": len(baseline_manifests),
            "candidate_manifest_count": len(candidate_manifests),
            "baseline_modules": sorted(baseline_modules),
            "candidate_modules": sorted(candidate_modules),
            "rows": row_stats,
            "execution": "independent full rows; batch=1; no concatenation",
        },
        "evaluation_seconds": elapsed,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_baseline": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"locked comparison complete in {elapsed:.2f}s: "
        f"KL {deltas['kl_forward']:+.8g}, Top-1 {deltas['top1_agreement']:+.4%}, "
        f"Top-5 {deltas['top5_overlap']:+.4%}, Top-10 {deltas['top10_overlap']:+.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
