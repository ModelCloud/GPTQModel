#!/usr/bin/env python3
"""Run a paired ARC-Challenge gate for ordinary and P4-refined QVQ prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.validate_qvq_p4_live_prefix import _install_prefix_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--baseline-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _prompt_ids(tokenizer, question: str, *, apply_chat_template: bool) -> list[int]:
    prompt = f"Question: {question}\nAnswer:"
    if apply_chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError("chat template did not render a non-empty prompt")
        prompt = rendered
    return [int(token) for token in tokenizer(prompt, add_special_tokens=False).input_ids]


def _choice_ids(tokenizer, choice: str) -> list[int]:
    continuation = choice if choice[:1].isspace() else f" {choice}"
    tokens = [int(token) for token in tokenizer(continuation, add_special_tokens=False).input_ids]
    if not tokens:
        raise ValueError("ARC choice continuation tokenized to an empty sequence")
    return tokens


@torch.inference_mode()
def _choice_loglikelihoods(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    choices: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    sequences = [prompt_ids + choice for choice in choices]
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, : len(sequence)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    raw_scores = []
    normalized_scores = []
    start = len(prompt_ids) - 1
    for row, choice in enumerate(choices):
        token_logits = logits[row, start : start + len(choice)]
        targets = torch.tensor(choice, dtype=torch.long, device=device)
        score = torch.log_softmax(token_logits, dim=-1).gather(1, targets[:, None]).sum().item()
        raw_scores.append(float(score))
        normalized_scores.append(float(score) / len(choice))
    return raw_scores, normalized_scores


def _prediction(scores: list[float]) -> int:
    return max(range(len(scores)), key=scores.__getitem__)


def _margin(scores: list[float], gold: int) -> float:
    return float(scores[gold] - max(score for index, score in enumerate(scores) if index != gold))


def _arm_summary(samples: list[dict[str, object]], arm: str) -> dict[str, object]:
    count = len(samples)
    raw_correct = sum(sample[arm]["raw_prediction"] == sample["gold_index"] for sample in samples)
    norm_correct = sum(sample[arm]["normalized_prediction"] == sample["gold_index"] for sample in samples)
    return {
        "rows": count,
        "accuracy": raw_correct / count,
        "accuracy_normalized": norm_correct / count,
        "raw_correct": raw_correct,
        "normalized_correct": norm_correct,
        "mean_gold_margin": sum(float(sample[arm]["raw_gold_margin"]) for sample in samples) / count,
        "mean_gold_margin_normalized": (
            sum(float(sample[arm]["normalized_gold_margin"]) for sample in samples) / count
        ),
    }


def _paired_summary(samples: list[dict[str, object]], left: str, right: str) -> dict[str, object]:
    result = {}
    for prediction_key, label in (("raw_prediction", "raw"), ("normalized_prediction", "normalized")):
        left_to_right = 0
        right_to_left = 0
        changed = 0
        for sample in samples:
            gold = sample["gold_index"]
            left_prediction = sample[left][prediction_key]
            right_prediction = sample[right][prediction_key]
            changed += left_prediction != right_prediction
            left_to_right += left_prediction != gold and right_prediction == gold
            right_to_left += left_prediction == gold and right_prediction != gold
        result[label] = {
            "prediction_changes": changed,
            "wrong_to_correct": left_to_right,
            "correct_to_wrong": right_to_left,
            "net_correct": left_to_right - right_to_left,
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 1 or args.row_offset < 0 or args.max_rows < 1 or args.report_every < 1:
        raise ValueError("layers/max-rows/report-every must be positive and row-offset must be nonnegative")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    if args.layers > source_layers:
        raise ValueError("requested layers exceed the source model")
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

    print("Loading dense, ordinary-YAQA, and P4-refined full models", flush=True)
    models = {"dense": load_model(), "baseline": load_model(), "candidate": load_model()}
    baseline_manifests, baseline_modules = _install_prefix_artifacts(
        models["baseline"], paths=args.baseline_prefix, model_path=args.model
    )
    candidate_manifests, candidate_modules = _install_prefix_artifacts(
        models["candidate"], paths=args.candidate_prefix, model_path=args.model
    )
    if set(baseline_modules) != set(candidate_modules):
        raise ValueError("paired ARC prefixes must quantize the exact same module set")

    dataset = Dataset.from_file(str(args.dataset_arrow))
    stop = min(args.row_offset + args.max_rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_offset, stop)]
    if not documents:
        raise ValueError("ARC row selection is empty")

    token_hash = hashlib.sha256()
    samples = []
    started = time.perf_counter()
    for relative_index, document in enumerate(documents):
        prompt_ids = _prompt_ids(tokenizer, document["question"], apply_chat_template=args.apply_chat_template)
        choice_texts = list(document["choices"]["text"])
        choices = [_choice_ids(tokenizer, choice) for choice in choice_texts]
        labels = list(document["choices"]["label"])
        gold = labels.index(document["answerKey"])
        token_hash.update(json.dumps([prompt_ids, choices], separators=(",", ":")).encode())
        sample = {
            "index": args.row_offset + relative_index,
            "id": document["id"],
            "gold_index": gold,
            "choice_labels": labels,
        }
        for arm, model in models.items():
            raw, normalized = _choice_loglikelihoods(
                model,
                prompt_ids=prompt_ids,
                choices=choices,
                pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            sample[arm] = {
                "raw_prediction": _prediction(raw),
                "normalized_prediction": _prediction(normalized),
                "raw_gold_margin": _margin(raw, gold),
                "normalized_gold_margin": _margin(normalized, gold),
                "choice_logprobs": raw,
                "choice_logprobs_normalized": normalized,
            }
        samples.append(sample)
        if (relative_index + 1) % args.report_every == 0 or relative_index + 1 == len(documents):
            print(f"ARC rows complete: {relative_index + 1}/{len(documents)}", flush=True)

    elapsed = time.perf_counter() - started
    summaries = {arm: _arm_summary(samples, arm) for arm in models}
    report = {
        "settings": {
            "model": str(args.model.resolve()),
            "dataset_arrow": str(args.dataset_arrow.resolve()),
            "dataset_rows": len(dataset),
            "row_start": args.row_offset,
            "row_end_exclusive": stop,
            "rows": len(documents),
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "apply_chat_template": args.apply_chat_template,
            "prompt": "Question: {question}\\nAnswer:",
            "continuation": "leading-space choice text",
            "batching": "one question per step; all choices in one batch",
            "token_contract_sha256": token_hash.hexdigest(),
            "baseline_prefix": [str(path.resolve()) for path in args.baseline_prefix],
            "candidate_prefix": [str(path.resolve()) for path in args.candidate_prefix],
            "baseline_manifest_count": len(baseline_manifests),
            "candidate_manifest_count": len(candidate_manifests),
            "quantized_modules": sorted(baseline_modules),
        },
        "evaluation_seconds": elapsed,
        "summary": summaries,
        "paired": {
            "baseline_to_candidate": _paired_summary(samples, "baseline", "candidate"),
            "dense_to_baseline": _paired_summary(samples, "dense", "baseline"),
            "dense_to_candidate": _paired_summary(samples, "dense", "candidate"),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paired = report["paired"]["baseline_to_candidate"]
    print(
        f"ARC complete in {elapsed:.2f}s: baseline={summaries['baseline']['accuracy']:.4f}/"
        f"{summaries['baseline']['accuracy_normalized']:.4f} candidate={summaries['candidate']['accuracy']:.4f}/"
        f"{summaries['candidate']['accuracy_normalized']:.4f} flips={paired}",
        flush=True,
    )


if __name__ == "__main__":
    main()
