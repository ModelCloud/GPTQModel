#!/usr/bin/env python3
"""Run a paired GSM8K task gate for ordinary and P4-refined QVQ prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.validate_qvq_p4_live_prefix import _install_prefix_artifacts

_STRICT_ANSWER = re.compile(r"The answer is (\-?[$0-9.,]+)\.?", re.IGNORECASE)
_FLEXIBLE_ANSWER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, default=Path("tests/tasks/gsm8k/gsm8k-cot.yaml"))
    parser.add_argument("--baseline-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--apply-chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_task_contract(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    samples = document["fewshot_config"]["samples"]
    fewshots = [{"question": str(sample["question"]), "target": str(sample["target"])} for sample in samples]
    until = tuple(str(value) for value in document["generation_kwargs"]["until"])
    if int(document["num_fewshot"]) != len(fewshots):
        raise ValueError("GSM8K task num_fewshot does not match its fixed sample count")
    return fewshots, until


def _task_prompt(question: str, fewshots: list[dict[str, str]]) -> str:
    demonstrations = [f"Q: {sample['question']}\n\nA: {sample['target']}" for sample in fewshots]
    demonstrations.append(f"Q: {question}\n\nA:")
    return "\n\n".join(demonstrations)


def _prompt_ids(tokenizer, prompt: str, *, apply_chat_template: bool) -> list[int]:
    if apply_chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError("chat template did not render a non-empty GSM8K prompt")
        prompt = rendered
    return [int(token) for token in tokenizer(prompt, add_special_tokens=False).input_ids]


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(",", "").replace("$", "").strip().rstrip(".")
    return normalized or None


def _extract_answers(text: str) -> tuple[str | None, str | None]:
    strict_matches = _STRICT_ANSWER.findall(text)
    flexible_matches = _FLEXIBLE_ANSWER.findall(text)
    strict = _normalize_answer(strict_matches[-1]) if strict_matches else None
    flexible_values = [left or right for left, right in flexible_matches]
    flexible = _normalize_answer(flexible_values[-1]) if flexible_values else None
    return strict, flexible


def _gold_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError("GSM8K target does not contain the expected #### delimiter")
    gold = _normalize_answer(answer.rsplit("####", 1)[1])
    if gold is None:
        raise ValueError("GSM8K target has an empty numeric answer")
    return gold


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    tokenizer,
    *,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
    until: tuple[str, ...],
) -> dict[str, object]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        use_cache=True,
    )
    elapsed = time.perf_counter() - started
    continuation = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    for stop in until:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    strict, flexible = _extract_answers(text)
    return {
        "token_ids": [int(token) for token in continuation],
        "text": text,
        "strict_answer": strict,
        "flexible_answer": flexible,
        "seconds": elapsed,
    }


def _arm_summary(samples: list[dict[str, object]], arm: str) -> dict[str, object]:
    rows = len(samples)
    strict_correct = sum(sample[arm]["strict_answer"] == sample["gold_answer"] for sample in samples)
    flexible_correct = sum(sample[arm]["flexible_answer"] == sample["gold_answer"] for sample in samples)
    return {
        "rows": rows,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / rows,
        "flexible_correct": flexible_correct,
        "flexible_accuracy": flexible_correct / rows,
        "strict_invalid": sum(sample[arm]["strict_answer"] is None for sample in samples),
        "flexible_invalid": sum(sample[arm]["flexible_answer"] is None for sample in samples),
        "generation_seconds": sum(float(sample[arm]["seconds"]) for sample in samples),
    }


def _paired_summary(samples: list[dict[str, object]], left: str, right: str) -> dict[str, object]:
    result = {}
    for answer_key, label in (("strict_answer", "strict"), ("flexible_answer", "flexible")):
        changed = wrong_to_correct = correct_to_wrong = 0
        for sample in samples:
            gold = sample["gold_answer"]
            left_answer = sample[left][answer_key]
            right_answer = sample[right][answer_key]
            changed += left_answer != right_answer
            wrong_to_correct += left_answer != gold and right_answer == gold
            correct_to_wrong += left_answer == gold and right_answer != gold
        result[label] = {
            "answer_changes": changed,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct": wrong_to_correct - correct_to_wrong,
        }
    return result


def _progress_summary(samples: list[dict[str, object]], arms: tuple[str, ...]) -> str:
    fields = []
    for arm in arms:
        correct = sum(sample[arm]["flexible_answer"] == sample["gold_answer"] for sample in samples)
        invalid = sum(sample[arm]["flexible_answer"] is None for sample in samples)
        fields.append(
            f"{arm}={correct}/{len(samples)}"
            f"(invalid={invalid})"
        )
    return " ".join(fields)


def main() -> None:
    args = _parser().parse_args()
    if (
        args.layers < 1
        or args.row_offset < 0
        or args.max_rows < 1
        or args.max_new_tokens < 1
        or args.report_every < 1
    ):
        raise ValueError("layers/max-rows/max-new-tokens/report-every must be positive and row-offset nonnegative")
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
        raise ValueError("paired GSM8K prefixes must quantize the exact same module set")

    fewshots, until = _load_task_contract(args.task_config)
    dataset = Dataset.from_file(str(args.dataset_arrow))
    stop = min(args.row_offset + args.max_rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_offset, stop)]
    if not documents:
        raise ValueError("GSM8K row selection is empty")

    token_hash = hashlib.sha256()
    samples = []
    started = time.perf_counter()
    for relative_index, document in enumerate(documents):
        prompt = _task_prompt(str(document["question"]), fewshots)
        prompt_ids = _prompt_ids(tokenizer, prompt, apply_chat_template=args.apply_chat_template)
        token_hash.update(json.dumps(prompt_ids, separators=(",", ":")).encode())
        sample = {
            "index": args.row_offset + relative_index,
            "question": str(document["question"]),
            "gold_answer": _gold_answer(str(document["answer"])),
        }
        for arm, model in models.items():
            sample[arm] = _generate(
                model,
                tokenizer,
                prompt_ids=prompt_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
                until=until,
            )
        samples.append(sample)
        if (relative_index + 1) % args.report_every == 0 or relative_index + 1 == len(documents):
            partial = _paired_summary(samples, "baseline", "candidate")
            progress = _progress_summary(samples, tuple(models))
            print(
                f"GSM8K rows complete: {relative_index + 1}/{len(documents)} {progress} paired={partial}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    summaries = {arm: _arm_summary(samples, arm) for arm in models}
    report = {
        "settings": {
            "model": str(args.model.resolve()),
            "dataset_arrow": str(args.dataset_arrow.resolve()),
            "task_config": str(args.task_config.resolve()),
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
            "fewshot_count": len(fewshots),
            "max_new_tokens": args.max_new_tokens,
            "until": list(until),
            "batching": "one independently generated row per model call",
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
    print(
        f"GSM8K complete in {elapsed:.2f}s: dense={summaries['dense']['flexible_accuracy']:.4f} "
        f"baseline={summaries['baseline']['flexible_accuracy']:.4f} "
        f"candidate={summaries['candidate']['flexible_accuracy']:.4f} "
        f"paired={report['paired']['baseline_to_candidate']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
