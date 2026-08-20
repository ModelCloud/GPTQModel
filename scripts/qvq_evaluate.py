#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic post-quantization evaluation for saved QVQ checkpoints.

``diagnostics`` compares final logits with the dense source on an explicitly
held-out dataset slice. ``tasks`` runs standard Evalution suites. This script
never prepares or quantizes a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402

if __package__:
    from scripts.qvq_quantize import DatasetSlice, load_dataset_slice
else:
    from qvq_quantize import DatasetSlice, load_dataset_slice


MMLU_HISTORY_SUBSETS = (
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
)
TASKS = {
    "arc_challenge": ("arc_challenge", True, {}),
    "gsm8k_platinum_cot": ("gsm8k_platinum_cot", True, {}),
    "mmlu_stem": ("mmlu_stem", False, {}),
    "mmlu_history": ("mmlu", False, {"subsets": MMLU_HISTORY_SUBSETS}),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnostics = subparsers.add_parser("diagnostics", help="Compare final logits with the dense model.")
    diagnostics.add_argument("--dense-model", required=True)
    diagnostics.add_argument("--checkpoint", type=Path, required=True)
    diagnostics.add_argument("--dataset", required=True)
    diagnostics.add_argument("--dataset-config")
    diagnostics.add_argument("--dataset-split", default="train")
    diagnostics.add_argument("--row-start", type=int, required=True)
    diagnostics.add_argument("--rows", type=int, default=512)
    diagnostics.add_argument("--text-column")
    diagnostics.add_argument("--max-length", type=int, help="Omit to preserve complete source rows.")
    diagnostics.add_argument("--device", default="cuda:0")
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--divergence-rows", type=int, default=300)
    diagnostics.add_argument("--divergence-tokens", type=int, default=32)
    diagnostics.add_argument("--include-topn", action="store_true", help="Also report Top-5/Top-10 overlap.")
    diagnostics.add_argument("--trust-remote-code", action="store_true")
    diagnostics.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    diagnostics.add_argument("--allow-calibration-overlap", action="store_true")

    tasks = subparsers.add_parser("tasks", help="Run Evalution tasks on the quantized checkpoint.")
    tasks.add_argument("--checkpoint", type=Path, required=True)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--batch-size", type=int, default=16)
    tasks.add_argument("--backend", choices=(BACKEND.QVQ.value, BACKEND.EXL3_EXLLAMA_V3.value), default="qvq")
    tasks.add_argument("--task", action="append", choices=tuple(TASKS), help="Repeat to select tasks.")
    return parser


def _canonical_source(source: str) -> str:
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else source


def _ranges_overlap(left_start: int, left_rows: int, right_start: int, right_rows: int) -> bool:
    return max(left_start, right_start) < min(left_start + left_rows, right_start + right_rows)


def validate_evaluation_is_held_out(
    checkpoint: Path,
    evaluation: DatasetSlice,
    *,
    allow_overlap: bool,
) -> None:
    """Reject overlap with any preparation stream recorded by the quantizer."""

    if allow_overlap:
        return
    manifest = checkpoint / "qvq_quantize_run.json"
    if not manifest.is_file():
        return
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for name, raw in payload.get("datasets", {}).items():
        if not isinstance(raw, dict):
            continue
        required = {"source", "split", "row_start", "rows"}
        if not required.issubset(raw):
            continue
        same_identity = (
            _canonical_source(str(raw["source"])) == evaluation.identity[0]
            and raw.get("config") == evaluation.config
            and str(raw["split"]) == evaluation.split
        )
        if same_identity and _ranges_overlap(
            int(raw["row_start"]), int(raw["rows"]), evaluation.row_start, evaluation.rows
        ):
            raise ValueError(
                f"Evaluation [{evaluation.row_start}, {evaluation.row_stop}) overlaps recorded `{name}` "
                f"preparation rows [{raw['row_start']}, {int(raw['row_start']) + int(raw['rows'])})"
            )


def _row_text(row: dict[str, Any], tokenizer, text_column: str | None) -> str:
    if text_column is not None:
        if text_column not in row:
            raise KeyError(f"Dataset has no requested text column `{text_column}`")
        value = row[text_column]
    else:
        value = next(
            (row[name] for name in ("text", "prompt", "content", "messages") if name in row),
            None,
        )
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=False)
    raise ValueError("Every evaluation row must provide nonempty text or a supported messages list")


def _model_logits(model, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    # Call the public causal-LM wrapper. Unwrapping ``.model`` bypasses the LM
    # head on common Transformers architectures and returns hidden states.
    output = model(**encoded, use_cache=False)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Model forward did not return tensor logits (output type: {type(output).__name__})")
    return logits


@torch.inference_mode()
def _diagnostics(args: argparse.Namespace) -> int:
    if args.row_start < 0 or args.rows < 1:
        raise ValueError("row start must be nonnegative and rows must be positive")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("max length must be positive")
    if args.divergence_rows < 1 or args.divergence_tokens < 1:
        raise ValueError("divergence sizes must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    evaluation_spec = DatasetSlice(
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.row_start,
        args.rows,
    )
    validate_evaluation_is_held_out(
        checkpoint,
        evaluation_spec,
        allow_overlap=args.allow_calibration_overlap,
    )
    dataset = load_dataset_slice(evaluation_spec)
    tokenizer = AutoTokenizer.from_pretrained(
        args.dense_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.perf_counter()
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).eval()
    dense_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    quantized = GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    quantized_load_seconds = time.perf_counter() - started

    kl_sum = torch.zeros((), dtype=torch.float64, device=args.device)
    top1_sum = torch.zeros((), dtype=torch.float64, device=args.device)
    top5_sum = torch.zeros((), dtype=torch.float64, device=args.device)
    top10_sum = torch.zeros((), dtype=torch.float64, device=args.device)
    token_total = 0
    divergence_sequences = 0
    divergence_matching_tokens = 0
    divergence_total_tokens = 0
    divergence_exact_sequences = 0
    divergence_first_sum = 0.0
    dense_forward_seconds = 0.0
    quantized_forward_seconds = 0.0

    for row_index, row in enumerate(dataset):
        text = _row_text(dict(row), tokenizer, args.text_column)
        tokenize_kwargs: dict[str, Any] = {"return_tensors": "pt", "truncation": args.max_length is not None}
        if args.max_length is not None:
            tokenize_kwargs["max_length"] = args.max_length
        encoded = {name: value.to(args.device) for name, value in tokenizer(text, **tokenize_kwargs).items()}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])

        started = time.perf_counter()
        dense_logits = _model_logits(dense, encoded)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        dense_forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        quantized_logits = _model_logits(quantized, encoded)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        quantized_forward_seconds += time.perf_counter() - started

        keep = attention_mask.to(dtype=torch.bool)
        dense_rows = dense_logits[keep].float()
        quantized_rows = quantized_logits[keep].float()
        if dense_rows.shape != quantized_rows.shape:
            raise ValueError(f"Dense/quantized logit mismatch: {dense_rows.shape} != {quantized_rows.shape}")
        dense_logp = F.log_softmax(dense_rows, dim=-1)
        quantized_logp = F.log_softmax(quantized_rows, dim=-1)
        kl_sum += (dense_logp.exp() * (dense_logp - quantized_logp)).sum(dim=-1).double().sum()
        dense_top1 = dense_rows.argmax(dim=-1)
        quantized_top1 = quantized_rows.argmax(dim=-1)
        top1_sum += dense_top1.eq(quantized_top1).double().sum()
        if args.include_topn:
            for width, accumulator in ((5, top5_sum), (10, top10_sum)):
                effective = min(width, dense_rows.shape[-1])
                dense_top = dense_rows.topk(effective, dim=-1).indices
                quantized_top = quantized_rows.topk(effective, dim=-1).indices
                overlap = (dense_top.unsqueeze(-1) == quantized_top.unsqueeze(-2)).any(dim=-1).double().mean(-1)
                accumulator += overlap.sum()
        token_total += dense_rows.shape[0]

        if row_index < args.divergence_rows and dense_rows.shape[0] >= args.divergence_tokens:
            matches = dense_top1[: args.divergence_tokens].eq(quantized_top1[: args.divergence_tokens])
            mismatch = (~matches).nonzero(as_tuple=False).flatten()
            first = int(mismatch[0]) + 1 if mismatch.numel() else args.divergence_tokens + 1
            divergence_sequences += 1
            divergence_matching_tokens += int(matches.sum())
            divergence_total_tokens += args.divergence_tokens
            divergence_exact_sequences += int(bool(matches.all()))
            divergence_first_sum += first

        if row_index == 0 or (row_index + 1) % 16 == 0 or row_index + 1 == len(dataset):
            print(f"[diagnostics] rows={row_index + 1}/{len(dataset)} valid_tokens={token_total}", flush=True)

    if token_total < 1:
        raise ValueError("Evaluation produced no valid tokens")
    result = {
        "dense_model": args.dense_model,
        "checkpoint": str(checkpoint),
        "evaluation": {**asdict(evaluation_spec), "valid_tokens": token_total, "max_length": args.max_length},
        "metrics": {
            "final_kl": float((kl_sum / token_total).item()),
            "top1_agreement": float((top1_sum / token_total).item()),
        },
        "divergence_300": {
            "requested_sequences": min(args.divergence_rows, len(dataset)),
            "valid_sequences": divergence_sequences,
            "token_horizon": args.divergence_tokens,
            "token_top1_agreement": (
                divergence_matching_tokens / divergence_total_tokens if divergence_total_tokens else None
            ),
            "exact_sequence_agreement": (
                divergence_exact_sequences / divergence_sequences if divergence_sequences else None
            ),
            "first_divergence_token": (
                divergence_first_sum / divergence_sequences if divergence_sequences else None
            ),
        },
        "seconds": {
            "dense_load": dense_load_seconds,
            "dense_forward": dense_forward_seconds,
            "quantized_load": quantized_load_seconds,
            "quantized_forward": quantized_forward_seconds,
        },
    }
    if args.include_topn:
        result["metrics"].update(
            top5_overlap=float((top5_sum / token_total).item()),
            top10_overlap=float((top10_sum / token_total).item()),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def _tasks(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    from tests.eval import evaluate, format_eval_result_table, get_eval_task_results

    selected = args.task or list(TASKS)
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "backend": args.backend,
        "dtype": "float16",
        "batch_size": args.batch_size,
        "tasks": {},
    }
    for label in selected:
        task, apply_chat_template, suite_kwargs = TASKS[label]
        started = time.perf_counter()
        output = evaluate(
            model_or_id_or_path=str(checkpoint),
            tasks=[task],
            backend=BACKEND(args.backend),
            model_args={"dtype": "float16"},
            batch_size=args.batch_size,
            apply_chat_template=apply_chat_template,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
            suite_kwargs=suite_kwargs,
            trust_remote_code=False,
        )
        print(format_eval_result_table(output), flush=True)
        metrics = get_eval_task_results(output)
        payload["tasks"][label] = {
            "evalution_task": task,
            "seconds": time.perf_counter() - started,
            "metrics": next(iter(metrics.values())) if metrics else {},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _diagnostics(args) if args.command == "diagnostics" else _tasks(args)


if __name__ == "__main__":
    raise SystemExit(main())
