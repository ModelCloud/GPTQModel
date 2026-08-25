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
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
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
from gptqmodel.utils.diagnostic_metrics import (  # noqa: E402
    greedy_trajectory_metrics,
    shared_prefix_top1_metrics,
)

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
    diagnostics.add_argument(
        "--skip-independent-rollout",
        action="store_true",
        help="Skip the slower independent greedy rollout while retaining shared-prefix @32 diagnostics.",
    )
    diagnostics.add_argument("--include-topn", action="store_true", help="Also report Top-5/Top-10 overlap.")
    diagnostics.add_argument("--trust-remote-code", action="store_true")
    diagnostics.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    diagnostics.add_argument("--allow-calibration-overlap", action="store_true")

    divergence = subparsers.add_parser(
        "divergence300",
        help="Run the canonical 300-prompt, 32-token independent greedy-trajectory comparison.",
    )
    divergence.add_argument("--dense-model", required=True)
    divergence.add_argument("--checkpoint", type=Path, required=True)
    divergence.add_argument("--dataset", type=Path, required=True, help="Pinned local 300-prompt JSONL manifest.")
    divergence.add_argument("--device", default="cuda:0")
    divergence.add_argument("--output", type=Path, required=True)
    divergence.add_argument("--max-prompt-tokens", type=int, default=16384)
    divergence.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Compute dtype used identically by the dense and quantized models.",
    )
    divergence.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
        help="Attention implementation used identically by dense and quantized models.",
    )
    divergence.add_argument("--trust-remote-code", action="store_true")
    divergence.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)

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
        return tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
    raise ValueError("Every evaluation row must provide nonempty text or a supported messages list")


def _encode_prompt(row: dict[str, Any], tokenizer, *, max_prompt_tokens: int) -> dict[str, torch.Tensor]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) for item in messages):
        raise ValueError("Divergence-300 rows must contain a nonempty `messages` conversation")
    previous_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
    finally:
        tokenizer.truncation_side = previous_side
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer chat template did not return an encoded input mapping")
    return dict(encoded)


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)) / denominator
    return center - half_width, center + half_width


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
def _greedy_rollout(model, encoded: dict[str, torch.Tensor], *, token_count: int, pad_token_id: int) -> torch.Tensor:
    """Generate one independent fixed-horizon greedy continuation."""

    input_ids = encoded["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("Divergence-300 diagnostics require evaluation batch size 1")
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=token_count,
        min_new_tokens=token_count,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    continuation = generated[:, input_ids.shape[1] :]
    if continuation.shape != (1, token_count):
        raise RuntimeError(
            f"greedy decoder returned {continuation.shape[1]} new tokens; expected exactly {token_count}"
        )
    return continuation[0]


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
    divergence_aligned_token_sum = 0.0
    divergence_aligned_match_sum = torch.zeros(args.divergence_tokens, dtype=torch.float64)
    divergence_survival_sum = 0.0
    divergence_prefix_survival_sum = torch.zeros(args.divergence_tokens, dtype=torch.float64)
    divergence_first_sum = 0.0
    shared_prefix_sequences = 0
    shared_prefix_top1_sum = 0.0
    shared_prefix_exact_sum = 0.0
    shared_prefix_first_sum = 0.0
    legacy_first32_sequences = 0
    legacy_first32_top1_sum = 0.0
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

        if not args.skip_independent_rollout and row_index < args.divergence_rows:
            dense_trajectory = _greedy_rollout(
                dense,
                encoded,
                token_count=args.divergence_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            quantized_trajectory = _greedy_rollout(
                quantized,
                encoded,
                token_count=args.divergence_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            trajectory = greedy_trajectory_metrics(
                dense_trajectory,
                quantized_trajectory,
                token_count=args.divergence_tokens,
            )
            divergence_sequences += 1
            divergence_aligned_token_sum += float(trajectory["aligned_token_agreement"].item())
            divergence_aligned_match_sum += trajectory["aligned_token_matches"].double().cpu()
            divergence_survival_sum += float(trajectory["trajectory_survival"].item())
            divergence_prefix_survival_sum += trajectory["prefix_survival"].double().cpu()
            divergence_first_sum += float(trajectory["first_divergence_token"].item())

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
        if row_index < args.divergence_rows:
            # Match llama.cpp's KL benchmark conditioning policy: score after
            # a half-context warmup, while retaining a fixed 32-position
            # horizon for this harness.
            shared_prefix = shared_prefix_top1_metrics(
                dense_rows,
                quantized_rows,
                token_count=args.divergence_tokens,
                start_index=dense_rows.shape[0] // 2,
            )
            if shared_prefix is not None:
                shared_prefix_sequences += 1
                shared_prefix_top1_sum += float(shared_prefix["top1_agreement"].item())
                shared_prefix_exact_sum += float(shared_prefix["exact_sequence_agreement"].item())
                shared_prefix_first_sum += float(shared_prefix["first_mismatch_token"].item())
            legacy_first32 = shared_prefix_top1_metrics(
                dense_rows,
                quantized_rows,
                token_count=args.divergence_tokens,
            )
            if legacy_first32 is not None:
                legacy_first32_sequences += 1
                legacy_first32_top1_sum += float(legacy_first32["top1_agreement"].item())
        if args.include_topn:
            for width, accumulator in ((5, top5_sum), (10, top10_sum)):
                effective = min(width, dense_rows.shape[-1])
                dense_top = dense_rows.topk(effective, dim=-1).indices
                quantized_top = quantized_rows.topk(effective, dim=-1).indices
                overlap = (dense_top.unsqueeze(-1) == quantized_top.unsqueeze(-2)).any(dim=-1).double().mean(-1)
                accumulator += overlap.sum()
        token_total += dense_rows.shape[0]

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
        "divergence_300_at_32": {
            "display_name": "Divergence-300 @32",
            "requested_sequences": min(args.divergence_rows, len(dataset)),
            "valid_sequences": divergence_sequences,
            "token_horizon": args.divergence_tokens,
            "protocol": "independent_greedy_rollout",
            "published_aggregation_status": "Unsloth has not published its scalar reduction; report both reductions",
            "skipped": args.skip_independent_rollout,
            "independent_token_top1_agreement_at_32": (
                divergence_aligned_token_sum / divergence_sequences if divergence_sequences else None
            ),
            "independent_token_top1_by_position": (
                {
                    str(position): float(
                        divergence_aligned_match_sum[position - 1].item() / divergence_sequences
                    )
                    for position in range(1, args.divergence_tokens + 1)
                }
                if divergence_sequences
                else None
            ),
            "independent_token_top1_by_horizon": (
                {
                    str(horizon): float(
                        divergence_aligned_match_sum[:horizon].sum().item()
                        / (divergence_sequences * horizon)
                    )
                    for horizon in range(1, args.divergence_tokens + 1)
                }
                if divergence_sequences
                else None
            ),
            "exact_trajectory_agreement_at_32": (
                divergence_survival_sum / divergence_sequences if divergence_sequences else None
            ),
            "exact_prefix_survival_by_horizon": (
                {
                    str(horizon): float(divergence_prefix_survival_sum[horizon - 1].item() / divergence_sequences)
                    for horizon in range(1, args.divergence_tokens + 1)
                }
                if divergence_sequences
                else None
            ),
            "trajectory_survival": divergence_survival_sum / divergence_sequences if divergence_sequences else None,
            "exact_sequence_agreement": (
                divergence_survival_sum / divergence_sequences if divergence_sequences else None
            ),
            "aligned_token_agreement": (
                divergence_aligned_token_sum / divergence_sequences if divergence_sequences else None
            ),
            "first_divergence_token": (
                divergence_first_sum / divergence_sequences if divergence_sequences else None
            ),
        },
        "sp_top1_32_w50": {
            "requested_sequences": min(args.divergence_rows, len(dataset)),
            "valid_sequences": shared_prefix_sequences,
            "token_horizon": args.divergence_tokens,
            "display_name": "Shared-prefix top-1 agreement@32, with 50% context warmup",
            "protocol": "teacher_forced_shared_prefix",
            "position_policy": "start_at_50_percent_context",
            "top1_agreement": shared_prefix_top1_sum / shared_prefix_sequences if shared_prefix_sequences else None,
            "exact_sequence_agreement": (
                shared_prefix_exact_sum / shared_prefix_sequences if shared_prefix_sequences else None
            ),
            "first_mismatch_token": (
                shared_prefix_first_sum / shared_prefix_sequences if shared_prefix_sequences else None
            ),
        },
        "legacy_shared_prefix_first_32": {
            "requested_sequences": min(args.divergence_rows, len(dataset)),
            "valid_sequences": legacy_first32_sequences,
            "token_horizon": args.divergence_tokens,
            "protocol": "teacher_forced_shared_prefix",
            "position_policy": "start_at_token_zero",
            "top1_agreement": (
                legacy_first32_top1_sum / legacy_first32_sequences if legacy_first32_sequences else None
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


@torch.inference_mode()
def _divergence300(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if not dataset_path.is_file() or dataset_path.suffix.lower() != ".jsonl":
        raise FileNotFoundError(f"Pinned Divergence-300 JSONL does not exist: {dataset_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_path}")
    if args.max_prompt_tokens < 1:
        raise ValueError("max prompt tokens must be positive")

    evaluation_spec = DatasetSlice(str(dataset_path), None, "train", 0, 300)
    validate_evaluation_is_held_out(checkpoint, evaluation_spec, allow_overlap=False)
    dataset = load_dataset_slice(evaluation_spec)
    if len(dataset) != 300:
        raise RuntimeError(f"Divergence-300 requires exactly 300 prompts, got {len(dataset)}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.dense_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    started = time.perf_counter()
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        dtype=model_dtype,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).eval()
    dense_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    quantized = GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.QVQ,
        dtype=model_dtype,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    quantized_load_seconds = time.perf_counter() - started

    horizon = 32
    prefix_successes = torch.zeros(horizon, dtype=torch.int64)
    aligned_token_sum = 0.0
    aligned_match_sum = torch.zeros(horizon, dtype=torch.int64)
    first_divergence_sum = 0.0
    prompt_token_total = 0
    prompts_at_token_cap = 0
    dense_generate_seconds = 0.0
    quantized_generate_seconds = 0.0
    per_source: dict[str, dict[str, int]] = {}
    prompt_results = []
    for row_index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        encoded_cpu = _encode_prompt(row, tokenizer, max_prompt_tokens=args.max_prompt_tokens)
        prompt_tokens = int(encoded_cpu["input_ids"].shape[1])
        prompt_token_total += prompt_tokens
        prompts_at_token_cap += int(prompt_tokens == args.max_prompt_tokens)
        encoded = {name: value.to(args.device) for name, value in encoded_cpu.items()}

        started = time.perf_counter()
        dense_tokens = _greedy_rollout(
            dense,
            encoded,
            token_count=horizon,
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        dense_generate_seconds += time.perf_counter() - started

        started = time.perf_counter()
        quantized_tokens = _greedy_rollout(
            quantized,
            encoded,
            token_count=horizon,
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        quantized_generate_seconds += time.perf_counter() - started

        metrics = greedy_trajectory_metrics(dense_tokens, quantized_tokens, token_count=horizon)
        prefix = metrics["prefix_survival"].to(dtype=torch.int64).cpu()
        prefix_successes += prefix
        aligned_match_sum += metrics["aligned_token_matches"].to(dtype=torch.int64).cpu()
        aligned_token_sum += float(metrics["aligned_token_agreement"].item())
        first_divergence_sum += float(metrics["first_divergence_token"].item())
        exact = int(metrics["trajectory_survival"].item())
        source_group = str(row.get("source_group", "unknown"))
        source_stats = per_source.setdefault(
            source_group,
            {"prompts": 0, "exact_at_32": 0, "aligned_token_matches": 0},
        )
        source_stats["prompts"] += 1
        source_stats["exact_at_32"] += exact
        source_stats["aligned_token_matches"] += int(metrics["aligned_token_matches"].sum().item())
        prompt_results.append(
            {
                "manifest_index": row.get("manifest_index", row_index),
                "source_group": source_group,
                "source_id": row.get("source_id"),
                "prompt_sha256": row.get("prompt_sha256"),
                "prompt_tokens": prompt_tokens,
                "exact_at_32": bool(exact),
                "first_divergence_token": int(metrics["first_divergence_token"].item()),
                "aligned_token_agreement": float(metrics["aligned_token_agreement"].item()),
                "dense_token_ids": dense_tokens.tolist(),
                "quantized_token_ids": quantized_tokens.tolist(),
                "dense_text": tokenizer.decode(dense_tokens, skip_special_tokens=False),
                "quantized_text": tokenizer.decode(quantized_tokens, skip_special_tokens=False),
            }
        )
        if row_index == 0 or (row_index + 1) % 10 == 0:
            print(
                f"[Divergence-300 @32] prompts={row_index + 1}/300 "
                f"exact={int(prefix_successes[-1])}/{row_index + 1}",
                flush=True,
            )

    successes_at_32 = int(prefix_successes[-1].item())
    confidence_low, confidence_high = _wilson_interval(successes_at_32, 300)
    for stats in per_source.values():
        stats["exact_trajectory_agreement_at_32"] = stats["exact_at_32"] / stats["prompts"]  # type: ignore[assignment]
        stats["independent_token_top1_agreement_at_32"] = (  # type: ignore[assignment]
            stats["aligned_token_matches"] / (stats["prompts"] * horizon)
        )
    result = {
        "schema": "qvq.divergence300.result.v2",
        "dense_model": args.dense_model,
        "checkpoint": str(checkpoint),
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "prompts": 300,
            "max_prompt_tokens": args.max_prompt_tokens,
            "prompt_tokens_total": prompt_token_total,
            "prompts_at_token_cap": prompts_at_token_cap,
        },
        "decoding": {
            "algorithm": "independent_greedy_argmax",
            "do_sample": False,
            "new_tokens": horizon,
            "chat_template": True,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        },
        "divergence_300_at_32": {
            "display_name": "Divergence-300 @32",
            "protocol": "independent_greedy_rollout",
            "published_aggregation_status": (
                "Unsloth has not published its scalar reduction; report both aligned-token top-1 and exact survival"
            ),
            "independent_token_top1_agreement_at_32": aligned_token_sum / 300,
            "independent_token_top1_by_position": {
                str(index + 1): int(value.item()) / 300 for index, value in enumerate(aligned_match_sum)
            },
            "independent_token_top1_by_horizon": {
                str(index + 1): int(aligned_match_sum[: index + 1].sum().item()) / (300 * (index + 1))
                for index in range(horizon)
            },
            "exact_trajectory_agreement_at_32": successes_at_32 / 300,
            "exact_prompts_at_32": successes_at_32,
            "exact_trajectory_wilson_95_percent": [confidence_low, confidence_high],
            "exact_prefix_survival_by_horizon": {
                str(index + 1): int(value.item()) / 300 for index, value in enumerate(prefix_successes)
            },
            "mean_first_divergence_token": first_divergence_sum / 300,
            "per_source": dict(sorted(per_source.items())),
        },
        "seconds": {
            "dense_load": dense_load_seconds,
            "quantized_load": quantized_load_seconds,
            "dense_generate": dense_generate_seconds,
            "quantized_generate": quantized_generate_seconds,
        },
        "prompts": prompt_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "prompts"}, indent=2, sort_keys=True))
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
    if args.command == "diagnostics":
        return _diagnostics(args)
    if args.command == "divergence300":
        return _divergence300(args)
    return _tasks(args)


if __name__ == "__main__":
    raise SystemExit(main())
