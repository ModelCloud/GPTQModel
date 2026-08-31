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
import importlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

try:
    from datetime import UTC
except ImportError:  # Python 3.10 compatibility
    UTC = timezone.utc
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Evalution already maintains sample-level running metrics in progress titles.
# LogBar otherwise suppresses them when it detects an agent or CI environment.
os.environ["LOGBAR_FORCE_PROGRESS"] = "1"

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
    "mmlu_humanities": ("mmlu", False, {"subsets": "humanities"}),
    "mmlu_history": ("mmlu", False, {"subsets": MMLU_HISTORY_SUBSETS}),
}


class _ChoiceProgressAsRows:
    """Expose four MMLU choice completions as one completed question row."""

    def __init__(self, progress: Any, *, choices_per_row: int) -> None:
        self._progress = progress
        self._choices_per_row = choices_per_row
        self._completed_choices = 0
        self._row_advanced = False

    def next(self) -> _ChoiceProgressAsRows:
        self._completed_choices += 1
        self._row_advanced = self._completed_choices % self._choices_per_row == 0
        if self._row_advanced:
            self._progress.next()
        return self

    def draw(self) -> _ChoiceProgressAsRows:
        if self._row_advanced:
            self._progress.draw()
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._progress, name)


@contextmanager
def _mmlu_question_row_progress(enabled: bool) -> Generator[None]:
    """Render MMLU progress in completed rows while preserving choice-level work."""

    if not enabled:
        yield
        return

    mmlu_module = importlib.import_module("evalution.benchmarks.mmlu")
    original_manual_progress = mmlu_module.manual_progress
    choices_per_row = len(mmlu_module._MMLU_LABELS)

    def manual_row_progress(total: int, *, title: str, subtitle: str) -> Any:
        if not title.endswith(": scoring answer choices"):
            return original_manual_progress(total, title=title, subtitle=subtitle)
        if total % choices_per_row:
            raise ValueError(f"MMLU choice request total {total} is not divisible by {choices_per_row}")
        progress = original_manual_progress(
            total // choices_per_row,
            title=title.removesuffix("scoring answer choices") + "completed question rows",
            subtitle=f"{subtitle} choices_per_row={choices_per_row}",
        )
        return _ChoiceProgressAsRows(progress, choices_per_row=choices_per_row)

    mmlu_module.manual_progress = manual_row_progress
    try:
        yield
    finally:
        mmlu_module.manual_progress = original_manual_progress


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

    micro_math = subparsers.add_parser(
        "micro_math",
        help="Run the fast, held-out math capability proxy suite.",
    )
    micro_math.add_argument("--dense-model", required=True)
    micro_math.add_argument("--checkpoint", type=Path, required=True)
    micro_math.add_argument("--dataset", type=Path, required=True, help="Pinned disjoint Mini-GSM JSONL.")
    micro_math.add_argument("--manifest", type=Path, required=True, help="Dataset binding manifest.")
    micro_math.add_argument("--output", type=Path, required=True)
    micro_math.add_argument("--rows", type=int, default=64)
    micro_math.add_argument("--rollout-tokens", type=int, default=48)
    micro_math.add_argument("--max-prompt-tokens", type=int, default=2048)
    micro_math.add_argument("--device", default="cuda:0")
    micro_math.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    micro_math.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    micro_math.add_argument("--trust-remote-code", action="store_true")
    micro_math.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    micro_math.add_argument("--allow-calibration-overlap", action="store_true")

    tasks = subparsers.add_parser("tasks", help="Run Evalution tasks on the quantized checkpoint.")
    tasks.add_argument("--checkpoint", type=Path, required=True)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument(
        "--resume",
        action="store_true",
        help="Append missing tasks to a compatible task report, publishing atomically after every task.",
    )
    tasks.add_argument("--batch-size", type=int, default=64)
    tasks.add_argument("--backend", choices=(BACKEND.QVQ.value, BACKEND.EXL3_EXLLAMA_V3.value), default="qvq")
    tasks.add_argument("--device", default="cuda:0")
    tasks.add_argument(
        "--attn-implementation",
        default="paged|flash_attention_2",
        choices=("paged|flash_attention_2", "paged|sdpa"),
        help="Paged attention backend; paged mode also activates native continuous batching.",
    )
    tasks.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy alias: capture both continuous-batching varlen and decode graphs.",
    )
    tasks.add_argument(
        "--cuda-graph-mode",
        choices=("auto", "off", "varlen", "decode", "both"),
        default=None,
        help=(
            "Transformers ContinuousBatchingConfig graph policy. The default is decode-only "
            "(varlen prefill eager, decode captured); use auto to let the engine decide."
        ),
    )
    tasks.add_argument("--allow-block-sharing", action=argparse.BooleanOptionalAction, default=None)
    tasks.add_argument("--max-batch-tokens", type=int, default=None)
    tasks.add_argument("--max-blocks-per-request", type=int, default=None)
    tasks.add_argument("--use-async-batching", action=argparse.BooleanOptionalAction, default=None)
    tasks.add_argument("--q-padding-interval-size", type=int, default=None)
    tasks.add_argument("--kv-padding-interval-size", type=int, default=None)
    tasks.add_argument("--max-cached-graphs", type=int, default=None)
    tasks.add_argument(
        "--loglikelihood-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reuse repeated long loglikelihood prefixes through a bounded KV cache "
            "(enabled by default by Evalution)."
        ),
    )
    tasks.add_argument(
        "--loglikelihood-prefix-cache-prewarm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Prefill each shared subject prefix before scoring its question suffixes; "
            "reports warmup misses separately from steady-state hits (enabled by default)."
        ),
    )
    tasks.add_argument(
        "--loglikelihood-prefix-cache-prewarm-batch-size",
        type=int,
        default=None,
        help="Maximum number of distinct prefixes in one prewarm forward (default: 32).",
    )
    tasks.add_argument(
        "--loglikelihood-prefix-cache-release-after-group",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Release scorer prefix KV entries after the MMLU subject group completes.",
    )
    tasks.add_argument(
        "--loglikelihood-prefix-cache-min-tokens",
        type=int,
        default=None,
        help="Minimum shared prefix length eligible for scorer-side KV reuse.",
    )
    tasks.add_argument(
        "--loglikelihood-prefix-cache-max-entries",
        type=int,
        default=None,
        help="Maximum GPU-resident scorer prefix KV entries.",
    )
    tasks.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for a smoke/canary run; omit for the complete task split.",
    )
    tasks.add_argument("--task", action="append", choices=tuple(TASKS), help="Repeat to select tasks.")
    return parser


def _canonical_source(source: str) -> str:
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else source


def _publish_snapshot_evaluation(checkpoint: Path, result_path: Path, result: dict[str, Any], *, kind: str) -> None:
    """Keep an immutable, discoverable copy of every post-quant evaluation in the checkpoint."""
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        return
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result_digest = hashlib.sha256(
        (serialized + f"\nsource:{result_path.expanduser().resolve()}\n").encode("utf-8")
    ).hexdigest()[:12]
    # A digest-qualified filename makes every snapshot append-only.  Keep the
    # historical fixed filename as a compatibility alias only when it has not
    # been used before; never overwrite a prior evaluation in-place.
    stem = f"post_quant_eval_result_{kind}_{result_digest}"
    snapshot_json = checkpoint / f"{stem}.json"
    if not snapshot_json.exists():
        snapshot_json.write_text(serialized, encoding="utf-8")
    lines = [f"# Post-quantization evaluation: {kind}", "", f"Source report: `{result_path}`", ""]
    if isinstance(result.get("metrics"), dict):
        lines += ["## Metrics", "", *[f"- **{k}**: {v}" for k, v in result["metrics"].items()], ""]
    if isinstance(result.get("divergence_300_at_32"), dict):
        d = result["divergence_300_at_32"]
        lines += ["## Divergence-300", "", f"- **D300 token top-1**: {d.get('independent_token_top1_agreement_at_32')}", f"- **Exact trajectory agreement**: {d.get('exact_trajectory_agreement_at_32')}", ""]
    if isinstance(result.get("tasks"), dict):
        lines += ["## Tasks", ""]
        for label, task in result["tasks"].items():
            lines.append(f"- **{label}**: {task.get('metrics', {})}")
    snapshot_md = checkpoint / f"{stem}.md"
    if not snapshot_md.exists():
        snapshot_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    legacy_stem = f"post_quant_eval_result_{kind}"
    legacy_json = checkpoint / f"{legacy_stem}.json"
    legacy_md = checkpoint / f"{legacy_stem}.md"
    if not legacy_json.exists():
        legacy_json.write_text(serialized, encoding="utf-8")
    if not legacy_md.exists():
        legacy_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def validate_divergence_manifest_binding(checkpoint: Path, dataset_path: Path) -> None:
    """Require canonical D300 evaluations to use the manifest-bound file."""

    run_manifest = checkpoint / "qvq_quantize_run.json"
    if not run_manifest.is_file():
        return
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    binding = payload.get("disjointness_manifest")
    if not isinstance(binding, dict) or not binding.get("strict_required"):
        return
    evaluations = binding.get("evaluation_bindings")
    if not isinstance(evaluations, dict):
        raise RuntimeError(
            "checkpoint requires a strict disjointness manifest but has no evaluation bindings"
        )
    requested = dataset_path.expanduser().resolve()
    digest = hashlib.sha256()
    with requested.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    requested_sha = digest.hexdigest()
    matches = []
    for label in ("d300", "d300_locked"):
        item = evaluations.get(label)
        if isinstance(item, dict) and item.get("path"):
            bound_path = Path(item["path"]).expanduser().resolve()
            if bound_path == requested:
                matches.append((label, item))
    if len(matches) != 1:
        raise RuntimeError(
            "canonical divergence evaluation dataset is not bound by the checkpoint's strict disjointness manifest: "
            f"{requested}"
        )
    label, item = matches[0]
    if item.get("sha256") != requested_sha:
        raise RuntimeError(
            f"checkpoint-bound {label} manifest changed on disk: {requested}"
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


_MICRO_STRICT_ANSWER = re.compile(r"####\s*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))")
_MICRO_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def _micro_normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").replace("$", "").strip().rstrip(".")
    return value or None


def _micro_answer(text: str) -> str | None:
    strict = _MICRO_STRICT_ANSWER.findall(text)
    if strict:
        return _micro_normalize_answer(strict[-1])
    values = _MICRO_NUMBER.findall(text)
    return _micro_normalize_answer(values[-1]) if values else None


def _micro_equal(left: str | None, right: str | None) -> bool:
    left, right = _micro_normalize_answer(left), _micro_normalize_answer(right)
    if left is None or right is None:
        return False
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def _micro_question_text(row: Mapping[str, Any]) -> str:
    value = row.get("question")
    if isinstance(value, str) and value.strip():
        return value
    messages = row.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if isinstance(messages, (list, tuple)):
        return "\n".join(
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, Mapping) and item.get("role") == "user"
        )
    raise ValueError("micro-math rows require a non-empty question or user messages")


def _micro_chat_ids(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if not isinstance(encoded, torch.Tensor):
        raise TypeError("chat template must return token IDs")
    return [int(token) for token in encoded.reshape(-1).tolist()]


def _micro_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _micro_validate_inputs(
    checkpoint: Path,
    dataset_path: Path,
    manifest_path: Path,
    *,
    rows: int,
    allow_calibration_overlap: bool,
) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.get("dataset", {})
    if manifest.get("schema") != "qvq.micro_math.v1" or manifest.get("status") != "pass":
        raise RuntimeError("micro-math manifest is missing its passing v1 contract")
    if Path(str(binding.get("path", ""))).expanduser().resolve() != dataset_path.resolve():
        raise RuntimeError("micro-math dataset path does not match its manifest")
    if binding.get("sha256") != _micro_file_sha256(dataset_path):
        raise RuntimeError("micro-math dataset changed after its manifest was generated")
    raw_rows = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_rows.append(json.loads(line))
    if len(raw_rows) < rows:
        raise ValueError(f"micro-math dataset has {len(raw_rows)} rows, requested {rows}")
    selected = raw_rows[:rows]
    from scripts.check_calibration_disjointness import digest, normalize

    own_hashes: set[str] = set()
    for index, row in enumerate(selected):
        normalized_hash = digest(normalize(_micro_question_text(row)))
        if normalized_hash in own_hashes:
            raise RuntimeError(f"micro-math dataset contains a duplicate question at row {index}")
        own_hashes.add(normalized_hash)

    if allow_calibration_overlap:
        return selected
    run_manifest = checkpoint / "qvq_quantize_run.json"
    if not run_manifest.is_file():
        raise RuntimeError("checkpoint has no quantization manifest for micro-math disjointness")
    run = json.loads(run_manifest.read_text(encoding="utf-8"))
    disjoint = run.get("disjointness_manifest")
    if not isinstance(disjoint, Mapping) or not disjoint.get("strict_required") or disjoint.get("status") != "pass":
        raise RuntimeError("micro-math requires a passing strict disjointness manifest on the checkpoint")
    calibration_hashes: set[str] = set()
    for raw in run.get("datasets", {}).values():
        if not isinstance(raw, Mapping) or not raw.get("source") or int(raw.get("rows", 0)) < 1:
            continue
        spec = DatasetSlice(
            str(raw["source"]), raw.get("config"), str(raw.get("split", "train")),
            int(raw.get("row_start", 0)), int(raw["rows"]),
        )
        for item in load_dataset_slice(spec):
            calibration_hashes.add(digest(normalize(_micro_question_text(dict(item)))))
    overlap = sorted(own_hashes & calibration_hashes)
    if overlap:
        raise RuntimeError(f"micro-math calibration overlap detected ({len(overlap)} normalized questions)")
    return selected


@torch.inference_mode()
def _greedy_rollout(model, encoded: dict[str, torch.Tensor], *, token_count: int, pad_token_id: int) -> torch.Tensor:
    """Generate one independent fixed-horizon greedy continuation.

    ``generate(min_new_tokens=N)`` is not a literal greedy rollout: the
    generation processor suppresses EOS until the minimum length is reached.
    Divergence-300 treats EOS as an ordinary token and therefore must make
    exactly ``N`` argmax decisions itself.  We still use the model KV cache
    when the model provides one, but never install a stopping criterion.
    """

    input_ids = encoded["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("Divergence-300 diagnostics require evaluation batch size 1")

    if token_count < 1:
        raise ValueError("greedy rollout token_count must be positive")
    del pad_token_id  # EOS/PAD are intentionally not special in this metric.

    generated = input_ids.clone()
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()
    position_ids = encoded.get("position_ids")
    if position_ids is not None:
        position_ids = position_ids.clone()
    past_key_values = None

    # Keep auxiliary model inputs (for example token_type_ids) but do not
    # reuse the caller's input_ids/attention mask objects after appending.
    static_inputs = {
        key: value
        for key, value in encoded.items()
        if key not in {"input_ids", "attention_mask", "position_ids", "past_key_values"}
    }
    for _ in range(token_count):
        if past_key_values is None:
            model_inputs = dict(static_inputs)
            model_inputs["input_ids"] = generated
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if position_ids is not None:
                model_inputs["position_ids"] = position_ids
        else:
            model_inputs = dict(static_inputs)
            model_inputs["input_ids"] = generated[:, -1:]
            model_inputs["past_key_values"] = past_key_values
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            if position_ids is not None:
                model_inputs["position_ids"] = position_ids[:, -1:]
            # Auxiliary per-token inputs must follow the one-token cached
            # input, while scalar/global inputs are safe to reuse unchanged.
            for key, value in tuple(model_inputs.items()):
                if key in {"input_ids", "attention_mask", "position_ids", "past_key_values"}:
                    continue
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] > 1:
                    model_inputs[key] = value[:, -1:]

        output = model(**model_inputs, use_cache=True)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, Mapping):
            logits = output.get("logits")
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("model forward did not return [batch, sequence, vocab] logits")
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
            )
        if position_ids is not None:
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=1)

        next_past = getattr(output, "past_key_values", None)
        if next_past is None and isinstance(output, Mapping):
            next_past = output.get("past_key_values")
        if next_past is None and isinstance(output, (tuple, list)) and len(output) > 1:
            # Standard tuple causal-LM outputs place the cache immediately
            # after logits when it is requested.
            next_past = output[1]
        past_key_values = next_past

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
    _publish_snapshot_evaluation(checkpoint, args.output, result, kind="diagnostics")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


@torch.inference_mode()
def _micro_math(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_path}")
    if args.rows < 1 or args.rollout_tokens < 1 or args.max_prompt_tokens < 1:
        raise ValueError("micro-math rows, rollout tokens, and max prompt tokens must be positive")
    rows = _micro_validate_inputs(
        checkpoint,
        dataset_path,
        manifest_path,
        rows=args.rows,
        allow_calibration_overlap=args.allow_calibration_overlap,
    )
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

    dense_ce_sum = 0.0
    quant_ce_sum = 0.0
    delta_kl_sum = 0.0
    dense_answer_logprob_sum = 0.0
    quant_answer_logprob_sum = 0.0
    dense_answer_margin_sum = 0.0
    quant_answer_margin_sum = 0.0
    answer_token_count = 0
    critical_dense_target = 0
    critical_quant_target = 0
    critical_quant_dense = 0
    critical_count = 0
    dense_rollout_correct = 0
    quant_rollout_correct = 0
    dense_rollout_invalid = 0
    quant_rollout_invalid = 0
    dense_wrong_quant_right = 0
    dense_right_quant_wrong = 0
    teacher_forced_tokens = 0
    dense_forward_seconds = 0.0
    quantized_forward_seconds = 0.0
    dense_rollout_seconds = 0.0
    quantized_rollout_seconds = 0.0
    per_row: list[dict[str, Any]] = []

    for row_index, raw in enumerate(rows):
        row = dict(raw)
        question = _micro_question_text(row)
        answer = str(row.get("answer", ""))
        if "####" not in answer:
            raise ValueError(f"micro-math row {row_index} answer has no #### delimiter")
        gold = _micro_normalize_answer(answer.rsplit("####", 1)[1])
        if gold is None:
            raise ValueError(f"micro-math row {row_index} has an empty numeric answer")
        prompt_messages = [{"role": "user", "content": question}]
        prompt_ids = _micro_chat_ids(tokenizer, prompt_messages, add_generation_prompt=True)
        target_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError(f"micro-math row {row_index} has an empty target")
        if len(prompt_ids) + len(target_ids) > args.max_prompt_tokens:
            raise ValueError(f"micro-math row {row_index} exceeds --max-prompt-tokens")
        full_ids = prompt_ids + [int(token) for token in target_ids]
        encoded = {
            "input_ids": torch.tensor([full_ids], dtype=torch.long, device=args.device),
            "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=args.device),
        }
        target = encoded["input_ids"][:, len(prompt_ids):]
        started = time.perf_counter()
        dense_logits = _model_logits(dense, encoded)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        dense_forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        quant_logits = _model_logits(quantized, encoded)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        quantized_forward_seconds += time.perf_counter() - started

        start = len(prompt_ids) - 1
        dense_scores = dense_logits[:, start:-1, :].float()
        quant_scores = quant_logits[:, start:-1, :].float()
        if dense_scores.shape[1] != target.shape[1]:
            raise RuntimeError("micro-math target/logit alignment mismatch")
        dense_logp = F.log_softmax(dense_scores, dim=-1)
        quant_logp = F.log_softmax(quant_scores, dim=-1)
        dense_gold = dense_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        quant_gold = quant_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        dense_ce = float((-dense_gold.mean()).item())
        quant_ce = float((-quant_gold.mean()).item())
        token_count = int(target.shape[1])
        dense_ce_sum += dense_ce * token_count
        quant_ce_sum += quant_ce * token_count
        delta_kl_sum += float(
            (dense_logp.exp() * (dense_logp - quant_logp)).sum(dim=-1).mean().item()
        ) * token_count
        teacher_forced_tokens += token_count

        suffix_ids = [int(token) for token in tokenizer(answer.rsplit("####", 1)[1], add_special_tokens=False).input_ids]
        suffix_len = min(len(suffix_ids), token_count)
        suffix_slice = slice(token_count - suffix_len, token_count)
        dense_answer_logprob = float(dense_gold[:, suffix_slice].sum().item())
        quant_answer_logprob = float(quant_gold[:, suffix_slice].sum().item())
        dense_answer_logprob_sum += dense_answer_logprob
        quant_answer_logprob_sum += quant_answer_logprob
        dense_answer_margins = dense_scores[:, suffix_slice, :].topk(2, dim=-1).values
        quant_answer_margins = quant_scores[:, suffix_slice, :].topk(2, dim=-1).values
        dense_answer_margin = dense_gold[:, suffix_slice] - torch.where(
            dense_scores[:, suffix_slice, :].argmax(dim=-1).eq(target[:, suffix_slice]),
            dense_answer_margins[..., 1],
            dense_answer_margins[..., 0],
        )
        quant_answer_margin = quant_gold[:, suffix_slice] - torch.where(
            quant_scores[:, suffix_slice, :].argmax(dim=-1).eq(target[:, suffix_slice]),
            quant_answer_margins[..., 1],
            quant_answer_margins[..., 0],
        )
        dense_answer_margin_sum += float(dense_answer_margin.sum().item())
        quant_answer_margin_sum += float(quant_answer_margin.sum().item())
        answer_token_count += suffix_len

        critical_mask = torch.tensor(
            [bool(re.search(r"[0-9+\-*/=<>%]", tokenizer.decode([token], skip_special_tokens=False))) for token in target_ids],
            dtype=torch.bool,
            device=args.device,
        ).unsqueeze(0)
        if bool(critical_mask.any()):
            dense_pred = dense_scores.argmax(dim=-1)
            quant_pred = quant_scores.argmax(dim=-1)
            critical_dense_target += int((dense_pred.eq(target) & critical_mask).sum().item())
            critical_quant_target += int((quant_pred.eq(target) & critical_mask).sum().item())
            critical_quant_dense += int((quant_pred.eq(dense_pred) & critical_mask).sum().item())
            critical_count += int(critical_mask.sum().item())

        prompt_encoded = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=args.device),
            "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=args.device),
        }
        started = time.perf_counter()
        dense_cont = _greedy_rollout(
            dense, prompt_encoded, token_count=args.rollout_tokens, pad_token_id=tokenizer.pad_token_id
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        dense_rollout_seconds += time.perf_counter() - started
        started = time.perf_counter()
        quant_cont = _greedy_rollout(
            quantized, prompt_encoded, token_count=args.rollout_tokens, pad_token_id=tokenizer.pad_token_id
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        quantized_rollout_seconds += time.perf_counter() - started
        dense_text = tokenizer.decode(dense_cont, skip_special_tokens=True)
        quant_text = tokenizer.decode(quant_cont, skip_special_tokens=True)
        dense_answer = _micro_answer(dense_text)
        quant_answer = _micro_answer(quant_text)
        dense_ok = _micro_equal(dense_answer, gold)
        quant_ok = _micro_equal(quant_answer, gold)
        dense_rollout_correct += int(dense_ok)
        quant_rollout_correct += int(quant_ok)
        dense_wrong_quant_right += int(not dense_ok and quant_ok)
        dense_right_quant_wrong += int(dense_ok and not quant_ok)
        dense_rollout_invalid += int(dense_answer is None)
        quant_rollout_invalid += int(quant_answer is None)
        per_row.append(
            {
                "id": row.get("id", f"row-{row_index}"),
                "gold_answer": gold,
                "dense_answer": dense_answer,
                "quantized_answer": quant_answer,
                "dense_correct": dense_ok,
                "quantized_correct": quant_ok,
                "dense_ce": dense_ce,
                "quantized_ce": quant_ce,
                "delta_ce": quant_ce - dense_ce,
                "dense_answer_logprob": dense_answer_logprob,
                "quantized_answer_logprob": quant_answer_logprob,
                "dense_answer_margin": float(dense_answer_margin.mean().item()),
                "quantized_answer_margin": float(quant_answer_margin.mean().item()),
                "critical_tokens": int(critical_mask.sum().item()),
                "dense_text": dense_text,
                "quantized_text": quant_text,
            }
        )
        if row_index == 0 or (row_index + 1) % 16 == 0 or row_index + 1 == len(rows):
            print(
                f"[micro-math] rows={row_index + 1}/{len(rows)} "
                f"quant_answer={quant_rollout_correct}/{row_index + 1} "
                f"invalid={quant_rollout_invalid}",
                flush=True,
            )

    if teacher_forced_tokens < 1 or answer_token_count < 1:
        raise RuntimeError("micro-math produced no teacher-forced or answer-suffix tokens")

    result = {
        "schema": "qvq.micro_math.result.v1",
        "dense_model": args.dense_model,
        "checkpoint": str(checkpoint),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _micro_file_sha256(dataset_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _micro_file_sha256(manifest_path),
            "rows": len(rows),
            "protocol": "GSM8K main train rows, disjoint from checkpoint preparation and benchmark manifests",
        },
        "protocol": {
            "teacher_forced": "reference reasoning answer tokens after a chat-template prompt",
            "rollout": "independent greedy argmax with EOS ordinary and fixed horizon",
            "rollout_tokens": args.rollout_tokens,
            "critical_token_regex": r"[0-9+\-*/=<>%]",
        },
        "metrics": {
            "mini_math_exact_answer_accuracy": quant_rollout_correct / len(rows),
            "mini_math_dense_exact_answer_accuracy": dense_rollout_correct / len(rows),
            "short_rollout_semantic_success": quant_rollout_correct / len(rows),
            "short_rollout_invalid": quant_rollout_invalid / len(rows),
            "dense_short_rollout_invalid": dense_rollout_invalid / len(rows),
            "reasoning_delta_ce": (quant_ce_sum - dense_ce_sum) / teacher_forced_tokens,
            "reasoning_dense_ce": dense_ce_sum / teacher_forced_tokens,
            "reasoning_quantized_ce": quant_ce_sum / teacher_forced_tokens,
            "reasoning_delta_kl": delta_kl_sum / teacher_forced_tokens,
            "answer_token_logprob_dense": dense_answer_logprob_sum / answer_token_count,
            "answer_token_logprob_quantized": quant_answer_logprob_sum / answer_token_count,
            "answer_token_logprob_delta": (quant_answer_logprob_sum - dense_answer_logprob_sum) / answer_token_count,
            "answer_token_logprob_retention": (
                quant_answer_logprob_sum / dense_answer_logprob_sum
                if dense_answer_logprob_sum != 0.0 else None
            ),
            "answer_token_logprob_dense_per_example": dense_answer_logprob_sum / len(rows),
            "answer_token_logprob_quantized_per_example": quant_answer_logprob_sum / len(rows),
            "answer_token_margin_dense": dense_answer_margin_sum / answer_token_count,
            "answer_token_margin_quantized": quant_answer_margin_sum / answer_token_count,
            "answer_token_margin_delta": (quant_answer_margin_sum - dense_answer_margin_sum) / answer_token_count,
            "answer_token_margin_retention": (
                quant_answer_margin_sum / dense_answer_margin_sum
                if dense_answer_margin_sum != 0.0 else None
            ),
            "critical_token_top1_dense_target": critical_dense_target / critical_count if critical_count else None,
            "critical_token_top1_quantized_target": critical_quant_target / critical_count if critical_count else None,
            "critical_token_top1_quantized_vs_dense": critical_quant_dense / critical_count if critical_count else None,
            "critical_token_top1_delta_vs_dense_target": (
                (critical_quant_target - critical_dense_target) / critical_count
                if critical_count else None
            ),
            "critical_token_count": critical_count,
            "mini_math_accuracy_delta_vs_dense": (quant_rollout_correct - dense_rollout_correct) / len(rows),
            "paired_dense_wrong_quantized_right": dense_wrong_quant_right,
            "paired_dense_right_quantized_wrong": dense_right_quant_wrong,
        },
        "counts": {
            "rows": len(rows),
            "teacher_forced_tokens": teacher_forced_tokens,
            "answer_tokens": answer_token_count,
            "dense_rollout_correct": dense_rollout_correct,
            "quantized_rollout_correct": quant_rollout_correct,
        },
        "seconds": {
            "dense_load": dense_load_seconds,
            "quantized_load": quantized_load_seconds,
            "dense_forward": dense_forward_seconds,
            "quantized_forward": quantized_forward_seconds,
            "dense_rollout": dense_rollout_seconds,
            "quantized_rollout": quantized_rollout_seconds,
        },
        "rows": per_row,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _publish_snapshot_evaluation(checkpoint, output_path, result, kind="micro_math")
    print(json.dumps({"metrics": result["metrics"], "counts": result["counts"]}, indent=2, sort_keys=True), flush=True)
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
    validate_divergence_manifest_binding(checkpoint, dataset_path)
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
            "eos_policy": "EOS is an ordinary token; exactly 32 argmax steps are evaluated",
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
    _publish_snapshot_evaluation(checkpoint, output_path, result, kind="divergence300")
    print(json.dumps({key: value for key, value in result.items() if key != "prompts"}, indent=2, sort_keys=True))
    return 0


def _tasks(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    from tests.eval import evaluate, format_eval_result_table, get_eval_task_results

    evalution_version = package_version("evalution")
    try:
        version_tuple = tuple(int(part) for part in evalution_version.split(".")[:3])
    except ValueError as exc:
        raise RuntimeError(f"Cannot validate Evalution version {evalution_version!r}") from exc
    if version_tuple < (0, 0, 17):
        raise RuntimeError(
            "tasks evaluation requires Evalution>=0.0.17 for automatic loglikelihood prefix "
            "caching and Transformers ContinuousBatchingConfig "
            f"graph routing; found Evalution=={evalution_version}"
        )

    graph_request, graph_mode = _resolve_cuda_graph_request(args)

    selected = args.task or list(TASKS)
    expected_payload: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "backend": args.backend,
        "dtype": "float16",
        "batch_size": args.batch_size,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "continuous_batching_required": True,
        "paged_attention_required": True,
        "cuda_graph_mode": graph_mode,
        "cuda_graph_requested": list(graph_request) if graph_request is not None else None,
        "loglikelihood_prefix_cache": args.loglikelihood_prefix_cache,
        "loglikelihood_prefix_cache_prewarm": args.loglikelihood_prefix_cache_prewarm,
        "loglikelihood_prefix_cache_prewarm_batch_size": args.loglikelihood_prefix_cache_prewarm_batch_size,
        "loglikelihood_prefix_cache_release_after_group": args.loglikelihood_prefix_cache_release_after_group,
        "loglikelihood_prefix_cache_min_tokens": args.loglikelihood_prefix_cache_min_tokens,
        "loglikelihood_prefix_cache_max_entries": args.loglikelihood_prefix_cache_max_entries,
        "max_rows": args.max_rows,
        "package_versions": {
            "evalution": package_version("evalution"),
            "gptqmodel": package_version("gptqmodel"),
            "logbar": package_version("logbar"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "tasks": {},
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        for key, expected in expected_payload.items():
            if key == "tasks":
                continue
            if payload.get(key) != expected:
                raise ValueError(
                    f"Cannot resume incompatible task report: `{key}` is {payload.get(key)!r}, expected {expected!r}"
                )
        if not isinstance(payload.get("tasks"), dict):
            raise ValueError("Cannot resume task report with a non-object `tasks` field")
    else:
        payload = expected_payload

    def publish() -> None:
        payload["updated_at_utc"] = datetime.now(UTC).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        _publish_snapshot_evaluation(checkpoint, args.output, payload, kind="tasks")

    for label in selected:
        if label in payload["tasks"]:
            print(f"Skipping completed task {label} from resumed report {args.output}", flush=True)
            continue
        task, apply_chat_template, suite_kwargs = TASKS[label]
        suite_kwargs = dict(suite_kwargs)
        if args.max_rows is not None:
            suite_kwargs["max_rows"] = args.max_rows
        started = time.perf_counter()
        with _mmlu_question_row_progress(task.startswith("mmlu")):
            output = evaluate(
                model_or_id_or_path=str(checkpoint),
                tasks=[task],
                backend=BACKEND(args.backend),
                model_args={
                    "dtype": "float16",
                    "device": args.device,
                    "attn_implementation": args.attn_implementation,
                    "use_cuda_graph": graph_request,
                    **{
                        key: value
                        for key, value in {
                            "allow_block_sharing": args.allow_block_sharing,
                            "max_batch_tokens": args.max_batch_tokens,
                            "max_blocks_per_request": args.max_blocks_per_request,
                            "use_async_batching": args.use_async_batching,
                            "q_padding_interval_size": args.q_padding_interval_size,
                            "kv_padding_interval_size": args.kv_padding_interval_size,
                            "max_cached_graphs": args.max_cached_graphs,
                            "loglikelihood_prefix_cache": args.loglikelihood_prefix_cache,
                            "loglikelihood_prefix_cache_prewarm": args.loglikelihood_prefix_cache_prewarm,
                            "loglikelihood_prefix_cache_prewarm_batch_size": args.loglikelihood_prefix_cache_prewarm_batch_size,
                            "loglikelihood_prefix_cache_release_after_group": args.loglikelihood_prefix_cache_release_after_group,
                            "loglikelihood_prefix_cache_min_tokens": args.loglikelihood_prefix_cache_min_tokens,
                            "loglikelihood_prefix_cache_max_entries": args.loglikelihood_prefix_cache_max_entries,
                        }.items()
                        if value is not None
                    },
                },
                batch_size=args.batch_size,
                apply_chat_template=apply_chat_template,
                gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
                suite_kwargs=suite_kwargs,
                trust_remote_code=False,
            )
        print(format_eval_result_table(output), flush=True)
        metrics = get_eval_task_results(output)
        engine = output.get("engine") if isinstance(output, dict) else None
        payload["tasks"][label] = {
            "evalution_task": task,
            "seconds": time.perf_counter() - started,
            "metrics": next(iter(metrics.values())) if metrics else {},
            "engine": engine if isinstance(engine, dict) else {},
        }
        publish()
        print(f"Published completed task {label} to {args.output}", flush=True)
    if not args.output.exists():
        publish()
    return 0


def _resolve_cuda_graph_request(args: argparse.Namespace) -> tuple[tuple[bool, bool] | None, str]:
    """Resolve the CLI graph policy into Transformers' (varlen, decode) tuple."""

    legacy = getattr(args, "use_cuda_graph", None)
    mode = getattr(args, "cuda_graph_mode", None)
    if legacy is not None and mode is not None:
        raise ValueError("pass only one of --use-cuda-graph/--no-use-cuda-graph or --cuda-graph-mode")
    if legacy is not None:
        return (bool(legacy), bool(legacy)), "both" if legacy else "off"
    mode = mode or "decode"
    requests = {
        "auto": None,
        "off": (False, False),
        "varlen": (True, False),
        "decode": (False, True),
        "both": (True, True),
    }
    return requests[mode], mode


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnostics":
        return _diagnostics(args)
    if args.command == "micro_math":
        return _micro_math(args)
    if args.command == "divergence300":
        return _divergence300(args)
    return _tasks(args)


if __name__ == "__main__":
    raise SystemExit(main())
