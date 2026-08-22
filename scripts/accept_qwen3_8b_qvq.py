#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Materialize and gate Qwen3-8B-Instruct QVQ W2/W2.1 acceptance evidence.

The ``evaluate`` command must be run after quantization and a fresh
``GPTQModel.load``. Per-cell evidence is an isolated intervention: a hook
replaces one dense projection output with the freshly reloaded QVQ projection
evaluated on the same dense input, then measures the resulting final logits.
Thus per-cell ``final_kl_nats`` is direct final-logit evidence, not local KL.

``diverse_32`` is the fraction of exactly 32 predeclared, pairwise-disjoint
prompts whose last-token top-1 prediction matches the dense model. It is
computed globally and for every isolated layer/role intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.qvq_acceptance import (
    MANIFEST_SPLITS,
    REPORT_SCHEMA_VERSION,
    account_serialized_state,
    census_reloaded_model,
    materialize_manifest,
    validate_acceptance_report,
    validate_manifest_disjointness,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    files = sorted(path for path in checkpoint.iterdir() if path.is_file() and path.suffix in {".json", ".safetensors"})
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError("checkpoint has no serialized safetensors artifact")
    result = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return result


def _manifest(args: argparse.Namespace) -> int:
    materialize_manifest(args.split, _read_jsonl(args.input), args.output, expected_count=args.count)
    return 0


def _dataset_content(row: Mapping[str, Any]) -> Any:
    for key in ("text", "prompt", "content", "messages"):
        value = row.get(key)
        if (isinstance(value, str) and value.strip()) or (isinstance(value, list) and value):
            return value
    raise ValueError("dataset row has no nonempty text, prompt, content, or messages field")


def _export_frozen_splits(args: argparse.Namespace) -> int:
    """Export stable JSONL inputs and manifests from one pinned dataset revision."""

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, revision=args.dataset_revision)
    ranges = {
        "calibration": (0, 512),
        "yaqa_tuning": (512, 512),
        "validation": (1024, 512),
        "held_out_diagnostics": (1536, 512),
    }
    if len(dataset) < 2560:
        raise ValueError(f"frozen split plan requires at least 2560 rows, dataset has {len(dataset)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, (start, count) in ranges.items():
        records = [
            {
                "identity": f"hf:{args.dataset}@{args.dataset_revision}:{args.dataset_config}:{args.dataset_split}:{row}",
                "content": _dataset_content(dict(dataset[row])),
            }
            for row in range(start, start + count)
        ]
        jsonl = args.output_dir / f"{split}.jsonl"
        jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        materialize_manifest(split, records, args.output_dir / f"{split}.manifest.json", expected_count=count)

    # Diverse-32 is deterministic length-stratified coverage of a separate
    # 512-row pool. Sort by canonical UTF-8 content length then stable row ID,
    # partition into 32 bins of 16, and take each bin midpoint (rank 8).
    pool = [(len(json.dumps(_dataset_content(dict(dataset[row])), sort_keys=True).encode()), row) for row in range(2048, 2560)]
    pool.sort()
    selected = [pool[bin_index * 16 + 8][1] for bin_index in range(32)]
    records = [
        {
            "identity": f"hf:{args.dataset}@{args.dataset_revision}:{args.dataset_config}:{args.dataset_split}:{row}",
            "content": _dataset_content(dict(dataset[row])),
        }
        for row in selected
    ]
    jsonl = args.output_dir / "diverse_32.jsonl"
    jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    materialize_manifest("diverse_32", records, args.output_dir / "diverse_32.manifest.json", expected_count=32)
    _check_manifests(args.output_dir)
    return 0


def _check_manifests(directory: Path) -> dict[str, Any]:
    manifests = {
        split: json.loads((directory / f"{split}.manifest.json").read_text(encoding="utf-8"))
        for split in MANIFEST_SPLITS
    }
    return validate_manifest_disjointness(manifests)


def _verify_records_against_manifest(split: str, records: list[dict[str, Any]], directory: Path) -> None:
    expected = json.loads((directory / f"{split}.manifest.json").read_text(encoding="utf-8"))
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary:
        actual = materialize_manifest(split, records, Path(temporary) / "manifest.json")
    if actual != expected:
        raise RuntimeError(f"{split} evaluation JSONL does not match its accepted identity/content manifest")


def _verify_quantization_streams(checkpoint: Path, manifest_dir: Path) -> dict[str, Any]:
    path = checkpoint / "qvq_quantize_run.json"
    if not path.is_file():
        raise RuntimeError("checkpoint is missing qvq_quantize_run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_files = {
        "calibration": manifest_dir / "calibration.jsonl",
        "yaqa": manifest_dir / "yaqa_tuning.jsonl",
        "validation": manifest_dir / "validation.jsonl",
    }
    evidence = {}
    for split, expected_path in expected_files.items():
        raw = payload.get("datasets", {}).get(split)
        if not isinstance(raw, dict):
            raise TypeError(f"quantization report lacks explicit {split} stream")
        actual_path = Path(str(raw.get("source", ""))).expanduser().resolve()
        if actual_path != expected_path.resolve() or raw.get("row_start") != 0 or raw.get("rows") != 512:
            raise RuntimeError(f"quantization {split} stream does not match the frozen manifest JSONL")
        evidence[split] = {"source": str(actual_path), "rows": 512, "manifest_verified": True}
    if payload.get("layer_scope") != "all":
        raise RuntimeError("quantization report did not request all decoder layers")
    return evidence


def _gate(args: argparse.Namespace) -> int:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_acceptance_report(report)
    print(json.dumps({"accepted": True, "report": str(args.report)}, sort_keys=True))
    return 0


def _extract_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("causal language model did not return tensor logits")
    return logits


def _encode(tokenizer, content: Any, device: str) -> dict[str, torch.Tensor]:
    if isinstance(content, list):
        text = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=False)
    elif isinstance(content, str) and content.strip():
        text = content
    else:
        raise ValueError("evaluation content must be nonempty text or a chat-message list")
    return {key: value.to(device) for key, value in tokenizer(text, return_tensors="pt").items()}


def _last_logits(model, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = _extract_logits(model(**encoded, use_cache=False))
    mask = encoded.get("attention_mask")
    position = int(mask.sum().item()) - 1 if mask is not None else logits.shape[1] - 1
    return logits[0, position].float()


def _metric_accumulator() -> dict[str, float]:
    return {"top1": 0.0, "kl": 0.0, "count": 0.0, "diverse_matches": 0.0, "diverse_count": 0.0}


def _add_metric(acc: dict[str, float], dense: torch.Tensor, candidate: torch.Tensor, *, diverse: bool) -> None:
    if dense.shape != candidate.shape or not torch.isfinite(candidate).all():
        raise RuntimeError("candidate final logits are non-finite or shape-incompatible")
    acc["top1"] += float(dense.argmax() == candidate.argmax())
    acc["kl"] += float(F.kl_div(candidate.log_softmax(-1), dense.softmax(-1), reduction="sum").item())
    acc["count"] += 1
    if diverse:
        acc["diverse_matches"] += float(dense.argmax() == candidate.argmax())
        acc["diverse_count"] += 1


def _finish(acc: Mapping[str, float]) -> dict[str, Any]:
    if acc["count"] <= 0 or acc["diverse_count"] != 32:
        raise RuntimeError(f"incomplete metric coverage: {dict(acc)}")
    return {
        "coverage_complete": True,
        "sample_count": int(acc["count"]),
        "diverse_32_sample_count": int(acc["diverse_count"]),
        "top1_agreement": acc["top1"] / acc["count"],
        "final_kl_nats": acc["kl"] / acc["count"],
        "diverse_32": acc["diverse_matches"] / 32,
    }


@contextmanager
def _projection_intervention(dense_module: torch.nn.Module, quant_module: torch.nn.Module) -> Iterator[None]:
    def replace(_module, inputs, _output):
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("projection hook requires exactly one tensor input")
        return quant_module(inputs[0])

    handle = dense_module.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    disjointness = _check_manifests(args.manifest_dir)
    validation = _read_jsonl(args.validation_jsonl)
    diverse = _read_jsonl(args.diverse_jsonl)
    if not validation or len(diverse) != 32:
        raise ValueError("validation must be nonempty and diverse JSONL must contain exactly 32 records")
    _verify_records_against_manifest("validation", validation, args.manifest_dir)
    _verify_records_against_manifest("diverse_32", diverse, args.manifest_dir)
    quantization_streams = _verify_quantization_streams(args.checkpoint, args.manifest_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, revision=args.revision, local_files_only=True)
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        revision=args.revision,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    # This load, not the in-memory quantizer output, is the acceptance authority.
    quantized = GPTQModel.load(
        str(args.checkpoint.resolve()),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    cells = census_reloaded_model(quantized)
    accounting = account_serialized_state(quantized.state_dict(), cells, maximum_bpw=args.maximum_bpw)
    dense_modules = dict(dense.named_modules())
    quant_modules = dict(quantized.named_modules())
    global_acc = _metric_accumulator()
    cell_acc = {(cell.layer, cell.role): _metric_accumulator() for cell in cells}

    for is_diverse, records in ((False, validation), (True, diverse)):
        for record in records:
            encoded = _encode(tokenizer, record["content"], args.device)
            dense_logits = _last_logits(dense, encoded)
            _add_metric(global_acc, dense_logits, _last_logits(quantized, encoded), diverse=is_diverse)
            for cell in cells:
                dense_module = dense_modules.get(cell.name)
                quant_module = quant_modules.get(cell.name)
                if dense_module is None or quant_module is None:
                    raise RuntimeError(f"fresh models do not share requested projection path {cell.name}")
                with _projection_intervention(dense_module, quant_module):
                    intervened_logits = _last_logits(dense, encoded)
                _add_metric(cell_acc[(cell.layer, cell.role)], dense_logits, intervened_logits, diverse=is_diverse)

    thresholds = {
        "top1_agreement_min": args.score_min,
        "diverse_32_min": args.score_min,
        "final_kl_max_nats": args.final_kl_max_nats,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact": {
            "checkpoint": str(args.checkpoint.resolve()),
            "dense_model": args.dense_model,
            "revision": args.revision,
            "fresh_reload_verified": True,
            "checkpoint_sha256": _checkpoint_hashes(args.checkpoint),
        },
        "census": {"expected": 252, "actual": len(cells), "complete": True},
        "accounting": accounting,
        "manifests": {**disjointness, "quantization_streams": quantization_streams},
        "metric_semantics": {
            "top1_agreement": "fraction of last-token argmax IDs equal to the dense model",
            "final_kl_nats": "mean KL(dense final-logit distribution || candidate final-logit distribution), in nats",
            "diverse_32": "fraction of exactly 32 predeclared diverse prompts with matching last-token top-1",
            "cell_evidence": "isolated dense-model projection-output replacement using the reloaded QVQ module",
        },
        "thresholds": thresholds,
        "global": _finish(global_acc),
        "cells": [
            {"layer": cell.layer, "role": cell.role, "module": cell.name, **_finish(cell_acc[(cell.layer, cell.role)])}
            for cell in cells
        ],
    }
    validate_acceptance_report(report)
    _write_new(args.output, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--split", choices=MANIFEST_SPLITS, required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--count", type=int)
    manifest.set_defaults(handler=_manifest)
    export = commands.add_parser("export-frozen-splits")
    export.add_argument("--dataset", required=True)
    export.add_argument("--dataset-config")
    export.add_argument("--dataset-split", default="train")
    export.add_argument("--dataset-revision", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.set_defaults(handler=_export_frozen_splits)
    gate = commands.add_parser("gate")
    gate.add_argument("--report", type=Path, required=True)
    gate.set_defaults(handler=_gate)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dense-model", required=True)
    evaluate.add_argument("--revision", required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--manifest-dir", type=Path, required=True)
    evaluate.add_argument("--validation-jsonl", type=Path, required=True)
    evaluate.add_argument("--diverse-jsonl", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--maximum-bpw", type=float, default=2.1)
    evaluate.add_argument("--score-min", type=float, default=0.85)
    evaluate.add_argument("--final-kl-max-nats", type=float, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
