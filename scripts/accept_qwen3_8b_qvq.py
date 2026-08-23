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
import math
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
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
    QWEN3_DENSE_ARTIFACT_SHA256,
    QWEN3_PINNED_REVISION,
    REPORT_SCHEMA_VERSION,
    AcceptanceError,
    ProjectionCell,
    acceptance_observation_digest,
    account_serialized_checkpoint,
    census_reloaded_model,
    expected_cells,
    expected_projection_dimensions,
    expected_projection_name,
    hash_canonical_qwen3_payloads,
    materialize_manifest,
    select_diverse_32,
    validate_acceptance_report,
    validate_diverse_selection,
    validate_manifest_disjointness,
    validate_qwen3_model_artifact,
)
from gptqmodel.utils.qvq_acceptance_controller import (
    CONTROLLER_STAGES,
    AcceptanceController,
    controller_authority_receipt,
    controller_environment,
    emit_controller_measurement,
    trusted_python_executable,
    verify_controller_signature,
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _artifact_hashes(directory: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    )
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise RuntimeError(f"artifact has no serialized safetensors files: {directory}")
    result = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"failed to parse {label} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"{label} must contain one JSON object: {path}")
    return payload


def _manifest(args: argparse.Namespace) -> int:
    materialize_manifest(
        args.split, _read_jsonl(args.input), args.output, expected_count=args.count
    )
    return 0


def _dataset_content(row: Mapping[str, Any]) -> Any:
    for key in ("text", "prompt", "content", "messages"):
        value = row.get(key)
        if (isinstance(value, str) and value.strip()) or (
            isinstance(value, list) and value
        ):
            return value
    raise ValueError(
        "dataset row has no nonempty text, prompt, content, or messages field"
    )


def _export_frozen_splits(args: argparse.Namespace) -> int:
    """Export stable JSONL inputs and manifests from one pinned dataset revision."""

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
    )
    ranges = {
        "calibration": (0, 512),
        "yaqa_tuning": (512, 512),
        "validation": (1024, 512),
        "held_out_diagnostics": (1536, 512),
    }
    if len(dataset) < 2560:
        raise ValueError(
            f"frozen split plan requires at least 2560 rows, dataset has {len(dataset)}"
        )
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
        jsonl.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        materialize_manifest(
            split,
            records,
            args.output_dir / f"{split}.manifest.json",
            expected_count=count,
        )

    pool_records = [
        {
            "identity": f"hf:{args.dataset}@{args.dataset_revision}:{args.dataset_config}:{args.dataset_split}:{row}",
            "content": _dataset_content(dict(dataset[row])),
        }
        for row in range(2048, 2560)
    ]
    pool_jsonl = args.output_dir / "diverse_pool_512.jsonl"
    pool_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in pool_records),
        encoding="utf-8",
    )
    materialize_manifest(
        "diverse_pool_512",
        pool_records,
        args.output_dir / "diverse_pool_512.manifest.json",
        expected_count=512,
    )
    records = select_diverse_32(pool_records)
    jsonl = args.output_dir / "diverse_32.jsonl"
    jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    materialize_manifest(
        "diverse_32",
        records,
        args.output_dir / "diverse_32.manifest.json",
        expected_count=32,
    )
    _check_manifests(args.output_dir)
    return 0


def _check_manifests(directory: Path) -> dict[str, Any]:
    manifests = {
        split: _read_json_object(
            directory / f"{split}.manifest.json", f"{split} identity manifest"
        )
        for split in MANIFEST_SPLITS
    }
    evidence = validate_manifest_disjointness(manifests)
    evidence["diverse_selection"] = validate_diverse_selection(
        _read_jsonl(directory / "diverse_pool_512.jsonl"),
        _read_jsonl(directory / "diverse_32.jsonl"),
    )
    evidence["manifest_sha256"] = {
        split: _sha256_file(directory / f"{split}.manifest.json")
        for split in MANIFEST_SPLITS
    }
    evidence["content_jsonl_sha256"] = {
        split: _sha256_file(directory / f"{split}.jsonl") for split in MANIFEST_SPLITS
    }
    return evidence


def _verify_all_manifest_sources(directory: Path) -> None:
    for split in MANIFEST_SPLITS:
        _verify_records_against_manifest(
            split, _read_jsonl(directory / f"{split}.jsonl"), directory
        )


def _verify_records_against_manifest(
    split: str, records: list[dict[str, Any]], directory: Path
) -> None:
    expected = _read_json_object(
        directory / f"{split}.manifest.json", f"{split} identity manifest"
    )

    with TemporaryDirectory() as temporary:
        actual = materialize_manifest(split, records, Path(temporary) / "manifest.json")
    if actual != expected:
        raise RuntimeError(
            f"{split} evaluation JSONL does not match its accepted identity/content manifest"
        )


def _verify_quantization_streams(
    checkpoint: Path, manifest_dir: Path, dense_model: Path
) -> dict[str, Any]:
    path = checkpoint / "qvq_quantize_run.json"
    if not path.is_file():
        raise RuntimeError("checkpoint is missing qvq_quantize_run.json")
    payload = _read_json_object(path, "qvq_quantize_run.json")
    recorded_model = Path(str(payload.get("model", ""))).expanduser().resolve()
    if recorded_model != dense_model.resolve():
        raise RuntimeError(
            "quantization report dense-model path does not match the acceptance reference"
        )
    quantize_config = payload.get("quantize_config")
    if not isinstance(quantize_config, dict):
        raise TypeError("quantization report lacks a resolved quantize_config")
    expected_config = {
        "bits": 2,
        "format": "qvq_v2b2_p32",
        "group_size": -1,
        "rounding": "yaqa",
        "sym": True,
        "pack_dtype": "int32",
        "bank_count": 2,
    }
    mismatches = {
        key: {"expected": expected, "actual": quantize_config.get(key)}
        for key, expected in expected_config.items()
        if quantize_config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"quantization report does not match the frozen W2 config: {mismatches}"
        )
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
        if (
            actual_path != expected_path.resolve()
            or raw.get("row_start") != 0
            or raw.get("rows") != 512
        ):
            raise RuntimeError(
                f"quantization {split} stream does not match the frozen manifest JSONL"
            )
        manifest_path = expected_path.with_suffix(".manifest.json")
        current_content_hash = _sha256_file(actual_path)
        current_manifest_hash = _sha256_file(manifest_path)
        if raw.get("content_sha256") != current_content_hash:
            raise RuntimeError(
                f"quantization {split} content hash does not match the frozen JSONL"
            )
        if (
            Path(str(raw.get("identity_manifest", ""))).expanduser().resolve()
            != manifest_path.resolve()
        ):
            raise RuntimeError(
                f"quantization {split} identity-manifest path is not authoritative"
            )
        if raw.get("identity_manifest_sha256") != current_manifest_hash:
            raise RuntimeError(
                f"quantization {split} identity-manifest hash does not match"
            )
        evidence[split] = {
            "source": str(actual_path),
            "rows": 512,
            "content_sha256": current_content_hash,
            "identity_manifest": str(manifest_path.resolve()),
            "identity_manifest_sha256": current_manifest_hash,
            "manifest_verified": True,
        }
    if payload.get("layer_scope") != "all":
        raise RuntimeError("quantization report did not request all decoder layers")
    binding = payload.get("dense_source_binding")
    if not isinstance(binding, dict):
        raise TypeError("quantization report lacks dense source start/end binding")
    start, end = binding.get("start"), binding.get("end")
    if (
        not isinstance(start, dict)
        or not isinstance(end, dict)
        or start.get("stage") != "quantization_start"
        or end.get("stage") != "quantization_end"
        or start.get("controller_run_nonce") != binding.get("controller_run_nonce")
        or end.get("controller_run_nonce") != binding.get("controller_run_nonce")
        or start.get("process_instance_id") != binding.get("producer_process_instance_id")
        or end.get("process_instance_id") != binding.get("producer_process_instance_id")
        or start.get("producer_stage_nonce") != binding.get("producer_stage_nonce")
        or end.get("producer_stage_nonce") != binding.get("producer_stage_nonce")
        or start.get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or end.get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or start.get("observation_sha256") != acceptance_observation_digest(start)
        or end.get("observation_sha256") != acceptance_observation_digest(end)
        or end.get("previous_observation_sha256") != start.get("observation_sha256")
    ):
        raise RuntimeError("quantization report dense source binding is mismatched, missing, or unpinned")
    parity = payload.get("packed_payload_parity")
    partial = payload.get("acceptance_controller_partial")
    if payload.get("controller_pending_evaluation") is True:
        if (
            not isinstance(parity, dict)
            or not isinstance(partial, dict)
            or not verify_controller_signature(partial)
            or tuple(record.get("stage") for record in partial.get("processes", ())) != CONTROLLER_STAGES[:2]
            or parity.get("controller_pending") is not True
        ):
            raise RuntimeError("quantization report lacks controller-observed producer/reload authority")
        evidence["packed_payload_parity"] = parity
        evidence["dense_source_binding"] = binding
        evidence["quantize_config"] = expected_config
        return evidence
    raise RuntimeError("quantization report is not an active controller-owned evaluation handoff")


@torch.inference_mode()
def _payload_hashes(args: argparse.Namespace) -> int:
    controller = controller_environment()
    if controller["stage"] != "fresh_process_reload":
        raise RuntimeError("payload reload must be spawned by the independent controller")
    model = GPTQModel.load(
        str(args.checkpoint.resolve()),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": args.device},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    payload = hash_canonical_qwen3_payloads(model, census_reloaded_model(model))
    emit_controller_measurement("fresh_process_reload", {"payload": payload})
    _write_new(args.output, payload)
    return 0


def _controller_parity(records: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    producer, reload = records[:2]
    parity = {
        "verified": complete,
        "controller_run_nonce": producer["event"]["run_nonce"],
        "pre_save": {
            **producer["event"]["measurement"]["pre_save"],
            "stage": producer["stage"],
            "stage_nonce": producer["stage_nonce"],
            "process_instance_id": producer["process_instance_id"],
        },
        "fresh_process": {
            **reload["event"]["measurement"],
            "stage": reload["stage"],
            "stage_nonce": reload["stage_nonce"],
            "process_instance_id": reload["process_instance_id"],
        },
    }
    if complete:
        evaluation = records[2]
        parity["evaluation_reload"] = {
            "payload": evaluation["event"]["measurement"]["payload"],
            "stage": evaluation["stage"],
            "stage_nonce": evaluation["stage_nonce"],
            "process_instance_id": evaluation["process_instance_id"],
        }
    else:
        parity["controller_pending"] = True
    return parity


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _controlled_run(args: argparse.Namespace) -> int:
    """Run producer, reload, and evaluation as controller-owned process instances."""

    if args.output.exists() or args.controller_authority_output.exists():
        raise FileExistsError("refusing to overwrite acceptance report or controller authority receipt")
    controller = AcceptanceController()
    trusted_python = trusted_python_executable()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    producer_command = [
        trusted_python,
        str(REPO_ROOT / "scripts" / "qvq_quantize.py"),
        "--model",
        args.dense_model,
        "--output",
        str(checkpoint),
        "--quant-config",
        str(args.quant_config.expanduser().resolve()),
        "--calibration-dataset",
        str(manifest_dir / "calibration.jsonl"),
        "--calibration-rows",
        "512",
        "--yaqa-dataset",
        str(manifest_dir / "yaqa_tuning.jsonl"),
        "--yaqa-rows",
        "512",
        "--validation-dataset",
        str(manifest_dir / "validation.jsonl"),
        "--validation-rows",
        "512",
        "--device",
        args.device,
        "--verify-qwen3-acceptance-payload-parity",
    ]
    producer = controller.spawn_stage(
        "quantization_producer", producer_command, cwd=REPO_ROOT, timeout=args.controller_timeout
    )
    with TemporaryDirectory(prefix="qwen3-controller-") as temporary:
        temporary_path = Path(temporary)
        reload_output = temporary_path / "reload.json"
        reload_command = [
            trusted_python,
            str(REPO_ROOT / "scripts" / "accept_qwen3_8b_qvq.py"),
            "payload-hashes",
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--output",
            str(reload_output),
        ]
        reload = controller.spawn_stage(
            "fresh_process_reload", reload_command, cwd=REPO_ROOT, timeout=args.controller_timeout
        )
        if producer["event"]["measurement"]["pre_save"]["payload"] != reload["event"]["measurement"]["payload"]:
            raise AcceptanceError("controller observed packed payload drift between live producer and fresh reload")
        run_path = checkpoint / "qvq_quantize_run.json"
        run = _read_json_object(run_path, "controller-owned quantization report")
        run["dense_source_binding"] = producer["event"]["measurement"]["dense_source_binding"]
        run["packed_payload_parity"] = _controller_parity([producer, reload], complete=False)
        run["acceptance_controller_partial"] = controller.signed_transcript(require_complete=False)
        run["controller_pending_evaluation"] = True
        _replace_json(run_path, run)

        draft_output = temporary_path / "acceptance-draft.json"
        evaluation_command = [
            trusted_python,
            str(REPO_ROOT / "scripts" / "accept_qwen3_8b_qvq.py"),
            "evaluate",
            "--dense-model",
            args.dense_model,
            "--revision",
            args.revision,
            "--checkpoint",
            str(checkpoint),
            "--manifest-dir",
            str(manifest_dir),
            "--validation-jsonl",
            str(args.validation_jsonl.expanduser().resolve()),
            "--held-out-diagnostics-jsonl",
            str(args.held_out_diagnostics_jsonl.expanduser().resolve()),
            "--diverse-jsonl",
            str(args.diverse_jsonl.expanduser().resolve()),
            "--device",
            args.device,
            "--maximum-bpw",
            str(args.maximum_bpw),
            "--score-min",
            str(args.score_min),
            "--final-kl-max-nats",
            str(args.final_kl_max_nats),
            "--output",
            str(draft_output),
        ]
        evaluation = controller.spawn_stage(
            "acceptance_evaluation", evaluation_command, cwd=REPO_ROOT, timeout=args.controller_timeout
        )
        transcript = controller.signed_transcript()
        final_parity = _controller_parity([producer, reload, evaluation], complete=True)
        run["packed_payload_parity"] = final_parity
        run["acceptance_controller"] = transcript
        run.pop("acceptance_controller_partial", None)
        run.pop("controller_pending_evaluation", None)
        _replace_json(run_path, run)

        report = _read_json_object(draft_output, "controller-owned acceptance draft")
        report["artifact"]["packed_payload_parity"] = final_parity
        report["artifact"]["dense_source_binding"] = run["dense_source_binding"]
        report["artifact"]["acceptance_controller"] = transcript
        report["artifact"]["checkpoint_sha256"] = _artifact_hashes(checkpoint)
        authority = controller_authority_receipt(transcript)
        validate_acceptance_report(report, controller_authority=authority)
        _write_new(args.output, report)
        _write_new(args.controller_authority_output, authority)
    return 0


def _gate(args: argparse.Namespace) -> int:
    submitted = _read_json_object(args.report, "submitted acceptance report")
    controller_authority = _read_json_object(args.controller_authority, "controller authority receipt")
    validate_acceptance_report(submitted, controller_authority=controller_authority)
    dense = validate_qwen3_model_artifact(Path(args.dense_model), require_pinned_dense=True)
    checkpoint = validate_qwen3_model_artifact(args.checkpoint, require_pinned_dense=False)
    cells = [
        ProjectionCell(
            layer,
            role,
            expected_projection_name(layer, role),
            *expected_projection_dimensions(role),
            ("SU", "SV", "bank_alt_id", "bank_ids", "trellis"),
        )
        for layer, role in expected_cells()
    ]
    recomputed_accounting = account_serialized_checkpoint(
        args.checkpoint, cells, maximum_bpw=args.maximum_bpw
    )
    if (
        submitted["artifact"]["dense_model_identity"] != dense
        or submitted["artifact"]["checkpoint_model_identity"] != checkpoint
        or submitted["artifact"]["checkpoint_sha256"] != _artifact_hashes(args.checkpoint)
        or submitted["accounting"] != recomputed_accounting
        or submitted["manifests"] | {"quantization_streams": None}
        != _check_manifests(args.manifest_dir) | {"quantization_streams": None}
    ):
        raise AcceptanceError("submitted report does not match controller-bound artifact recomputation")
    print(
        json.dumps(
            {"accepted": True, "report": str(args.report), "artifact_recomputed": True},
            sort_keys=True,
        )
    )
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
        text = tokenizer.apply_chat_template(
            content, tokenize=False, add_generation_prompt=False
        )
    elif isinstance(content, str) and content.strip():
        text = content
    else:
        raise ValueError(
            "evaluation content must be nonempty text or a chat-message list"
        )
    return {
        key: value.to(device)
        for key, value in tokenizer(text, return_tensors="pt").items()
    }


def _last_logits(model, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = _extract_logits(model(**encoded, use_cache=False))
    mask = encoded.get("attention_mask")
    position = int(mask.sum().item()) - 1 if mask is not None else logits.shape[1] - 1
    return logits[0, position].float()


def _metric_accumulator() -> dict[str, float]:
    return {
        "top1": 0.0,
        "kl": 0.0,
        "count": 0.0,
        "diverse_matches": 0.0,
        "diverse_count": 0.0,
    }


def _add_metric(
    acc: dict[str, float],
    dense: torch.Tensor,
    candidate: torch.Tensor,
    *,
    diverse: bool,
) -> None:
    if (
        dense.shape != candidate.shape
        or not torch.isfinite(dense).all()
        or not torch.isfinite(candidate).all()
    ):
        raise RuntimeError(
            "dense/candidate final logits are non-finite or shape-incompatible"
        )
    acc["top1"] += float(dense.argmax() == candidate.argmax())
    acc["kl"] += float(
        F.kl_div(candidate.log_softmax(-1), dense.softmax(-1), reduction="sum").item()
    )
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
def _projection_intervention(
    dense_module: torch.nn.Module, quant_module: torch.nn.Module
) -> Iterator[None]:
    def replace(_module, inputs, _output):
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("projection hook requires exactly one tensor input")
        return quant_module(inputs[0])

    handle = dense_module.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


def _resolve_runtime_module(
    modules: Mapping[str, torch.nn.Module],
    canonical_name: str,
    *,
    preferred_name: str | None = None,
) -> torch.nn.Module:
    if preferred_name is not None and preferred_name in modules:
        return modules[preferred_name]
    matches = [
        module
        for name, module in modules.items()
        if name == canonical_name or name.endswith("." + canonical_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one runtime module for {canonical_name!r}, found {len(matches)}"
        )
    return matches[0]


@torch.inference_mode()
def _evaluate(args: argparse.Namespace) -> int:
    controller = controller_environment()
    if controller["stage"] != "acceptance_evaluation":
        raise RuntimeError("acceptance evaluation must be spawned by the independent controller")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        not math.isfinite(args.maximum_bpw)
        or args.maximum_bpw <= 0
        or args.maximum_bpw > 2.1
    ):
        raise ValueError(
            "maximum BPW must be finite, positive, and no greater than 2.1"
        )
    if not math.isfinite(args.score_min) or not 0.85 <= args.score_min <= 1:
        raise ValueError("score minimum must be a finite fraction in [0.85, 1]")
    if not math.isfinite(args.final_kl_max_nats) or args.final_kl_max_nats < 0:
        raise ValueError(
            "final KL maximum must be finite, nonnegative, and expressed in nats"
        )
    if args.revision != QWEN3_PINNED_REVISION:
        raise ValueError(
            f"Qwen3 acceptance revision must be exactly {QWEN3_PINNED_REVISION}"
        )
    disjointness = _check_manifests(args.manifest_dir)
    _verify_all_manifest_sources(args.manifest_dir)
    validation = _read_jsonl(args.validation_jsonl)
    held_out = _read_jsonl(args.held_out_diagnostics_jsonl)
    diverse = _read_jsonl(args.diverse_jsonl)
    if len(validation) != 512 or len(held_out) != 512 or len(diverse) != 32:
        raise ValueError(
            "evaluation requires exactly 512 validation, 512 held-out diagnostic, and 32 diverse records"
        )
    _verify_records_against_manifest("validation", validation, args.manifest_dir)
    _verify_records_against_manifest(
        "held_out_diagnostics", held_out, args.manifest_dir
    )
    _verify_records_against_manifest("diverse_32", diverse, args.manifest_dir)
    dense_model_path = Path(args.dense_model).expanduser().resolve()
    if not dense_model_path.is_dir():
        raise FileNotFoundError(
            f"pinned dense model directory does not exist: {dense_model_path}"
        )
    dense_identity = validate_qwen3_model_artifact(
        dense_model_path, require_pinned_dense=True
    )
    checkpoint_identity = validate_qwen3_model_artifact(
        args.checkpoint, require_pinned_dense=False
    )
    quantization_streams = _verify_quantization_streams(
        args.checkpoint, args.manifest_dir, dense_model_path
    )
    recorded_parity = quantization_streams.pop("packed_payload_parity")
    dense_source_binding = quantization_streams.pop("dense_source_binding")

    tokenizer = AutoTokenizer.from_pretrained(
        args.dense_model, revision=args.revision, local_files_only=True
    )
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
    evaluation_payload = hash_canonical_qwen3_payloads(quantized, cells)
    if evaluation_payload != recorded_parity["fresh_process"]["payload"]:
        raise AcceptanceError(
            "evaluation reload packed payload differs from fresh-process post-save payload"
        )
    packed_payload_parity = {
        **recorded_parity,
        "evaluation_reload": {
            "stage": "acceptance_evaluation",
            "stage_nonce": controller["stage_nonce"],
            "process_instance_id": controller["process_instance_id"],
            "payload": evaluation_payload,
        },
    }
    accounting = account_serialized_checkpoint(
        args.checkpoint, cells, maximum_bpw=args.maximum_bpw
    )
    dense_modules = dict(dense.named_modules())
    quant_modules = dict(quantized.named_modules())
    global_acc = _metric_accumulator()
    cell_acc = {(cell.layer, cell.role): _metric_accumulator() for cell in cells}

    for is_diverse, records in (
        (False, validation),
        (False, held_out),
        (True, diverse),
    ):
        for record in records:
            encoded = _encode(tokenizer, record["content"], args.device)
            dense_logits = _last_logits(dense, encoded)
            _add_metric(
                global_acc,
                dense_logits,
                _last_logits(quantized, encoded),
                diverse=is_diverse,
            )
            for cell in cells:
                dense_module = _resolve_runtime_module(dense_modules, cell.name)
                quant_module = _resolve_runtime_module(
                    quant_modules, cell.name, preferred_name=cell.runtime_name
                )
                with _projection_intervention(dense_module, quant_module):
                    intervened_logits = _last_logits(dense, encoded)
                _add_metric(
                    cell_acc[(cell.layer, cell.role)],
                    dense_logits,
                    intervened_logits,
                    diverse=is_diverse,
                )

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
            "checkpoint_sha256": _artifact_hashes(args.checkpoint),
            "dense_model_sha256": dense_identity["artifact_sha256"],
            "dense_model_identity": dense_identity,
            "checkpoint_model_identity": checkpoint_identity,
            "packed_payload_parity": packed_payload_parity,
            "dense_source_binding": dense_source_binding,
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
            {
                "layer": cell.layer,
                "role": cell.role,
                "module": cell.name,
                **_finish(cell_acc[(cell.layer, cell.role)]),
            }
            for cell in cells
        ],
    }
    acceptance_claim = {
        key: report[key] for key in ("accounting", "manifests", "thresholds", "global", "cells")
    }
    emit_controller_measurement(
        "acceptance_evaluation",
        {
            "payload": evaluation_payload,
            "acceptance_evidence_sha256": hashlib.sha256(
                json.dumps(acceptance_claim, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    )
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
    payload_hashes = commands.add_parser(
        "payload-hashes", help="Fresh-load and hash canonical Qwen3 QVQ payloads."
    )
    payload_hashes.add_argument("--checkpoint", type=Path, required=True)
    payload_hashes.add_argument("--device", default="cuda:0")
    payload_hashes.add_argument("--output", type=Path, required=True)
    payload_hashes.set_defaults(handler=_payload_hashes)

    def add_artifact_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--dense-model", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--manifest-dir", type=Path, required=True)
        command.add_argument("--validation-jsonl", type=Path, required=True)
        command.add_argument("--held-out-diagnostics-jsonl", type=Path, required=True)
        command.add_argument("--diverse-jsonl", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--maximum-bpw", type=float, default=2.1)
        command.add_argument("--score-min", type=float, default=0.85)
        command.add_argument("--final-kl-max-nats", type=float, required=True)

    gate = commands.add_parser(
        "gate", help="Recompute artifacts/evaluation and compare the submitted report."
    )
    add_artifact_arguments(gate)
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--controller-authority", type=Path, required=True)
    gate.set_defaults(handler=_gate)
    evaluate = commands.add_parser("evaluate")
    add_artifact_arguments(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    controlled = commands.add_parser(
        "controlled-run", help="Controller-spawn quantization, reload, and acceptance evaluation."
    )
    add_artifact_arguments(controlled)
    controlled.add_argument("--quant-config", type=Path, required=True)
    controlled.add_argument("--output", type=Path, required=True)
    controlled.add_argument("--controller-authority-output", type=Path, required=True)
    controlled.add_argument("--controller-timeout", type=float)
    controlled.set_defaults(handler=_controlled_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
