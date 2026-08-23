# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed Qwen3-8B QVQ checkpoint acceptance primitives.

This module intentionally contains no model download or quantization side effects.  It
validates evidence produced *after* a fresh checkpoint reload and rejects incomplete
or ambiguous reports.  The executable orchestration lives in
``scripts/accept_qwen3_8b_qvq.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.utils.qvq_acceptance_controller import (
    CONTROLLER_SCHEMA,
    CONTROLLER_STAGES,
    _TrustedResources,
    controller_authority_receipt,
    controller_dataset_evidence,
    validate_required_command,
    verify_controller_signature,
)

QWEN3_8B_LAYER_COUNT = 36
QWEN3_PROJECTION_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN3_EXPECTED_MODULE_COUNT = QWEN3_8B_LAYER_COUNT * len(QWEN3_PROJECTION_ROLES)
MANIFEST_SPLITS = (
    "calibration",
    "yaqa_tuning",
    "validation",
    "held_out_diagnostics",
    "diverse_pool_512",
    "diverse_32",
)
MANIFEST_DISJOINT_PAIRS = tuple(
    (left, right)
    for left_index, left in enumerate(MANIFEST_SPLITS)
    for right in MANIFEST_SPLITS[left_index + 1 :]
    if (left, right) != ("diverse_pool_512", "diverse_32")
)
REPORT_SCHEMA_VERSION = 4
QWEN3_REQUESTED_BITS = 2.0
QWEN3_PINNED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QWEN3_LOCAL_TARGET = Path("/monster/data/model/Qwen3-8B")
QWEN3_DENSE_CONFIG_SHA256 = (
    "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"
)
QWEN3_DENSE_ARTIFACT_SHA256 = {
    "config.json": QWEN3_DENSE_CONFIG_SHA256,
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
}
QWEN3_HUB_CONTENT_IDENTITIES = {
    "config.json": "d46195ac87f837ad233d02b2f80f148bf7c005e0",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "model.safetensors.index.json": "2b85c00f1b118961cd7a477e2bba0fe197a4ce1a",
}
QWEN3_MODEL_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
}
QWEN3_PROJECTION_DIMENSIONS = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 1024),
    "v_proj": (4096, 1024),
    "o_proj": (4096, 4096),
    "gate_proj": (4096, 12288),
    "up_proj": (4096, 12288),
    "down_proj": (12288, 4096),
}
QWEN3_PROJECTION_DENSE_WEIGHT_COUNT = QWEN3_8B_LAYER_COUNT * sum(
    in_features * out_features
    for in_features, out_features in QWEN3_PROJECTION_DIMENSIONS.values()
)
QVQ_PAYLOAD_HASH_SCHEME = "qvq-module-payload-sha256-v2"
QVQ_OBSERVATION_HASH_SCHEME = "qvq-acceptance-observation-sha256-v1"
DIVERSE_SELECTION_SCHEME = "canonical-content-utf8-length/identity-utf8/32-bins-of-16/midpoint-rank-8-v1"
PROJECTION_BOUNDARY = (
    "All serialized tensors whose state-dict key is the exact requested decoder projection prefix or a child of "
    "that prefix. The denominator is the sum of dense in_features*out_features for the 252 requested projections. "
    "Embeddings, norms, LM head, and every other non-target tensor are excluded from BPW and reported separately."
)


class AcceptanceError(RuntimeError):
    """Raised when acceptance evidence is absent, inconsistent, or below contract."""


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class ProjectionCell:
    layer: int
    role: str
    name: str
    in_features: int
    out_features: int
    required_tensors: tuple[str, ...] = ("trellis", "SU", "SV")
    runtime_name: str | None = None

    @property
    def dense_weight_count(self) -> int:
        return self.in_features * self.out_features


def expected_projection_name(layer: int, role: str) -> str:
    stem = "self_attn" if role in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.layers.{layer}.{stem}.{role}"


def expected_cells() -> tuple[tuple[int, str], ...]:
    return tuple(
        (layer, role)
        for layer in range(QWEN3_8B_LAYER_COUNT)
        for role in QWEN3_PROJECTION_ROLES
    )


def expected_projection_names() -> tuple[str, ...]:
    return tuple(
        expected_projection_name(layer, role) for layer, role in expected_cells()
    )


def expected_projection_dimensions(role: str) -> tuple[int, int]:
    try:
        return QWEN3_PROJECTION_DIMENSIONS[role]
    except KeyError as error:
        raise ValueError(f"unknown Qwen3 projection role {role!r}") from error


def canonical_packed_tensor_schema(layer: int, role: str) -> dict[str, dict[str, Any]]:
    """Return the exact five-tensor serialized schema for one canonical projection."""

    module_name = expected_projection_name(layer, role)
    in_features, out_features = expected_projection_dimensions(role)
    return expected_packed_tensor_metadata(
        ProjectionCell(layer, role, module_name, in_features, out_features)
    )


def expected_packed_tensor_metadata(cell: ProjectionCell) -> dict[str, dict[str, Any]]:
    """Return the one canonical serialized V2B2-P32 W2 payload for a projection."""

    tile_count = (cell.in_features // 16) * (cell.out_features // 16)
    return {
        f"{cell.name}.trellis": {"shape": [tile_count, 16], "dtype": "I32"},
        f"{cell.name}.SU": {"shape": [cell.in_features], "dtype": "F32"},
        f"{cell.name}.SV": {"shape": [cell.out_features], "dtype": "F32"},
        f"{cell.name}.bank_ids": {"shape": [tile_count], "dtype": "U8"},
        f"{cell.name}.bank_alt_id": {"shape": [1], "dtype": "U8"},
    }


def acceptance_observation_digest(observation: Mapping[str, Any]) -> str:
    """Hash an evidence record excluding its self-authenticating digest field."""

    payload = dict(observation)
    payload.pop("observation_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    _hash_frame(digest, QVQ_OBSERVATION_HASH_SCHEME.encode("ascii"))
    _hash_frame(digest, encoded)
    return digest.hexdigest()


def seal_acceptance_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(observation)
    sealed["observation_sha256"] = acceptance_observation_digest(sealed)
    return sealed


def _valid_sealed_observation(observation: Any) -> bool:
    return (
        isinstance(observation, dict)
        and observation.get("observation_sha256") == acceptance_observation_digest(observation)
    )


def _valid_controller_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def qwen3_payload_aggregate(module_hashes: Mapping[str, str], tensor_counts: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    _hash_frame(digest, QVQ_PAYLOAD_HASH_SCHEME.encode("utf-8"))
    for module_name in sorted(module_hashes, key=lambda value: value.encode("utf-8")):
        _hash_frame(digest, module_name.encode("utf-8"))
        _hash_frame(digest, bytes.fromhex(module_hashes[module_name]))
        _hash_frame(digest, struct.pack("<Q", tensor_counts[module_name]))
    return digest.hexdigest()


def valid_qwen3_payload_hashes(payload: Any) -> bool:
    """Validate the complete canonical 252-module live/reloaded payload census."""

    if not isinstance(payload, dict):
        return False
    if set(payload) != {"scheme", "module_count", "module_sha256", "module_tensor_counts", "aggregate_sha256"}:
        return False
    module_hashes = payload.get("module_sha256")
    tensor_counts = payload.get("module_tensor_counts")
    structurally_valid = (
        payload.get("scheme") == QVQ_PAYLOAD_HASH_SCHEME
        and payload.get("module_count") == QWEN3_EXPECTED_MODULE_COUNT
        and isinstance(module_hashes, dict)
        and set(module_hashes) == set(expected_projection_names())
        and all(_valid_sha256(value) for value in module_hashes.values())
        and isinstance(tensor_counts, dict)
        and set(tensor_counts) == set(expected_projection_names())
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in tensor_counts.values())
        and _valid_sha256(payload.get("aggregate_sha256"))
    )
    if not structurally_valid:
        return False
    return payload["aggregate_sha256"] == qwen3_payload_aggregate(module_hashes, tensor_counts)


def validate_producer_pre_save_measurement(
    measurement: Any,
    *,
    event: Mapping[str, Any],
    controller_datasets: Mapping[str, Any] | None = None,
    controller_quant_config_sha256: str | None = None,
    controller_quant_config: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed before save unless the producer supplied the entire pinned live evidence."""

    if not isinstance(measurement, dict) or set(measurement) != {
        "dense_source_binding", "pre_save", "quantize_config", "quant_config_authority_sha256", "layer_scope",
        "datasets",
    }:
        raise AcceptanceError("producer pre-save measurement is absent")
    dense = measurement.get("dense_source_binding")
    pre_save = measurement.get("pre_save")
    if controller_quant_config is None:
        raise AcceptanceError("producer validation lacks retained quantization-config authority")
    actual_config = measurement.get("quantize_config")
    pinned_config_path = Path(__file__).resolve().parents[2] / "configs" / "qwen3_8b_qvq_w2_acceptance.json"
    if (
        not isinstance(actual_config, dict)
        or actual_config != controller_quant_config
        or measurement.get("quant_config_authority_sha256")
        != (controller_quant_config_sha256 or _sha256_file(pinned_config_path))
        or measurement.get("layer_scope") != "all"
    ):
        raise AcceptanceError("producer config/scope is not the canonical all-layer Qwen3 acceptance run")
    datasets = measurement.get("datasets")
    if (
        not isinstance(datasets, dict)
        or set(datasets) != {"calibration", "yaqa", "validation"}
        or controller_datasets is None
        or datasets != controller_datasets
    ):
        raise AcceptanceError("producer calibration/evaluation manifest evidence is incomplete")
    for evidence in datasets.values():
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {
                "source", "config", "split", "row_start", "rows", "content_sha256", "identity_manifest",
                "identity_manifest_sha256", "manifest_verified",
            }
            or evidence.get("rows") != 512
            or not _valid_sha256(evidence.get("content_sha256"))
            or not _valid_sha256(evidence.get("identity_manifest_sha256"))
            or evidence.get("manifest_verified") is not True
        ):
            raise AcceptanceError("producer calibration/evaluation manifest evidence is invalid")
    if (
        not isinstance(event, dict)
        or set(event) != {
            "run_nonce", "controller_instance_id", "controller_pid", "stage", "stage_nonce",
            "process_instance_id", "measurement",
        }
        or not isinstance(dense, dict)
        or set(dense) != {
            "controller_run_nonce", "producer_process_instance_id", "producer_stage_nonce", "start", "end",
        }
        or not isinstance(pre_save, dict)
        or set(pre_save) != {"dense_source_end_sha256", "payload"}
    ):
        raise AcceptanceError("producer dense/live pre-save evidence is incomplete")
    start = dense.get("start")
    end = dense.get("end")
    if (
        not isinstance(start, dict)
        or set(start) != {
            "stage", "controller_run_nonce", "process_instance_id", "producer_stage_nonce", "artifact_sha256",
            "observation_sha256",
        }
        or not isinstance(end, dict)
        or set(end) != {
            "stage", "controller_run_nonce", "process_instance_id", "producer_stage_nonce", "artifact_sha256",
            "previous_observation_sha256", "observation_sha256",
        }
    ):
        raise AcceptanceError("producer payload/dense binding failed pre-save controller validation")
    identity = (event.get("run_nonce"), event.get("process_instance_id"), event.get("stage_nonce"))
    if (
        (dense.get("controller_run_nonce"), dense.get("producer_process_instance_id"), dense.get("producer_stage_nonce")) != identity
        or not _valid_sealed_observation(dense.get("start"))
        or not _valid_sealed_observation(dense.get("end"))
        or dense["start"].get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or dense["end"].get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or dense["end"].get("previous_observation_sha256") != dense["start"].get("observation_sha256")
        or pre_save.get("dense_source_end_sha256") != dense["end"].get("observation_sha256")
        or not valid_qwen3_payload_hashes(pre_save.get("payload"))
    ):
        raise AcceptanceError("producer payload/dense binding failed pre-save controller validation")


def validate_controller_transcript(
    transcript: Any,
    *,
    dense_binding: Any,
    parity: Any,
    report: Mapping[str, Any],
    controller_authority: Any,
    trust_resources: _TrustedResources,
) -> None:
    """Validate controller-signed process facts and bind every parity measurement to them."""

    if not isinstance(transcript, dict) or transcript.get("schema") != CONTROLLER_SCHEMA:
        raise AcceptanceError("artifact lacks the independent acceptance controller transcript")
    try:
        expected_authority = controller_authority_receipt(transcript, resources=trust_resources)
    except (RuntimeError, TypeError, ValueError) as error:
        raise AcceptanceError("acceptance controller transcript cannot produce an authority receipt") from error
    if not isinstance(controller_authority, dict) or controller_authority != expected_authority:
        raise AcceptanceError("report is not bound to the independently supplied controller authority receipt")
    if not verify_controller_signature(transcript, resources=trust_resources):
        raise AcceptanceError("acceptance controller Ed25519 signature is absent or invalid")
    if not _valid_controller_identity(transcript.get("controller_instance_id")) or not _valid_controller_identity(
        transcript.get("run_nonce")
    ):
        raise AcceptanceError("acceptance controller identities are not unpredictable 256-bit values")
    records = transcript.get("processes")
    if transcript.get("trust_config_sha256") != hashlib.sha256(
        canonical_content(trust_resources.trust)
    ).hexdigest():
        raise AcceptanceError("controller transcript is not bound to the external operator trust configuration")
    if (
        not isinstance(records, list)
        or not records
        or transcript.get("controller_datasets") != records[0].get("controller_datasets")
    ):
        raise AcceptanceError("controller manifest observations are absent or split")
    controller_pid = transcript.get("controller_pid")
    controller_parent_pid = transcript.get("controller_parent_pid")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (controller_pid, controller_parent_pid)):
        raise AcceptanceError("acceptance controller identity/parent PID binding is invalid")
    if not isinstance(records, list) or len(records) != len(CONTROLLER_STAGES):
        raise AcceptanceError("acceptance controller spawn/exit records are missing")
    if tuple(record.get("stage") for record in records if isinstance(record, dict)) != CONTROLLER_STAGES:
        raise AcceptanceError("acceptance controller stages are missing, reordered, or replayed")
    instances: set[str] = set()
    nonces: set[str] = set()
    previous_hash = None
    previous_exit = None
    parsed_commands: list[dict[str, str]] = []
    for record in records:
        stage = record["stage"]
        event = record.get("event")
        instance = record.get("process_instance_id")
        stage_nonce = record.get("stage_nonce")
        if (
            not _valid_controller_identity(instance)
            or not _valid_controller_identity(stage_nonce)
            or instance in instances
            or stage_nonce in nonces
        ):
            raise AcceptanceError("controller process-instance identities/nonces are invalid or replayed")
        instances.add(instance)
        nonces.add(stage_nonce)
        if (
            not isinstance(record.get("pid"), int)
            or isinstance(record.get("pid"), bool)
            or record["pid"] <= 0
            or not isinstance(record.get("parent_pid"), int)
            or record["parent_pid"] <= 0
            or record.get("parent_pid") != controller_pid
            or record.get("exit_code") != 0
            or record.get("pidfd_bound_through_wait") is not True
        ):
            raise AcceptanceError(f"controller {stage} spawn/exit metadata is invalid or unsuccessful")
        spawned = record.get("spawned_monotonic_ns")
        received = record.get("event_received_monotonic_ns")
        acknowledged = record.get("acknowledgement_sent_monotonic_ns")
        exited = record.get("exited_monotonic_ns")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (spawned, received, acknowledged, exited)) or not (
            spawned <= received <= acknowledged <= exited
        ):
            raise AcceptanceError(f"controller {stage} spawn/event/exit ordering is invalid")
        if previous_exit is not None and spawned < previous_exit:
            raise AcceptanceError("controller stages overlap or are not sequentially observed")
        previous_exit = exited
        if record.get("previous_record_sha256") != previous_hash:
            raise AcceptanceError("acceptance controller record chain is broken")
        unsigned_record = dict(record)
        record_hash = unsigned_record.pop("record_sha256", None)
        if record_hash != hashlib.sha256(canonical_content(unsigned_record)).hexdigest():
            raise AcceptanceError("acceptance controller record digest is invalid")
        previous_hash = record_hash
        argv = record.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise AcceptanceError(f"controller {stage} canonical argv is absent")
        argv_digest = hashlib.sha256(b"\0".join(os.fsencode(item) for item in argv)).hexdigest()
        if record.get("argv_sha256") != argv_digest:
            raise AcceptanceError(f"controller {stage} argv digest is invalid")
        execution_argv = record.get("execution_argv")
        if (
            not isinstance(execution_argv, list)
            or len(execution_argv) != len(argv)
            or execution_argv[0] != argv[0]
            or execution_argv[2:] != argv[2:]
            or not isinstance(execution_argv[1], str)
            or not execution_argv[1].startswith("/proc/self/fd/")
            or not execution_argv[1].removeprefix("/proc/self/fd/").isdigit()
            or record.get("execution_argv_sha256")
            != hashlib.sha256(b"\0".join(os.fsencode(item) for item in execution_argv)).hexdigest()
        ):
            raise AcceptanceError(f"controller {stage} canonical required command has invalid descriptor execution")
        try:
            command_values = validate_required_command(stage, argv, resources=trust_resources)
        except (RuntimeError, ValueError) as error:
            raise AcceptanceError(f"controller {stage} command is not the canonical required command") from error
        parsed_commands.append(command_values)
        os_process = record.get("os_process")
        trust = trust_resources.trust
        expected_cmdline = b"\0".join(os.fsencode(item) for item in execution_argv) + b"\0"
        python_identity = trust_resources.files["python_executable"].identity
        if (
            not isinstance(os_process, dict)
            or os_process.get("pid") != record.get("pid")
            or os_process.get("ppid") != controller_pid
            or not isinstance(os_process.get("start_time_ticks"), int)
            or os_process.get("start_time_ticks") <= 0
            or os_process.get("executable_sha256") != trust["python_executable_sha256"]
            or not isinstance(os_process.get("executable_device"), int)
            or not isinstance(os_process.get("executable_inode"), int)
            or (os_process.get("executable_device"), os_process.get("executable_inode"))
            != (python_identity.st_dev, python_identity.st_ino)
            or os_process.get("cmdline_sha256") != hashlib.sha256(expected_cmdline).hexdigest()
        ):
            raise AcceptanceError(f"controller {stage} command is not the canonical required command")
        if stage == CONTROLLER_STAGES[0]:
            try:
                observed_datasets = controller_dataset_evidence(command_values)
            except RuntimeError as error:
                raise AcceptanceError("controller cannot reproduce trusted producer manifests") from error
            if observed_datasets != transcript.get("controller_datasets"):
                raise AcceptanceError("signed controller manifest observations mismatch trusted pre-run inputs")
        acknowledgement = record.get("acknowledgement")
        if not isinstance(acknowledgement, dict) or acknowledgement.get("acknowledged_event_sha256") != record.get("event_sha256"):
            raise AcceptanceError(f"controller {stage} acknowledgement is absent or unbound")
        if stage == CONTROLLER_STAGES[0] and (
            acknowledgement.get("validation_policy") != "qwen3-producer-pre-save-v1"
            or not isinstance(acknowledgement.get("validated_before_save_monotonic_ns"), int)
            or not received <= acknowledgement["validated_before_save_monotonic_ns"] <= acknowledged
        ):
            raise AcceptanceError("producer acknowledgement does not prove pre-save evidence validation")
        if (
            not isinstance(event, dict)
            or event.get("run_nonce") != transcript["run_nonce"]
            or event.get("controller_instance_id") != transcript["controller_instance_id"]
            or event.get("controller_pid") != controller_pid
            or event.get("stage") != stage
            or event.get("stage_nonce") != stage_nonce
            or event.get("process_instance_id") != instance
            or record.get("event_sha256") != hashlib.sha256(canonical_content(event)).hexdigest()
        ):
            raise AcceptanceError(f"controller {stage} event is not bound to its spawned process instance")

    producer_command, reload_command, evaluation_command = parsed_commands
    manifest_dir = str(Path(producer_command["--calibration-dataset"]).parent)
    cross_stage_paths = (
        producer_command["--output"] == reload_command["--checkpoint"] == evaluation_command["--checkpoint"]
        and producer_command["--yaqa-dataset"] == str(Path(manifest_dir) / "yaqa_tuning.jsonl")
        and producer_command["--validation-dataset"] == evaluation_command["--validation-jsonl"]
        and evaluation_command["--manifest-dir"] == manifest_dir
        and evaluation_command["--held-out-diagnostics-jsonl"] == str(Path(manifest_dir) / "held_out_diagnostics.jsonl")
        and evaluation_command["--diverse-jsonl"] == str(Path(manifest_dir) / "diverse_32.jsonl")
    )
    if not cross_stage_paths:
        raise AcceptanceError("controller stage paths are split across checkpoint or manifest authorities")
    if (
        report.get("accounting", {}).get("maximum_bpw") != float(evaluation_command["--maximum-bpw"])
        or report.get("thresholds") != {
            "top1_agreement_min": float(evaluation_command["--score-min"]),
            "diverse_32_min": float(evaluation_command["--score-min"]),
            "final_kl_max_nats": float(evaluation_command["--final-kl-max-nats"]),
        }
    ):
        raise AcceptanceError("report thresholds are split from the locked controller evaluation policy")

    producer_event, reload_event, evaluation_event = (record["event"] for record in records)
    producer_measurement = producer_event.get("measurement")
    reload_measurement = reload_event.get("measurement")
    evaluation_measurement = evaluation_event.get("measurement")
    if not all(isinstance(value, dict) for value in (producer_measurement, reload_measurement, evaluation_measurement)):
        raise AcceptanceError("controller stage measurements are absent")
    if producer_measurement.get("dense_source_binding") != dense_binding:
        raise AcceptanceError("dense source binding was not emitted by the controller-spawned producer")
    if producer_measurement.get("datasets") != transcript.get("controller_datasets"):
        raise AcceptanceError("producer manifest assertions do not match controller-opened trusted inputs")
    if records[0].get("controller_live_payload") != producer_measurement.get("pre_save", {}).get("payload"):
        raise AcceptanceError("producer payload digests are not backed by controller-hashed live tensor bytes")
    expected_parity = {
        "verified": True,
        "controller_run_nonce": transcript["run_nonce"],
        "pre_save": {
            **producer_measurement.get("pre_save", {}),
            "stage": CONTROLLER_STAGES[0],
            "stage_nonce": producer_event["stage_nonce"],
            "process_instance_id": producer_event["process_instance_id"],
        },
        "fresh_process": {
            **reload_measurement,
            "stage": CONTROLLER_STAGES[1],
            "stage_nonce": reload_event["stage_nonce"],
            "process_instance_id": reload_event["process_instance_id"],
        },
        "evaluation_reload": {
            "payload": evaluation_measurement.get("payload"),
            "stage": CONTROLLER_STAGES[2],
            "stage_nonce": evaluation_event["stage_nonce"],
            "process_instance_id": evaluation_event["process_instance_id"],
        },
    }
    if parity != expected_parity:
        raise AcceptanceError("parity evidence is self-authored, replayed, or not controller-bound")
    claim = {key: report.get(key) for key in ("accounting", "manifests", "thresholds", "global", "cells")}
    if evaluation_measurement.get("acceptance_evidence_sha256") != hashlib.sha256(canonical_content(claim)).hexdigest():
        raise AcceptanceError("acceptance metrics were not emitted by the controller-spawned evaluation process")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AcceptanceError(f"required {label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"failed to parse {label} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"{label} must contain one JSON object: {path}")
    return payload


def _validate_pinned_dense_hashes(actual: Mapping[str, str]) -> None:
    actual_names = set(actual)
    expected_names = set(QWEN3_DENSE_ARTIFACT_SHA256)
    if actual_names != expected_names:
        raise AcceptanceError(
            "pinned dense artifact file set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )
    digest_mismatches = {
        name: {"expected": QWEN3_DENSE_ARTIFACT_SHA256[name], "actual": actual[name]}
        for name in sorted(expected_names)
        if actual[name] != QWEN3_DENSE_ARTIFACT_SHA256[name]
    }
    if digest_mismatches:
        raise AcceptanceError(
            f"pinned dense artifact SHA-256 mismatch: {digest_mismatches}"
        )


def validate_qwen3_model_artifact(
    model_dir: Path, *, require_pinned_dense: bool
) -> dict[str, Any]:
    """Validate exact Qwen3-8B architecture and local revision/content authority."""

    model_dir = model_dir.expanduser().resolve()
    if require_pinned_dense and model_dir != QWEN3_LOCAL_TARGET.resolve():
        raise AcceptanceError(
            f"dense Qwen3 target must be exactly {QWEN3_LOCAL_TARGET}, got {model_dir}"
        )
    if not model_dir.is_dir():
        raise AcceptanceError(
            f"Qwen3 model artifact directory does not exist: {model_dir}"
        )
    config_path = model_dir / "config.json"
    config = _load_json_object(config_path, label="Qwen3 config.json")
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in QWEN3_MODEL_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise AcceptanceError(f"Qwen3-8B config identity mismatch: {mismatches}")
    config_sha256 = _sha256_file(config_path)
    if require_pinned_dense and config_sha256 != QWEN3_DENSE_CONFIG_SHA256:
        raise AcceptanceError(
            f"dense Qwen3 config hash mismatch: expected {QWEN3_DENSE_CONFIG_SHA256}, got {config_sha256}"
        )

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise AcceptanceError(
            f"Qwen3 model artifact has no safetensors files: {model_dir}"
        )
    index_path = model_dir / "model.safetensors.index.json"
    if len(shards) > 1 and not index_path.is_file():
        raise AcceptanceError(
            "sharded Qwen3 model artifact lacks model.safetensors.index.json"
        )
    layer_ids: set[int] = set()
    index_sha256 = None
    if index_path.is_file():
        index = _load_json_object(
            index_path, label="Qwen3 model.safetensors.index.json"
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise AcceptanceError("Qwen3 shard index lacks a nonempty weight_map")
        shard_names = {path.name for path in shards}
        if set(weight_map.values()) != shard_names:
            raise AcceptanceError(
                "Qwen3 shard index does not reference exactly the local safetensors shards"
            )
        for key in weight_map:
            parts = key.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_ids.add(int(parts[2]))
                except ValueError as error:
                    raise AcceptanceError(
                        f"Qwen3 shard index has malformed decoder layer key {key!r}"
                    ) from error
        index_sha256 = _sha256_file(index_path)
    else:
        metadata, _containers = _serialized_tensor_metadata(model_dir)
        for key in metadata:
            parts = key.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layer_ids.add(int(parts[2]))
                except ValueError as error:
                    raise AcceptanceError(
                        f"Qwen3 checkpoint has malformed decoder layer key {key!r}"
                    ) from error
    if layer_ids != set(range(QWEN3_8B_LAYER_COUNT)):
        raise AcceptanceError(
            f"Qwen3 artifact must contain exactly decoder layers 0..35 with no extras; found {sorted(layer_ids)}"
        )

    revision = None
    metadata_identities: dict[str, str] = {}
    artifact_sha256 = {path.name: _sha256_file(path) for path in [config_path, *shards]}
    if index_path.is_file():
        artifact_sha256[index_path.name] = index_sha256
    if require_pinned_dense:
        _validate_pinned_dense_hashes(artifact_sha256)
        metadata_root = model_dir / ".cache" / "huggingface" / "download"
        critical = [config_path, *shards]
        if index_path.is_file():
            critical.append(index_path)
        for artifact_path in critical:
            metadata_path = metadata_root / f"{artifact_path.name}.metadata"
            if not metadata_path.is_file():
                raise AcceptanceError(
                    f"pinned Qwen3 artifact lacks local Hub metadata for {artifact_path.name}"
                )
            try:
                lines = metadata_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError) as error:
                raise AcceptanceError(
                    f"failed to read Hub metadata for {artifact_path.name}: {error}"
                ) from error
            expected_content_identity = QWEN3_HUB_CONTENT_IDENTITIES.get(
                artifact_path.name
            )
            if (
                len(lines) < 2
                or lines[0] != QWEN3_PINNED_REVISION
                or lines[1] != expected_content_identity
            ):
                raise AcceptanceError(
                    f"Hub metadata for {artifact_path.name} is not pinned to {QWEN3_PINNED_REVISION}"
                )
            metadata_identities[artifact_path.name] = lines[1]
        revision = QWEN3_PINNED_REVISION
    return {
        "path": str(model_dir),
        "model_name": "Qwen3-8B",
        "instruction_identity": (
            "Post-trained Qwen3-8B with instruction-following and switchable thinking; local/model-card name is "
            "Qwen3-8B, not Qwen3-8B-Instruct"
        ),
        "revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "artifact_sha256": artifact_sha256,
        "config": {key: config[key] for key in QWEN3_MODEL_CONFIG},
        "decoder_layers": sorted(layer_ids),
        "hub_content_identities": metadata_identities,
    }


def census_reloaded_model(model: torch.nn.Module) -> list[ProjectionCell]:
    """Require exactly one packed QVQ module in every Qwen3 layer/role cell.

    Matching is by the terminal ``model.layers.<n>.<attention|mlp>.<role>`` path,
    which tolerates a loader-owned leading wrapper while rejecting duplicates.
    """

    found: dict[tuple[int, str], ProjectionCell] = {}
    projection_suffixes = tuple(f".{role}" for role in QWEN3_PROJECTION_ROLES)
    for name, module in model.named_modules(remove_duplicate=False):
        if not name.endswith(projection_suffixes):
            continue
        parts = name.split(".")
        try:
            layers_index = max(
                index for index, part in enumerate(parts) if part == "layers"
            )
            layer = int(parts[layers_index + 1])
        except (ValueError, IndexError):
            continue
        role = parts[-1]
        if (
            layer not in range(QWEN3_8B_LAYER_COUNT)
            or role not in QWEN3_PROJECTION_ROLES
        ):
            continue
        key = (layer, role)
        if key in found:
            raise AcceptanceError(
                f"duplicate requested projection cell {key}: {found[key].name!r} and {name!r}"
            )
        if type(module) is not QVQLinear:
            raise AcceptanceError(
                f"requested projection {name!r} is {type(module).__name__}, not the exact packed QVQLinear runtime"
            )
        if module.bits != QWEN3_REQUESTED_BITS:
            precision = (
                "higher-precision fallback"
                if module.bits > QWEN3_REQUESTED_BITS
                else "wrong-rate fallback"
            )
            raise AcceptanceError(
                f"requested projection {name!r} is a {precision} at W{module.bits:g}"
            )
        if module.bank_count != 2 or not module.v2b2_p32:
            raise AcceptanceError(
                f"requested projection {name!r} is not the frozen packed QVQ V2B2-P32 W2 format"
            )
        required = {"trellis", "SU", "SV"}
        if module.bank_count in (2, 4):
            required.add("bank_ids")
        if module.bank_count == 2:
            required.add("bank_alt_id")
        buffers = dict(module.named_buffers(recurse=False))
        missing_payload = sorted(required - buffers.keys())
        if missing_payload or any(
            buffers[key].is_meta for key in required if key in buffers
        ):
            raise AcceptanceError(
                f"requested projection {name!r} lacks loaded packed payload: {missing_payload}"
            )
        if module.bank_count in (2, 4) and not getattr(
            module, "_bank_ids_loaded", True
        ):
            raise AcceptanceError(
                f"requested projection {name!r} has no loaded bank selector authority"
            )
        canonical_name = expected_projection_name(layer, role)
        if not (name == canonical_name or name.endswith("." + canonical_name)):
            raise AcceptanceError(
                f"requested projection {name!r} does not have canonical Qwen3 identity {canonical_name!r}"
            )
        expected_in, expected_out = expected_projection_dimensions(role)
        actual_dimensions = (module.in_features, module.out_features)
        if actual_dimensions != (expected_in, expected_out):
            raise AcceptanceError(
                f"requested projection {name!r} dimension mismatch: expected "
                f"in_features={expected_in}, out_features={expected_out}; got {actual_dimensions}"
            )
        found[key] = ProjectionCell(
            layer,
            role,
            canonical_name,
            expected_in,
            expected_out,
            tuple(sorted(required)),
            name,
        )

    missing = sorted(set(expected_cells()) - found.keys())
    extra = sorted(set(found) - set(expected_cells()))
    if missing or extra or len(found) != QWEN3_EXPECTED_MODULE_COUNT:
        raise AcceptanceError(
            f"projection census must be exactly {QWEN3_EXPECTED_MODULE_COUNT}; found={len(found)}, "
            f"missing={missing}, extra={extra}"
        )
    return [found[key] for key in expected_cells()]


def _hash_frame(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def hash_qvq_module_payloads(
    modules: Iterable[tuple[str, torch.nn.Module]],
) -> dict[str, Any]:
    """Hash actual QVQ tensor payloads with an architecture-independent canonical encoding.

    V1 sorts modules and their direct parameters/buffers by UTF-8 name. Each module hash is
    SHA-256 over length-prefixed UTF-8 scheme/name records followed by, for every tensor,
    its name, torch dtype string, compact JSON shape, and contiguous CPU storage bytes.
    The aggregate uses the same framing over sorted ``(module name, module digest)`` pairs.
    """

    if sys.byteorder != "little":
        raise AcceptanceError("QVQ payload hashing v1 requires a little-endian host")
    keyed = sorted(modules, key=lambda item: item[0].encode("utf-8"))
    if not keyed or len({name for name, _module in keyed}) != len(keyed):
        raise AcceptanceError(
            "QVQ payload hashing requires nonempty unique module names"
        )
    module_hashes: dict[str, str] = {}
    tensor_counts: dict[str, int] = {}
    for module_name, module in keyed:
        tensors = {
            **dict(module.named_parameters(prefix="", recurse=False)),
            **dict(module.named_buffers(prefix="", recurse=False)),
        }
        if not tensors:
            raise AcceptanceError(f"QVQ module {module_name!r} has no tensor payload")
        digest = hashlib.sha256()
        _hash_frame(digest, QVQ_PAYLOAD_HASH_SCHEME.encode("utf-8"))
        _hash_frame(digest, module_name.encode("utf-8"))
        for tensor_name in sorted(tensors, key=lambda value: value.encode("utf-8")):
            tensor = tensors[tensor_name]
            if tensor.is_meta:
                raise AcceptanceError(
                    f"QVQ payload tensor {module_name}.{tensor_name} is meta"
                )
            value = tensor.detach().contiguous().cpu()
            raw = value.view(torch.uint8).numpy().tobytes(order="C")
            _hash_frame(digest, tensor_name.encode("utf-8"))
            _hash_frame(digest, str(value.dtype).encode("ascii"))
            _hash_frame(
                digest,
                json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
            )
            _hash_frame(digest, raw)
        module_hashes[module_name] = digest.hexdigest()
        tensor_counts[module_name] = len(tensors)
    return {
        "scheme": QVQ_PAYLOAD_HASH_SCHEME,
        "module_count": len(module_hashes),
        "module_sha256": module_hashes,
        "module_tensor_counts": tensor_counts,
        "aggregate_sha256": qwen3_payload_aggregate(module_hashes, tensor_counts),
    }


def hash_canonical_qwen3_payloads(
    model: torch.nn.Module, cells: Sequence[ProjectionCell]
) -> dict[str, Any]:
    runtime_modules = dict(model.named_modules(remove_duplicate=False))
    selected = []
    for cell in cells:
        runtime_name = cell.runtime_name or cell.name
        module = runtime_modules.get(runtime_name)
        if module is None:
            raise AcceptanceError(
                f"QVQ payload hashing cannot resolve runtime module {runtime_name!r}"
            )
        selected.append((cell.name, module))
    return hash_qvq_module_payloads(selected)


def iter_canonical_qwen3_payload_records(
    model: torch.nn.Module, cells: Sequence[ProjectionCell]
) -> Iterable[tuple[dict[str, Any], bytes]]:
    """Yield canonical tensor metadata/bytes for controller-side independent live hashing."""

    runtime_modules = dict(model.named_modules(remove_duplicate=False))
    for cell in sorted(cells, key=lambda item: item.name.encode("utf-8")):
        module = runtime_modules.get(cell.runtime_name or cell.name)
        if module is None:
            raise AcceptanceError(f"live payload stream cannot resolve {cell.name!r}")
        tensors = {
            **dict(module.named_parameters(prefix="", recurse=False)),
            **dict(module.named_buffers(prefix="", recurse=False)),
        }
        for tensor_name in sorted(tensors, key=lambda value: value.encode("utf-8")):
            tensor = tensors[tensor_name]
            if tensor.is_meta:
                raise AcceptanceError(f"live payload tensor {cell.name}.{tensor_name} is meta")
            value = tensor.detach().contiguous().cpu()
            raw = value.view(torch.uint8).numpy().tobytes(order="C")
            yield ({
                "module": cell.name,
                "tensor": tensor_name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "byte_count": len(raw),
            }, raw)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.float32: "F32",
    torch.int64: "I64",
    torch.float64: "F64",
}


def _serialized_tensor_metadata(
    checkpoint: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Parse exact safetensors data offsets without materializing model weights."""

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise AcceptanceError(f"checkpoint has no safetensors files: {checkpoint}")
    tensors: dict[str, dict[str, Any]] = {}
    containers: dict[str, int] = {}
    for path in files:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise AcceptanceError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            header_bytes = handle.read(header_size)
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AcceptanceError(f"invalid safetensors header: {path}") from error
        data_size = file_size - 8 - header_size
        if data_size < 0:
            raise AcceptanceError(f"safetensors header exceeds file size: {path}")
        containers[path.name] = file_size
        shard_extents: list[tuple[int, int, str]] = []
        for key, raw in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise AcceptanceError(
                    f"serialized tensor {key!r} is duplicated across safetensors shards"
                )
            if not isinstance(raw, dict) or set(raw) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise AcceptanceError(
                    f"invalid safetensors tensor metadata for {key!r}"
                )
            dtype = raw["dtype"]
            shape = raw["shape"]
            offsets = raw["data_offsets"]
            if dtype not in _SAFETENSORS_DTYPE_BYTES or not isinstance(shape, list):
                raise AcceptanceError(
                    f"unsupported serialized tensor dtype/shape for {key!r}: {dtype!r}, {shape!r}"
                )
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(item, int) for item in offsets)
            ):
                raise AcceptanceError(f"invalid serialized data offsets for {key!r}")
            start, stop = offsets
            expected = math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
            if (
                start < 0
                or stop < start
                or stop > data_size
                or stop - start != expected
            ):
                raise AcceptanceError(
                    f"serialized byte extent disagrees with dtype/shape for {key!r}"
                )
            shard_extents.append((start, stop, key))
            tensors[key] = {
                "bytes": stop - start,
                "dtype": dtype,
                "shape": shape,
                "shard": path.name,
            }
        cursor = 0
        for start, stop, key in sorted(shard_extents):
            if start != cursor:
                raise AcceptanceError(
                    f"safetensors shard has an overlap or unassigned data gap before {key!r}"
                )
            cursor = stop
        if cursor != data_size:
            raise AcceptanceError(
                f"safetensors shard has {data_size - cursor} unassigned trailing data bytes: {path}"
            )
    index_path = checkpoint / "model.safetensors.index.json"
    if len(files) > 1 and not index_path.is_file():
        raise AcceptanceError("sharded checkpoint lacks model.safetensors.index.json")
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AcceptanceError(
                f"failed to parse safetensors index at {index_path}: {error}"
            ) from error
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != set(tensors):
            raise AcceptanceError(
                "safetensors index weight_map does not exactly match serialized tensor keys"
            )
        for key, shard in weight_map.items():
            if tensors[key]["shard"] != shard:
                raise AcceptanceError(
                    f"safetensors index maps {key!r} to the wrong shard"
                )
    return tensors, containers


def account_serialized_checkpoint(
    checkpoint: Path, cells: Sequence[ProjectionCell], *, maximum_bpw: float = 2.1
) -> dict[str, Any]:
    """Account exact tensor data extents from saved safetensors shards."""

    metadata, containers = _serialized_tensor_metadata(checkpoint)
    report = _account_serialized_metadata(
        metadata, cells, maximum_bpw=maximum_bpw, require_complete=True
    )
    tensor_bytes = sum(item["bytes"] for item in metadata.values())
    report["container_files"] = containers
    report["container_total_bytes"] = sum(containers.values())
    report["container_header_and_padding_bytes"] = (
        report["container_total_bytes"] - tensor_bytes
    )
    return report


def _account_serialized_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
    cells: Sequence[ProjectionCell],
    *,
    maximum_bpw: float,
    require_complete: bool,
) -> dict[str, Any]:
    if not math.isfinite(maximum_bpw) or maximum_bpw <= 0:
        raise ValueError("maximum_bpw must be finite and positive")
    if require_complete and len(cells) != QWEN3_EXPECTED_MODULE_COUNT:
        raise AcceptanceError(
            f"accounting requires exactly {QWEN3_EXPECTED_MODULE_COUNT} projection cells"
        )
    cell_names = [cell.name for cell in cells]
    if len(set(cell_names)) != len(cell_names) or (
        require_complete and set(cell_names) != set(expected_projection_names())
    ):
        raise AcceptanceError(
            "accounting projection cells are not the exact canonical Qwen3 layer-role set"
        )
    if any(
        cell.in_features <= 0 or cell.out_features <= 0 or cell.dense_weight_count <= 0
        for cell in cells
    ):
        raise AcceptanceError(
            "accounting projection dimensions and denominators must be positive"
        )
    if require_complete:
        dimension_mismatches = {
            cell.name: {
                "expected": expected_projection_dimensions(cell.role),
                "actual": (cell.in_features, cell.out_features),
            }
            for cell in cells
            if (cell.in_features, cell.out_features)
            != expected_projection_dimensions(cell.role)
        }
        if dimension_mismatches:
            raise AcceptanceError(
                f"accounting projection dimensions are not authoritative: {dimension_mismatches}"
            )
    prefixes = sorted((cell.name, cell) for cell in cells)
    target: dict[str, int] = {}
    non_target: dict[str, int] = {}
    unassigned_cells: set[str] = {cell.name for cell in cells}
    per_module: dict[str, dict[str, Any]] = {
        cell.name: {
            "bytes": 0,
            "dense_weight_count": cell.dense_weight_count,
            "tensors": {},
        }
        for cell in cells
    }
    for key, item in metadata.items():
        size = item.get("bytes")
        if not isinstance(size, int) or size < 0:
            raise AcceptanceError(
                f"serialized tensor {key!r} has an invalid byte extent"
            )
        matches = [
            (prefix, cell)
            for prefix, cell in prefixes
            if key == prefix or key.startswith(prefix + ".")
        ]
        if len(matches) > 1:
            raise AcceptanceError(
                f"serialized tensor {key!r} maps to multiple requested projections"
            )
        if matches:
            prefix, _cell = matches[0]
            target[key] = size
            per_module[prefix]["bytes"] += size
            per_module[prefix]["tensors"][key] = dict(item)
            unassigned_cells.discard(prefix)
        else:
            non_target[key] = size
    if unassigned_cells:
        raise AcceptanceError(
            f"requested projections have no serialized tensors: {sorted(unassigned_cells)}"
        )
    for cell in cells:
        serialized = per_module[cell.name]["tensors"]
        expected_metadata = expected_packed_tensor_metadata(cell)
        missing = sorted(set(expected_metadata) - set(serialized))
        if missing:
            raise AcceptanceError(
                f"requested projection {cell.name!r} lacks serialized payload tensors: {missing}"
            )
        mismatches = {}
        for name, expected in expected_metadata.items():
            actual = serialized[name]
            actual_shape = actual.get("shape")
            actual_dtype = actual.get("dtype")
            if actual_shape != expected["shape"] or actual_dtype != expected["dtype"]:
                mismatches[name] = {
                    "expected": expected,
                    "actual": {"shape": actual_shape, "dtype": actual_dtype},
                }
        if mismatches:
            raise AcceptanceError(
                f"requested projection {cell.name!r} packed tensor metadata mismatch: {mismatches}"
            )
    dense_weight_count = sum(cell.dense_weight_count for cell in cells)
    if dense_weight_count <= 0:
        raise AcceptanceError("accounting projection denominator must be positive")
    if require_complete and dense_weight_count != QWEN3_PROJECTION_DENSE_WEIGHT_COUNT:
        raise AcceptanceError(
            "accounting projection denominator does not equal the pinned Qwen3-8B denominator"
        )
    target_bytes = sum(target.values())
    effective_bpw = (8 * target_bytes) / dense_weight_count
    report = {
        "boundary": PROJECTION_BOUNDARY,
        "requested_projection_dense_weight_count": dense_weight_count,
        "requested_projection_tensor_bytes": target_bytes,
        "effective_bpw": effective_bpw,
        "maximum_bpw": maximum_bpw,
        "target_tensors": target,
        "per_module": per_module,
        "dense_non_target_tensor_bytes": sum(non_target.values()),
        "dense_non_target_tensors": non_target,
    }
    if effective_bpw > maximum_bpw:
        raise AcceptanceError(
            f"serialized projection effective BPW {effective_bpw:.9f} exceeds {maximum_bpw:.9f}"
        )
    return report


def account_serialized_state(
    state: Mapping[str, torch.Tensor],
    cells: Sequence[ProjectionCell],
    *,
    maximum_bpw: float = 2.1,
) -> dict[str, Any]:
    """Account actual serialized tensor payload bytes at the projection boundary.

    ``state`` must come from the freshly reloaded model's state dict.  Safetensors
    stores tensor payloads without compression, so ``numel*element_size`` is the
    serialized data-region byte count; container header bytes are reported but not
    assigned to tensors by this projection-only boundary.
    """

    metadata = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise AcceptanceError(f"state entry {key!r} is not a tensor")
        metadata[key] = {
            "bytes": _tensor_nbytes(tensor),
            "dtype": _TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype, str(tensor.dtype)),
            "shape": list(tensor.shape),
        }
    return _account_serialized_metadata(
        metadata, cells, maximum_bpw=maximum_bpw, require_complete=False
    )


def canonical_content(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def materialize_manifest(
    split: str,
    records: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    if split not in MANIFEST_SPLITS:
        raise ValueError(f"unknown manifest split {split!r}")
    rows = []
    seen_identities: set[str] = set()
    seen_hashes: set[str] = set()
    for ordinal, record in enumerate(records):
        if "identity" not in record or "content" not in record:
            raise AcceptanceError(
                f"{split} record {ordinal} requires identity and content"
            )
        identity = str(record["identity"]).strip()
        if not identity:
            raise AcceptanceError(f"{split} record {ordinal} has an empty identity")
        content_hash = hashlib.sha256(canonical_content(record["content"])).hexdigest()
        if identity in seen_identities or content_hash in seen_hashes:
            raise AcceptanceError(
                f"duplicate identity or content within {split}: {identity}"
            )
        seen_identities.add(identity)
        seen_hashes.add(content_hash)
        rows.append(
            {"ordinal": ordinal, "identity": identity, "content_sha256": content_hash}
        )
    if expected_count is not None and len(rows) != expected_count:
        raise AcceptanceError(
            f"{split} requires exactly {expected_count} records, found {len(rows)}"
        )
    required_count = {"diverse_pool_512": 512, "diverse_32": 32}.get(split)
    if required_count is not None and len(rows) != required_count:
        raise AcceptanceError(
            f"{split} requires exactly {required_count} records, found {len(rows)}"
        )
    payload = {"schema_version": 1, "split": split, "count": len(rows), "samples": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def validate_manifest_disjointness(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = sorted(set(MANIFEST_SPLITS) - manifests.keys())
    if missing:
        raise AcceptanceError(f"missing required sample manifests: {missing}")
    sets: dict[str, tuple[set[str], set[str]]] = {}
    for split in MANIFEST_SPLITS:
        payload = manifests[split]
        if (
            payload.get("schema_version") != 1
            or payload.get("split") != split
            or not isinstance(payload.get("samples"), list)
        ):
            raise AcceptanceError(f"invalid manifest schema for {split}")
        if payload.get("count") != len(payload["samples"]):
            raise AcceptanceError(f"manifest count disagrees with samples for {split}")
        identities: set[str] = set()
        hashes: set[str] = set()
        for ordinal, row in enumerate(payload["samples"]):
            if not isinstance(row, dict) or row.get("ordinal") != ordinal:
                raise AcceptanceError(f"invalid sample row or ordinal in {split}")
            identity = row.get("identity")
            content_hash = row.get("content_sha256")
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(content_hash, str)
            ):
                raise AcceptanceError(f"invalid sample identity/hash in {split}")
            if len(content_hash) != 64 or any(
                character not in "0123456789abcdef" for character in content_hash
            ):
                raise AcceptanceError(f"invalid SHA-256 content hash in {split}")
            identities.add(identity)
            hashes.add(content_hash)
        if len(identities) != len(payload["samples"]) or len(hashes) != len(
            payload["samples"]
        ):
            raise AcceptanceError(f"duplicate identity or content in {split}")
        sets[split] = identities, hashes
    comparisons = []
    for left, right in MANIFEST_DISJOINT_PAIRS:
        identity_overlap = sorted(sets[left][0] & sets[right][0])
        content_overlap = sorted(sets[left][1] & sets[right][1])
        comparisons.append(
            {
                "left": left,
                "right": right,
                "identity_overlap": identity_overlap,
                "content_overlap": content_overlap,
            }
        )
        if identity_overlap or content_overlap:
            raise AcceptanceError(
                f"sample leakage between {left} and {right}: identities={identity_overlap}, hashes={content_overlap}"
            )
    return {
        "pairwise_disjoint": True,
        "counts": {
            split: len(manifests[split]["samples"]) for split in MANIFEST_SPLITS
        },
        "comparisons": comparisons,
    }


def select_diverse_32(pool: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Select deterministic length-stratified midpoints from the frozen 512-row pool."""

    if len(pool) != 512:
        raise AcceptanceError(
            f"diverse pool requires exactly 512 records, found {len(pool)}"
        )
    ranked = []
    identities: set[str] = set()
    content_hashes: set[str] = set()
    for ordinal, record in enumerate(pool):
        if "identity" not in record or "content" not in record:
            raise AcceptanceError(
                f"diverse pool record {ordinal} requires identity and content"
            )
        identity = str(record["identity"]).strip()
        content = canonical_content(record["content"])
        content_hash = hashlib.sha256(content).hexdigest()
        if not identity or identity in identities or content_hash in content_hashes:
            raise AcceptanceError(
                f"duplicate/empty identity or content in diverse pool at row {ordinal}"
            )
        identities.add(identity)
        content_hashes.add(content_hash)
        ranked.append((len(content), identity.encode("utf-8"), ordinal, record))
    ranked.sort(key=lambda item: item[:3])
    return [ranked[bin_index * 16 + 8][3] for bin_index in range(32)]


def validate_diverse_selection(
    pool: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = select_diverse_32(pool)
    expected_pairs = [
        (
            str(row["identity"]),
            hashlib.sha256(canonical_content(row["content"])).hexdigest(),
        )
        for row in expected
    ]
    actual_pairs = [
        (
            str(row.get("identity", "")),
            hashlib.sha256(canonical_content(row.get("content"))).hexdigest(),
        )
        for row in selected
    ]
    if actual_pairs != expected_pairs:
        raise AcceptanceError(
            "diverse_32 is not the deterministic length-stratified selection from diverse_pool_512"
        )
    return {
        "verified": True,
        "scheme": DIVERSE_SELECTION_SCHEME,
        "pool_count": 512,
        "selected_count": 32,
        "selected_identities": [identity for identity, _content_hash in actual_pairs],
        "selected_content_sha256": [
            content_hash for _identity, content_hash in actual_pairs
        ],
    }


def _validate_acceptance_report_retained(
    report: Mapping[str, Any],
    *,
    controller_authority: Mapping[str, Any] | None,
    trust_resources: _TrustedResources,
) -> None:
    """Validate complete machine-readable evidence and every declared gate."""

    required = {
        "schema_version",
        "artifact",
        "census",
        "accounting",
        "manifests",
        "thresholds",
        "global",
        "cells",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise AcceptanceError(f"acceptance report missing fields: {missing}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise AcceptanceError("unsupported acceptance report schema version")
    artifact = report["artifact"]
    if artifact.get("fresh_reload_verified") is not True:
        raise AcceptanceError("artifact was not verified through a fresh reload")
    hashes = artifact.get("checkpoint_sha256")
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise AcceptanceError(
            "artifact requires SHA-256 identities for serialized checkpoint files"
        )
    dense_hashes = artifact.get("dense_model_sha256")
    if (
        not isinstance(dense_hashes, dict)
        or not dense_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in dense_hashes.values()
        )
    ):
        raise AcceptanceError(
            "artifact requires SHA-256 identities for the pinned dense model files"
        )
    if dense_hashes != QWEN3_DENSE_ARTIFACT_SHA256:
        raise AcceptanceError("artifact dense model hashes are not the exact pinned digest map")
    dense_identity = artifact.get("dense_model_identity")
    checkpoint_identity = artifact.get("checkpoint_model_identity")
    for label, identity in (
        ("dense", dense_identity),
        ("checkpoint", checkpoint_identity),
    ):
        if (
            not isinstance(identity, dict)
            or identity.get("config") != QWEN3_MODEL_CONFIG
        ):
            raise AcceptanceError(
                f"artifact lacks exact {label} Qwen3-8B config identity"
            )
        if identity.get("decoder_layers") != list(range(QWEN3_8B_LAYER_COUNT)):
            raise AcceptanceError(
                f"artifact lacks exact {label} decoder-layer identity"
            )
        config_hash = identity.get("config_sha256")
        source_hashes = dense_hashes if label == "dense" else hashes
        if config_hash != source_hashes.get("config.json"):
            raise AcceptanceError(f"artifact {label} config hash is not content-bound")
    if dense_identity.get("revision") != QWEN3_PINNED_REVISION:
        raise AcceptanceError("dense artifact revision identity is not pinned")
    if dense_identity.get("config_sha256") != QWEN3_DENSE_CONFIG_SHA256:
        raise AcceptanceError(
            "dense artifact config identity is not the known local target"
        )
    if dense_identity.get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256:
        raise AcceptanceError(
            "dense artifact shard/index SHA-256 identity is not the pinned local target"
        )
    dense_binding = artifact.get("dense_source_binding")
    if not isinstance(dense_binding, dict):
        raise AcceptanceError("artifact lacks quantization-start/end dense source binding")
    start = dense_binding.get("start")
    end = dense_binding.get("end")
    controller_run_nonce = dense_binding.get("controller_run_nonce")
    producer_instance = dense_binding.get("producer_process_instance_id")
    producer_stage_nonce = dense_binding.get("producer_stage_nonce")
    if (
        not _valid_controller_identity(controller_run_nonce)
        or not _valid_controller_identity(producer_instance)
        or not _valid_controller_identity(producer_stage_nonce)
        or not _valid_sealed_observation(start)
        or not _valid_sealed_observation(end)
        or start.get("stage") != "quantization_start"
        or end.get("stage") != "quantization_end"
        or start.get("controller_run_nonce") != controller_run_nonce
        or end.get("controller_run_nonce") != controller_run_nonce
        or start.get("process_instance_id") != producer_instance
        or end.get("process_instance_id") != producer_instance
        or start.get("producer_stage_nonce") != producer_stage_nonce
        or end.get("producer_stage_nonce") != producer_stage_nonce
        or start.get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or end.get("artifact_sha256") != QWEN3_DENSE_ARTIFACT_SHA256
        or end.get("previous_observation_sha256") != start.get("observation_sha256")
    ):
        raise AcceptanceError("quantization dense source binding is incomplete, mismatched, or unpinned")
    parity = artifact.get("packed_payload_parity")
    if not isinstance(parity, dict) or parity.get("verified") is not True:
        raise AcceptanceError(
            "artifact lacks measured pre-save/fresh-process packed payload parity"
        )
    observations = [parity.get(name) for name in ("pre_save", "fresh_process", "evaluation_reload")]

    if any(not isinstance(item, dict) for item in observations):
        raise AcceptanceError("packed payload parity evidence is incomplete or not controller-bound")
    payloads = [item.get("payload") for item in observations]
    if any(not valid_qwen3_payload_hashes(payload) for payload in payloads):
        raise AcceptanceError(
            "packed payload parity evidence is incomplete or uses a noncanonical hash scheme"
        )
    if not (payloads[0] == payloads[1] == payloads[2]):
        raise AcceptanceError(
            "packed module payloads drifted across pre-save, fresh process, or evaluation reload"
        )
    census = report["census"]
    if census != {
        "expected": QWEN3_EXPECTED_MODULE_COUNT,
        "actual": QWEN3_EXPECTED_MODULE_COUNT,
        "complete": True,
    }:
        raise AcceptanceError(
            "checkpoint census is not exactly 252 complete projection modules"
        )
    accounting = report["accounting"]
    bpw = accounting.get("effective_bpw")
    maximum_bpw = accounting.get("maximum_bpw")
    if accounting.get("boundary") != PROJECTION_BOUNDARY:
        raise AcceptanceError("accounting boundary is absent or changed")
    if any(not _is_finite_real(value) for value in (bpw, maximum_bpw)):
        raise AcceptanceError("accounting BPW evidence is absent or non-finite")
    if maximum_bpw > 2.1 or bpw > maximum_bpw:
        raise AcceptanceError("serialized projection BPW exceeds the W2/W2.1 contract")
    per_module = accounting.get("per_module")
    if (
        not isinstance(per_module, dict)
        or len(per_module) != QWEN3_EXPECTED_MODULE_COUNT
    ):
        raise AcceptanceError("serialized accounting lacks all 252 per-module records")
    if set(per_module) != set(expected_projection_names()):
        raise AcceptanceError(
            "serialized accounting module keys are not the exact canonical 252 projections"
        )
    target_tensors = accounting.get("target_tensors")
    non_target_tensors = accounting.get("dense_non_target_tensors")
    if not isinstance(target_tensors, dict) or not isinstance(non_target_tensors, dict):
        raise AcceptanceError("serialized accounting tensor maps are absent")
    module_tensor_keys: set[str] = set()
    module_bytes = 0
    module_dense_weights = 0
    for module_name, module_record in per_module.items():
        if not isinstance(module_record, dict) or not isinstance(
            module_record.get("tensors"), dict
        ):
            raise AcceptanceError(
                f"serialized accounting record is malformed for {module_name}"
            )
        tensors = module_record["tensors"]
        required = {
            f"{module_name}.{name}"
            for name in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")
        }
        if not required.issubset(tensors):
            raise AcceptanceError(
                f"serialized accounting record lacks required packed tensors for {module_name}"
            )
        layer = int(module_name.split(".")[2])
        role = module_name.rsplit(".", 1)[-1]
        expected_in, expected_out = expected_projection_dimensions(role)
        expected_metadata = expected_packed_tensor_metadata(
            ProjectionCell(layer, role, module_name, expected_in, expected_out)
        )
        metadata_mismatches = {
            name: {
                "expected": expected,
                "actual": {
                    "shape": tensors[name].get("shape"),
                    "dtype": tensors[name].get("dtype"),
                },
            }
            for name, expected in expected_metadata.items()
            if tensors[name].get("shape") != expected["shape"]
            or tensors[name].get("dtype") != expected["dtype"]
        }
        if metadata_mismatches:
            raise AcceptanceError(
                f"serialized accounting packed tensor metadata mismatch for {module_name}: {metadata_mismatches}"
            )
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
            or item["bytes"] < 0
            for item in tensors.values()
        ):
            raise AcceptanceError(
                f"serialized accounting tensor metadata is malformed for {module_name}"
            )
        tensor_bytes = sum(item["bytes"] for item in tensors.values())
        if module_record.get("bytes") != tensor_bytes:
            raise AcceptanceError(
                f"serialized accounting byte subtotal disagrees for {module_name}"
            )
        dense_count = module_record.get("dense_weight_count")
        if (
            not isinstance(dense_count, int)
            or isinstance(dense_count, bool)
            or dense_count <= 0
        ):
            raise AcceptanceError(
                f"serialized accounting dense denominator is invalid for {module_name}"
            )
        if dense_count != expected_in * expected_out:
            raise AcceptanceError(
                f"serialized accounting dense denominator is not pinned for {module_name}"
            )
        module_tensor_keys.update(tensors)
        module_bytes += tensor_bytes
        module_dense_weights += dense_count
    if module_tensor_keys != set(target_tensors):
        raise AcceptanceError(
            "serialized accounting target tensor union is inconsistent"
        )
    for module_record in per_module.values():
        for key, item in module_record["tensors"].items():
            if target_tensors[key] != item["bytes"]:
                raise AcceptanceError(
                    f"serialized accounting target byte map disagrees for {key}"
                )
    if module_bytes != accounting.get("requested_projection_tensor_bytes"):
        raise AcceptanceError(
            "serialized accounting global target bytes are inconsistent"
        )
    if module_dense_weights != accounting.get(
        "requested_projection_dense_weight_count"
    ):
        raise AcceptanceError(
            "serialized accounting global dense denominator is inconsistent"
        )
    if module_dense_weights != QWEN3_PROJECTION_DENSE_WEIGHT_COUNT:
        raise AcceptanceError(
            "serialized accounting global dense denominator is not Qwen3-8B authoritative"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in non_target_tensors.values()
    ):
        raise AcceptanceError(
            "serialized accounting non-target tensor byte map is malformed"
        )
    if sum(non_target_tensors.values()) != accounting.get(
        "dense_non_target_tensor_bytes"
    ):
        raise AcceptanceError(
            "serialized accounting non-target byte total is inconsistent"
        )
    recomputed_bpw = 8 * module_bytes / module_dense_weights
    if not math.isclose(recomputed_bpw, bpw, rel_tol=0, abs_tol=1e-12):
        raise AcceptanceError("serialized accounting effective BPW is inconsistent")
    manifests = report["manifests"]
    comparisons = manifests.get("comparisons")
    if (
        manifests.get("pairwise_disjoint") is not True
        or not isinstance(comparisons, list)
        or len(comparisons) != len(MANIFEST_DISJOINT_PAIRS)
    ):
        raise AcceptanceError("manifest pairwise-disjoint evidence is incomplete")
    if any(
        item.get("identity_overlap") or item.get("content_overlap")
        for item in comparisons
    ):
        raise AcceptanceError("manifest report contains sample leakage")
    expected_pairs = list(MANIFEST_DISJOINT_PAIRS)
    actual_pairs = [
        (item.get("left"), item.get("right"))
        for item in comparisons
        if isinstance(item, dict)
    ]
    if actual_pairs != expected_pairs or len(set(actual_pairs)) != len(
        MANIFEST_DISJOINT_PAIRS
    ):
        raise AcceptanceError(
            "manifest evidence is not the exact unique canonical split pairs"
        )
    expected_counts = {
        "calibration": 512,
        "yaqa_tuning": 512,
        "validation": 512,
        "held_out_diagnostics": 512,
        "diverse_pool_512": 512,
        "diverse_32": 32,
    }
    if manifests.get("counts") != expected_counts:
        raise AcceptanceError(
            "manifest sample counts do not match the frozen acceptance plan"
        )
    selection = manifests.get("diverse_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("verified") is not True
        or selection.get("scheme") != DIVERSE_SELECTION_SCHEME
        or selection.get("pool_count") != 512
        or selection.get("selected_count") != 32
        or len(selection.get("selected_identities", [])) != 32
        or len(selection.get("selected_content_sha256", [])) != 32
        or len(set(selection.get("selected_identities", []))) != 32
        or len(set(selection.get("selected_content_sha256", []))) != 32
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in selection.get("selected_content_sha256", [])
        )
    ):
        raise AcceptanceError(
            "deterministic diverse-32 selection evidence is incomplete"
        )
    manifest_hashes = manifests.get("manifest_sha256")
    content_hashes = manifests.get("content_jsonl_sha256")
    for label, hash_map in (
        ("manifest", manifest_hashes),
        ("content JSONL", content_hashes),
    ):
        if (
            not isinstance(hash_map, dict)
            or set(hash_map) != set(MANIFEST_SPLITS)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hash_map.values()
            )
        ):
            raise AcceptanceError(
                f"{label} SHA-256 evidence is incomplete or malformed"
            )
    quantization_streams = manifests.get("quantization_streams")
    split_mapping = {
        "calibration": "calibration",
        "yaqa": "yaqa_tuning",
        "validation": "validation",
    }
    if not isinstance(quantization_streams, dict):
        raise AcceptanceError("quantization stream content authority is absent")
    for quant_name, manifest_name in split_mapping.items():
        stream = quantization_streams.get(quant_name)
        if not isinstance(stream, dict) or stream.get("manifest_verified") is not True:
            raise AcceptanceError(
                f"quantization stream authority is absent for {quant_name}"
            )
        if stream.get("content_sha256") != content_hashes[manifest_name]:
            raise AcceptanceError(
                f"quantization stream content hash is unbound for {quant_name}"
            )
        if stream.get("identity_manifest_sha256") != manifest_hashes[manifest_name]:
            raise AcceptanceError(
                f"quantization stream manifest hash is unbound for {quant_name}"
            )
    thresholds = report["thresholds"]
    for key in ("top1_agreement_min", "diverse_32_min"):
        value = thresholds.get(key)
        if not _is_finite_real(value) or value < 0.85 or value > 1:
            raise AcceptanceError(f"{key} must be a finite fraction in [0.85, 1]")
    kl_max = thresholds.get("final_kl_max_nats")
    if not _is_finite_real(kl_max) or kl_max < 0:
        raise AcceptanceError(
            "final_kl_max_nats must be an explicit finite nonnegative KL threshold, not a percentage"
        )

    cells = report["cells"]
    if not isinstance(cells, list):
        raise AcceptanceError("cells must be a list")
    keyed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for cell in cells:
        key = (cell.get("layer"), cell.get("role"))
        if key in keyed:
            raise AcceptanceError(f"duplicate result cell {key}")
        keyed[key] = cell
    missing_cells = sorted(set(expected_cells()) - keyed.keys())
    extra_cells = sorted(set(keyed) - set(expected_cells()))
    if missing_cells or extra_cells or len(cells) != QWEN3_EXPECTED_MODULE_COUNT:
        raise AcceptanceError(
            f"incomplete result coverage: missing={missing_cells}, extra={extra_cells}"
        )
    for key, cell in keyed.items():
        expected_name = expected_projection_name(*key)
        module_name = cell.get("module")
        if not isinstance(module_name, str) or not (
            module_name == expected_name or module_name.endswith("." + expected_name)
        ):
            raise AcceptanceError(
                f"result cell {key} has invalid module identity {module_name!r}"
            )
    for scope, evidence in [
        ("global", report["global"]),
        *[(str(key), keyed[key]) for key in expected_cells()],
    ]:
        if evidence.get("coverage_complete") is not True:
            raise AcceptanceError(f"{scope} coverage is incomplete")
        top1 = evidence.get("top1_agreement")
        diverse = evidence.get("diverse_32")
        final_kl = evidence.get("final_kl_nats")
        values = {
            "top1_agreement": top1,
            "diverse_32": diverse,
            "final_kl_nats": final_kl,
        }
        if any(not _is_finite_real(value) for value in values.values()):
            raise AcceptanceError(f"{scope} has absent or non-finite metrics: {values}")
        if (
            evidence.get("sample_count") != 1056
            or evidence.get("diverse_32_sample_count") != 32
        ):
            raise AcceptanceError(
                f"{scope} metric sample coverage does not match 512 validation + 512 held-out + 32 diverse"
            )
        if (
            top1 < thresholds["top1_agreement_min"]
            and diverse < thresholds["diverse_32_min"]
        ):
            raise AcceptanceError(
                f"{scope} fails both score alternatives: top-1={top1}, diverse-32={diverse}"
            )
        if final_kl > kl_max:
            raise AcceptanceError(
                f"{scope} final KL {final_kl} nats exceeds threshold {kl_max}"
            )
    validate_controller_transcript(
        artifact.get("acceptance_controller"),
        dense_binding=dense_binding,
        parity=parity,
        report=report,
        controller_authority=controller_authority,
        trust_resources=trust_resources,
    )


def validate_acceptance_report(
    report: Mapping[str, Any],
    *,
    controller_authority: Mapping[str, Any] | None = None,
    trust_resources: _TrustedResources | None = None,
) -> None:
    """Validate with one retained trust descriptor set for the entire operation."""

    if trust_resources is not None:
        trust_resources.revalidate()
        _validate_acceptance_report_retained(
            report, controller_authority=controller_authority, trust_resources=trust_resources
        )
        return
    with _TrustedResources() as owned:
        _validate_acceptance_report_retained(report, controller_authority=controller_authority, trust_resources=owned)


__all__ = [
    "MANIFEST_SPLITS",
    "PROJECTION_BOUNDARY",
    "QVQ_OBSERVATION_HASH_SCHEME",
    "QVQ_PAYLOAD_HASH_SCHEME",
    "QWEN3_DENSE_ARTIFACT_SHA256",
    "QWEN3_EXPECTED_MODULE_COUNT",
    "QWEN3_LOCAL_TARGET",
    "QWEN3_MODEL_CONFIG",
    "QWEN3_PINNED_REVISION",
    "QWEN3_PROJECTION_DENSE_WEIGHT_COUNT",
    "QWEN3_PROJECTION_DIMENSIONS",
    "AcceptanceError",
    "ProjectionCell",
    "acceptance_observation_digest",
    "account_serialized_checkpoint",
    "account_serialized_state",
    "census_reloaded_model",
    "expected_cells",
    "expected_packed_tensor_metadata",
    "expected_projection_dimensions",
    "expected_projection_names",
    "hash_canonical_qwen3_payloads",
    "hash_qvq_module_payloads",
    "materialize_manifest",
    "seal_acceptance_observation",
    "select_diverse_32",
    "validate_acceptance_report",
    "validate_diverse_selection",
    "validate_manifest_disjointness",
    "validate_qwen3_model_artifact",
]
