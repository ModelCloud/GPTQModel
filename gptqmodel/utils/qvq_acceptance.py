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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear

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
MANIFEST_SPLITS = ("calibration", "yaqa_tuning", "validation", "held_out_diagnostics", "diverse_32")
REPORT_SCHEMA_VERSION = 1
PROJECTION_BOUNDARY = (
    "All serialized tensors whose state-dict key is the exact requested decoder projection prefix or a child of "
    "that prefix. The denominator is the sum of dense in_features*out_features for the 252 requested projections. "
    "Embeddings, norms, LM head, and every other non-target tensor are excluded from BPW and reported separately."
)


class AcceptanceError(RuntimeError):
    """Raised when acceptance evidence is absent, inconsistent, or below contract."""


@dataclass(frozen=True)
class ProjectionCell:
    layer: int
    role: str
    name: str
    in_features: int
    out_features: int

    @property
    def dense_weight_count(self) -> int:
        return self.in_features * self.out_features


def expected_projection_name(layer: int, role: str) -> str:
    stem = "self_attn" if role in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"model.layers.{layer}.{stem}.{role}"


def expected_cells() -> tuple[tuple[int, str], ...]:
    return tuple((layer, role) for layer in range(QWEN3_8B_LAYER_COUNT) for role in QWEN3_PROJECTION_ROLES)


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
            layers_index = max(index for index, part in enumerate(parts) if part == "layers")
            layer = int(parts[layers_index + 1])
        except (ValueError, IndexError):
            continue
        role = parts[-1]
        if layer not in range(QWEN3_8B_LAYER_COUNT) or role not in QWEN3_PROJECTION_ROLES:
            continue
        key = (layer, role)
        if key in found:
            raise AcceptanceError(f"duplicate requested projection cell {key}: {found[key].name!r} and {name!r}")
        if type(module) is not QVQLinear:
            raise AcceptanceError(
                f"requested projection {name!r} is {type(module).__name__}, not the exact packed QVQLinear runtime"
            )
        if module.bits > 2.0:
            raise AcceptanceError(f"requested projection {name!r} is a higher-precision W{module.bits:g} fallback")
        required = {"trellis", "SU", "SV"}
        buffers = dict(module.named_buffers(recurse=False))
        missing_payload = sorted(required - buffers.keys())
        if missing_payload or any(buffers[key].is_meta for key in required if key in buffers):
            raise AcceptanceError(f"requested projection {name!r} lacks loaded packed payload: {missing_payload}")
        found[key] = ProjectionCell(layer, role, name, module.in_features, module.out_features)

    missing = sorted(set(expected_cells()) - found.keys())
    extra = sorted(set(found) - set(expected_cells()))
    if missing or extra or len(found) != QWEN3_EXPECTED_MODULE_COUNT:
        raise AcceptanceError(
            f"projection census must be exactly {QWEN3_EXPECTED_MODULE_COUNT}; found={len(found)}, "
            f"missing={missing}, extra={extra}"
        )
    return [found[key] for key in expected_cells()]


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def account_serialized_state(
    state: Mapping[str, torch.Tensor], cells: Sequence[ProjectionCell], *, maximum_bpw: float = 2.1
) -> dict[str, Any]:
    """Account actual serialized tensor payload bytes at the projection boundary.

    ``state`` must come from the freshly reloaded model's state dict.  Safetensors
    stores tensor payloads without compression, so ``numel*element_size`` is the
    serialized data-region byte count; container header bytes are reported but not
    assigned to tensors by this projection-only boundary.
    """

    if not math.isfinite(maximum_bpw) or maximum_bpw <= 0:
        raise ValueError("maximum_bpw must be finite and positive")
    prefixes = sorted((cell.name, cell) for cell in cells)
    target: dict[str, int] = {}
    non_target: dict[str, int] = {}
    unassigned_cells: set[str] = {cell.name for cell in cells}
    per_module: dict[str, dict[str, Any]] = {
        cell.name: {"bytes": 0, "dense_weight_count": cell.dense_weight_count, "tensors": {}} for cell in cells
    }
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise AcceptanceError(f"state entry {key!r} is not a tensor")
        matches = [(prefix, cell) for prefix, cell in prefixes if key == prefix or key.startswith(prefix + ".")]
        if len(matches) > 1:
            raise AcceptanceError(f"serialized tensor {key!r} maps to multiple requested projections")
        size = _tensor_nbytes(tensor)
        if matches:
            prefix, _cell = matches[0]
            target[key] = size
            per_module[prefix]["bytes"] += size
            per_module[prefix]["tensors"][key] = {"bytes": size, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            unassigned_cells.discard(prefix)
        else:
            non_target[key] = size
    if unassigned_cells:
        raise AcceptanceError(f"requested projections have no serialized tensors: {sorted(unassigned_cells)}")
    dense_weight_count = sum(cell.dense_weight_count for cell in cells)
    target_bytes = sum(target.values())
    effective_bpw = (8 * target_bytes) / dense_weight_count
    report = {
        "boundary": PROJECTION_BOUNDARY,
        "container_header_bytes": "excluded/unassigned",
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
        raise AcceptanceError(f"serialized projection effective BPW {effective_bpw:.9f} exceeds {maximum_bpw:.9f}")
    return report


def canonical_content(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def materialize_manifest(
    split: str, records: Iterable[Mapping[str, Any]], output: Path, *, expected_count: int | None = None
) -> dict[str, Any]:
    if split not in MANIFEST_SPLITS:
        raise ValueError(f"unknown manifest split {split!r}")
    rows = []
    seen_identities: set[str] = set()
    seen_hashes: set[str] = set()
    for ordinal, record in enumerate(records):
        if "identity" not in record or "content" not in record:
            raise AcceptanceError(f"{split} record {ordinal} requires identity and content")
        identity = str(record["identity"]).strip()
        if not identity:
            raise AcceptanceError(f"{split} record {ordinal} has an empty identity")
        content_hash = hashlib.sha256(canonical_content(record["content"])).hexdigest()
        if identity in seen_identities or content_hash in seen_hashes:
            raise AcceptanceError(f"duplicate identity or content within {split}: {identity}")
        seen_identities.add(identity)
        seen_hashes.add(content_hash)
        rows.append({"ordinal": ordinal, "identity": identity, "content_sha256": content_hash})
    if expected_count is not None and len(rows) != expected_count:
        raise AcceptanceError(f"{split} requires exactly {expected_count} records, found {len(rows)}")
    if split == "diverse_32" and len(rows) != 32:
        raise AcceptanceError(f"diverse_32 requires exactly 32 records, found {len(rows)}")
    payload = {"schema_version": 1, "split": split, "count": len(rows), "samples": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_manifest_disjointness(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = sorted(set(MANIFEST_SPLITS) - manifests.keys())
    if missing:
        raise AcceptanceError(f"missing required sample manifests: {missing}")
    sets: dict[str, tuple[set[str], set[str]]] = {}
    for split in MANIFEST_SPLITS:
        payload = manifests[split]
        if payload.get("split") != split or not isinstance(payload.get("samples"), list):
            raise AcceptanceError(f"invalid manifest schema for {split}")
        identities = {str(row.get("identity")) for row in payload["samples"]}
        hashes = {str(row.get("content_sha256")) for row in payload["samples"]}
        if len(identities) != len(payload["samples"]) or len(hashes) != len(payload["samples"]):
            raise AcceptanceError(f"duplicate identity or content in {split}")
        sets[split] = identities, hashes
    comparisons = []
    for left_index, left in enumerate(MANIFEST_SPLITS):
        for right in MANIFEST_SPLITS[left_index + 1 :]:
            identity_overlap = sorted(sets[left][0] & sets[right][0])
            content_overlap = sorted(sets[left][1] & sets[right][1])
            comparisons.append(
                {"left": left, "right": right, "identity_overlap": identity_overlap, "content_overlap": content_overlap}
            )
            if identity_overlap or content_overlap:
                raise AcceptanceError(
                    f"sample leakage between {left} and {right}: identities={identity_overlap}, hashes={content_overlap}"
                )
    return {"pairwise_disjoint": True, "comparisons": comparisons}


def validate_acceptance_report(report: Mapping[str, Any]) -> None:
    """Validate complete machine-readable evidence and every declared gate."""

    required = {"schema_version", "artifact", "census", "accounting", "manifests", "thresholds", "global", "cells"}
    missing = sorted(required - report.keys())
    if missing:
        raise AcceptanceError(f"acceptance report missing fields: {missing}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise AcceptanceError("unsupported acceptance report schema version")
    artifact = report["artifact"]
    if artifact.get("fresh_reload_verified") is not True:
        raise AcceptanceError("artifact was not verified through a fresh reload")
    hashes = artifact.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes.values()
    ):
        raise AcceptanceError("artifact requires SHA-256 identities for serialized checkpoint files")
    census = report["census"]
    if census != {"expected": QWEN3_EXPECTED_MODULE_COUNT, "actual": QWEN3_EXPECTED_MODULE_COUNT, "complete": True}:
        raise AcceptanceError("checkpoint census is not exactly 252 complete projection modules")
    accounting = report["accounting"]
    bpw = accounting.get("effective_bpw")
    maximum_bpw = accounting.get("maximum_bpw")
    if accounting.get("boundary") != PROJECTION_BOUNDARY:
        raise AcceptanceError("accounting boundary is absent or changed")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in (bpw, maximum_bpw)):
        raise AcceptanceError("accounting BPW evidence is absent or non-finite")
    if maximum_bpw > 2.1 or bpw > maximum_bpw:
        raise AcceptanceError("serialized projection BPW exceeds the W2/W2.1 contract")
    manifests = report["manifests"]
    comparisons = manifests.get("comparisons")
    if manifests.get("pairwise_disjoint") is not True or not isinstance(comparisons, list) or len(comparisons) != 10:
        raise AcceptanceError("manifest pairwise-disjoint evidence is incomplete")
    if any(item.get("identity_overlap") or item.get("content_overlap") for item in comparisons):
        raise AcceptanceError("manifest report contains sample leakage")
    thresholds = report["thresholds"]
    for key in ("top1_agreement_min", "diverse_32_min"):
        value = thresholds.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.85 or value > 1:
            raise AcceptanceError(f"{key} must be a finite fraction in [0.85, 1]")
    kl_max = thresholds.get("final_kl_max_nats")
    if not isinstance(kl_max, (int, float)) or not math.isfinite(kl_max) or kl_max < 0:
        raise AcceptanceError("final_kl_max_nats must be an explicit finite nonnegative KL threshold, not a percentage")

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
        raise AcceptanceError(f"incomplete result coverage: missing={missing_cells}, extra={extra_cells}")
    for scope, evidence in [("global", report["global"]), *[(str(key), keyed[key]) for key in expected_cells()]]:
        if evidence.get("coverage_complete") is not True:
            raise AcceptanceError(f"{scope} coverage is incomplete")
        top1 = evidence.get("top1_agreement")
        diverse = evidence.get("diverse_32")
        final_kl = evidence.get("final_kl_nats")
        values = {"top1_agreement": top1, "diverse_32": diverse, "final_kl_nats": final_kl}
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
            raise AcceptanceError(f"{scope} has absent or non-finite metrics: {values}")
        if top1 < thresholds["top1_agreement_min"]:
            raise AcceptanceError(f"{scope} top-1 agreement {top1} is below threshold")
        if diverse < thresholds["diverse_32_min"]:
            raise AcceptanceError(f"{scope} diverse-32 score {diverse} is below threshold")
        if final_kl > kl_max:
            raise AcceptanceError(f"{scope} final KL {final_kl} nats exceeds threshold {kl_max}")


__all__ = [
    "MANIFEST_SPLITS",
    "PROJECTION_BOUNDARY",
    "QWEN3_EXPECTED_MODULE_COUNT",
    "AcceptanceError",
    "ProjectionCell",
    "account_serialized_state",
    "census_reloaded_model",
    "expected_cells",
    "materialize_manifest",
    "validate_acceptance_report",
    "validate_manifest_disjointness",
]
