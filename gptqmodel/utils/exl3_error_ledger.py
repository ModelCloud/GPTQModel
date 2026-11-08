# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Durable, content-bound EXL3 quantization error ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "ds4rt.exl3-error-ledger"
LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "ds4rt-exl3-error-ledger.jsonl"
LEDGER_MANIFEST_FILENAME = "ds4rt-exl3-error-ledger.manifest.json"
JOURNAL_ENV = "GPTQMODEL_EXL3_ERROR_JOURNAL"
ROUTE_EVIDENCE_SCHEMA = "ds4rt.exl3-natural-route"
ROUTE_EVIDENCE_SCHEMA_VERSION = 1
ZERO_ROUTE_RECOVERY_SCHEMA = "ds4rt.exl3-zero-route-recovery"
ZERO_ROUTE_RECOVERY_SCHEMA_VERSION = 1
ZERO_ROUTE_RECOVERY_TRIGGER = "natural-route-count-below-1024"
ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE = "same-fixed-calibration-selection"
ZERO_ROUTE_RECOVERY_CAPTURE_METHOD = (
    "direct-expert-router-ranks-7-12-then-identity-residual"
)
ZERO_ROUTE_RECOVERY_SELECTION_POLICY = (
    "rank-ascending-then-fixed-replay-order-v1"
)
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN = 7
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX = 12
ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT = 1024
ZERO_ROUTE_RECOVERY_SELECTION_CAP = 1024
ZERO_ROUTE_RECOVERY_IDENTITY_POLICY = (
    "normalized-2i-residual-to-effective-count-1024-v2"
)
ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR = "router-near-rows"
ZERO_ROUTE_RECOVERY_MODE_IDENTITY = "identity-hessian"
ZERO_ROUTE_RECOVERY_MODE_MIXED = "empirical-plus-identity-hessian"
ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA = (
    "ds4rt.exl3-zero-route-recovery-authorization"
)
ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA_VERSION = 1
ZERO_ROUTE_RECOVERY_AUTHORIZATION_KINDS = {
    "immutable-family-join",
    "content-bound-execution-upgrade",
}
ZERO_ROUTE_RECOVERY_RECIPE_KEY = "zero_route_recovery_recipe"
PROJECTION_PROVENANCE_COMPACTION_CONTRACT = (
    "gptqmodel.exl3-projection-provenance-compaction-v1"
)

_BASE_EXPERT = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_MTP_EXPERT = re.compile(
    r"^mtp\.(?:layers\.)?(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_GLM5_NEXT_EXPERT = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_PROJECTION_NAMES = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}
_JOURNAL_INDEX_LOCK = threading.Lock()
_JOURNAL_INDEX: dict[Path, tuple[tuple[int, int, int, int], set[str]]] = {}


def _finite_json_value(value: Any, path: str = "record") -> Any:
    """Deep-copy a record while rejecting values JSON cannot bind safely."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite EXL3 ledger value at {path}: {value}")
        return value
    if isinstance(value, dict):
        return {
            str(key): _finite_json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _finite_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported EXL3 ledger value at {path}: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compact_projection_provenance(
    provenance: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Replace repeated recovery inventories with their content identities.

    A resumed run can carry a complete seed-file inventory in its run
    provenance. Repeating that inventory in every projection record is
    quadratic publication metadata, even though ``inventory_sha256`` already
    binds the exact list. Keep the auditable counts, byte total, source root,
    and digests while removing only the repeated list and duplicate family
    join. Older checkpoint records remain valid and can be normalized only
    when they are emitted into the final artifact ledger.
    """

    if provenance is None:
        return None
    clean = _finite_json_value(provenance, "provenance")
    if not isinstance(clean, dict):
        raise TypeError("EXL3 projection provenance must be a mapping")
    run = clean.get("run")
    if not isinstance(run, dict):
        return clean
    seed = run.get("projection_checkpoint_seed")
    if not isinstance(seed, dict):
        return clean
    if (
        seed.get("provenance_compaction_contract")
        == PROJECTION_PROVENANCE_COMPACTION_CONTRACT
    ):
        return clean

    files = seed.get("files")
    seed_family_join = seed.get("family_join")
    if files is None and seed_family_join is None:
        return clean
    summary = {
        key: deepcopy(seed[key])
        for key in (
            "contract",
            "root",
            "checkpoint_count",
            "total_bytes",
            "inventory_sha256",
        )
        if key in seed
    }
    summary["provenance_compaction_contract"] = (
        PROJECTION_PROVENANCE_COMPACTION_CONTRACT
    )
    if files is not None:
        files_sha256 = hashlib.sha256(_canonical_json_bytes(files)).hexdigest()
        inventory_sha256 = seed.get("inventory_sha256")
        if inventory_sha256 is not None and inventory_sha256 != files_sha256:
            raise ValueError(
                "EXL3 projection checkpoint seed inventory digest differs"
            )
        summary["files_sha256"] = files_sha256
    if seed_family_join is not None:
        family_join = clean.get("family_join")
        if family_join is not None and seed_family_join != family_join:
            raise ValueError(
                "EXL3 projection checkpoint seed family join differs"
            )
        summary["family_join_sha256"] = hashlib.sha256(
            _canonical_json_bytes(seed_family_join)
        ).hexdigest()
    run["projection_checkpoint_seed"] = summary
    return clean


def compact_projection_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return one unbound projection record with compact run provenance."""

    if not isinstance(record, dict):
        raise TypeError("EXL3 projection record must be a mapping")
    clean = dict(record)
    clean.pop("record_sha256", None)
    if "provenance" in clean:
        clean["provenance"] = compact_projection_provenance(
            clean.get("provenance")
        )
    return _finite_json_value(clean)


def routed_expert_identity(
    module_full_name: str, *, family_join: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Map a GPTQModel routed projection name to its stable DS4 identity."""

    glm5_next = _GLM5_NEXT_EXPERT.fullmatch(module_full_name)
    if glm5_next is not None:
        layer = int(glm5_next.group("layer"))
        # Qwen has the same nested target prefix as GLM5-Next, but its target
        # continues beyond layer 45 and its draft lives at mtp.layers.0. Bind
        # that distinction to immutable run provenance, preserving old GLM
        # checkpoint identities when no model type was recorded.
        qwen = isinstance(family_join, dict) and family_join.get("model_type") in {
            "qwen4_exp", "qwen4_exp_text"
        }
        return {
            "block_namespace": "mtp" if layer == 45 and not qwen else "base",
            "logical_layer": layer,
            "expert": int(glm5_next.group("expert")),
            "projection": _PROJECTION_NAMES[glm5_next.group("projection")],
        }

    for namespace, pattern in (("base", _BASE_EXPERT), ("mtp", _MTP_EXPERT)):
        match = pattern.fullmatch(module_full_name)
        if match is None:
            continue
        return {
            "block_namespace": namespace,
            "logical_layer": int(match.group("layer")),
            "expert": int(match.group("expert")),
            "projection": _PROJECTION_NAMES[match.group("projection")],
        }
    return None


def routed_expert_processor_layer(identity: dict[str, Any], *, family_join=None) -> int:
    """Translate a logical draft index to its integrated traversal index."""
    logical = identity["logical_layer"]
    if (
        identity["block_namespace"] == "mtp"
        and isinstance(family_join, dict)
        and family_join.get("model_type") in {"qwen4_exp", "qwen4_exp_text"}
    ):
        target_layers = family_join.get("target_layer_count")
        if isinstance(target_layers, bool) or not isinstance(target_layers, int) or target_layers <= 0:
            raise ValueError("Qwen integrated MTP requires its bound target layer count")
        return target_layers + logical
    return logical


def route_evidence_required(provenance: dict[str, Any] | None) -> bool:
    """Return whether one run explicitly requires natural-route evidence."""

    if not isinstance(provenance, dict):
        return False
    family_join = provenance.get("family_join")
    return (
        isinstance(family_join, dict)
        and family_join.get("route_evidence_contract") == ROUTE_EVIDENCE_SCHEMA
    )


def zero_route_recovery_enabled(provenance: dict[str, Any] | None) -> bool:
    """Return whether this immutable family permits under-coverage top-up."""

    if not isinstance(provenance, dict):
        return False
    family_join = provenance.get("family_join")
    return (
        isinstance(family_join, dict)
        and family_join.get("zero_route_recovery_contract")
        == ZERO_ROUTE_RECOVERY_SCHEMA
    )


def zero_route_recovery_recipe(
    family_join: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the content-bound route-recovery recipe for one model family.

    The original DeepSeek policy remains the default so existing immutable
    plans keep their byte-for-byte contract. New model families can bind a
    different adjacent-rank window (for example ranks 9--16 for a top-8 GLM
    router) without changing global quantizer behavior.
    """

    default = {
        "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
        "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
        "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
        "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
        "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
        "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
        "selection_cap": ZERO_ROUTE_RECOVERY_SELECTION_CAP,
        "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
        "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    }
    configured = (
        family_join.get(ZERO_ROUTE_RECOVERY_RECIPE_KEY)
        if isinstance(family_join, dict)
        else None
    )
    if configured is None:
        return default
    clean = _finite_json_value(configured, ZERO_ROUTE_RECOVERY_RECIPE_KEY)
    if not isinstance(clean, dict) or set(clean) != set(default):
        raise ValueError("EXL3 zero-route recovery recipe has invalid fields")
    integer_fields = (
        "candidate_rank_min",
        "candidate_rank_max",
        "selection_cap",
        "target_sample_count",
    )
    if (
        any(
            isinstance(clean.get(field), bool)
            or not isinstance(clean.get(field), int)
            or clean[field] <= 0
            for field in integer_fields
        )
        or clean["candidate_rank_max"] < clean["candidate_rank_min"]
        or clean["selection_cap"] > clean["target_sample_count"]
        or any(
            not isinstance(clean.get(field), str) or not clean[field]
            for field in (
                "trigger",
                "sample_source",
                "capture_method",
                "selection_policy",
                "identity_calibration_policy",
            )
        )
    ):
        raise ValueError("EXL3 zero-route recovery recipe is invalid")
    return clean


def validate_zero_route_recovery_authorization(
    authorization: dict[str, Any],
    *,
    family_join: dict[str, Any],
) -> dict[str, Any]:
    """Validate the immutable authority that permits under-coverage top-up."""

    clean = _finite_json_value(
        authorization,
        "zero_route_recovery_authorization",
    )
    recipe = zero_route_recovery_recipe(family_join)
    family_join_sha256 = hashlib.sha256(
        _canonical_json_bytes(family_join)
    ).hexdigest()
    digest = clean.get("authorization_sha256")
    if (
        clean.get("schema") != ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA
        or clean.get("schema_version")
        != ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA_VERSION
        or clean.get("kind") not in ZERO_ROUTE_RECOVERY_AUTHORIZATION_KINDS
        or clean.get("recovery_contract") != ZERO_ROUTE_RECOVERY_SCHEMA
        or clean.get("trigger") != recipe["trigger"]
        or clean.get("sample_source") != recipe["sample_source"]
        or clean.get("capture_method") != recipe["capture_method"]
        or clean.get("selection_policy") != recipe["selection_policy"]
        or clean.get("candidate_rank_min") != recipe["candidate_rank_min"]
        or clean.get("candidate_rank_max") != recipe["candidate_rank_max"]
        or clean.get("target_sample_count") != recipe["target_sample_count"]
        or clean.get("identity_calibration_policy")
        != recipe["identity_calibration_policy"]
        or clean.get("family_join_sha256") != family_join_sha256
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("EXL3 zero-route recovery authorization is invalid")
    if (
        clean["kind"] == "immutable-family-join"
        and (
            family_join.get("zero_route_recovery_contract")
            != ZERO_ROUTE_RECOVERY_SCHEMA
            or digest != family_join_sha256
        )
    ):
        raise ValueError("EXL3 family-join recovery authorization is invalid")
    return clean


def validate_route_evidence(
    evidence: dict[str, Any],
    *,
    identity: dict[str, Any],
    sample_count: int,
    allow_zero: bool = False,
) -> dict[str, Any]:
    """Validate one routed expert's exposure and gate-mass accounting."""

    clean = _finite_json_value(evidence, "route_evidence")
    integer_fields = (
        "router_calls",
        "router_token_count",
        "router_selected_route_count",
        "router_top_k",
        "expert_route_count",
    )
    if (
        clean.get("schema") != ROUTE_EVIDENCE_SCHEMA
        or clean.get("schema_version") != ROUTE_EVIDENCE_SCHEMA_VERSION
        or any(
            isinstance(clean.get(field), bool)
            or not isinstance(clean.get(field), int)
            or clean[field] <= 0
            for field in integer_fields[:-1]
        )
        or isinstance(clean.get("expert_route_count"), bool)
        or not isinstance(clean.get("expert_route_count"), int)
        or clean["expert_route_count"] < (0 if allow_zero else 1)
        or clean.get("block_namespace") != identity["block_namespace"]
        or clean.get("logical_layer") != identity["logical_layer"]
        or clean.get("expert") != identity["expert"]
        or clean["expert_route_count"] != int(sample_count)
        or clean["router_selected_route_count"]
        != clean["router_token_count"] * clean["router_top_k"]
        or not isinstance(clean.get("router_weight_dtypes"), list)
        or not clean["router_weight_dtypes"]
        or not all(
            isinstance(value, str) and value for value in clean["router_weight_dtypes"]
        )
        or not isinstance(clean.get("mask_modes"), list)
        or not clean["mask_modes"]
        or not all(isinstance(value, str) and value for value in clean["mask_modes"])
    ):
        raise ValueError("EXL3 natural-route evidence has an invalid contract")

    numeric_fields = (
        "expert_gate_weight_sum",
        "expert_gate_squared_mass",
        "total_gate_weight_sum",
        "total_gate_squared_mass",
        "expert_route_fraction",
        "expert_gate_weight_mass_fraction",
        "expert_gate_squared_mass_fraction",
        "expert_gate_weight_mean",
        "expert_gate_weight_rms",
    )
    if any(
        isinstance(clean.get(field), bool)
        or not isinstance(clean.get(field), (int, float))
        or clean[field] < 0
        for field in numeric_fields
    ):
        raise ValueError("EXL3 natural-route evidence has invalid gate metrics")

    route_count = clean["expert_route_count"]
    selected_count = clean["router_selected_route_count"]
    gate_sum = float(clean["expert_gate_weight_sum"])
    gate_sq = float(clean["expert_gate_squared_mass"])
    total_gate_sum = float(clean["total_gate_weight_sum"])
    total_gate_sq = float(clean["total_gate_squared_mass"])
    expected = {
        "expert_route_fraction": route_count / selected_count,
        "expert_gate_weight_mass_fraction": gate_sum / max(total_gate_sum, 1e-30),
        "expert_gate_squared_mass_fraction": gate_sq / max(total_gate_sq, 1e-30),
        "expert_gate_weight_mean": gate_sum / max(route_count, 1),
        "expert_gate_weight_rms": math.sqrt(gate_sq / max(route_count, 1)),
    }
    if (
        route_count > selected_count
        or (route_count == 0 and (gate_sum != 0 or gate_sq != 0))
        or (route_count > 0 and (gate_sum <= 0 or gate_sq <= 0))
        or total_gate_sum < gate_sum
        or total_gate_sq < gate_sq
        or any(
            not math.isclose(float(clean[field]), value, rel_tol=1e-12, abs_tol=1e-15)
            for field, value in expected.items()
        )
    ):
        raise ValueError("EXL3 natural-route evidence has inconsistent gate metrics")
    return clean


def validate_zero_route_recovery(
    evidence: dict[str, Any],
    *,
    identity: dict[str, Any],
    sample_count: int,
    family_join: dict[str, Any] | None = None,
    expected_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one deterministic same-selection under-coverage top-up."""

    clean = _finite_json_value(evidence, "zero_route_recovery")
    integer_fields = (
        "natural_sample_count",
        "router_augmented_sample_count",
        "identity_calibration_count",
        "total_sample_count",
        "forced_pass_count",
    )
    authorization = clean.get("authorization")
    candidate_histogram = clean.get("candidate_rank_histogram")
    candidate_gap = clean.get("candidate_score_gap")
    recovery_mode = clean.get("recovery_mode")
    if not isinstance(family_join, dict):
        raise ValueError("EXL3 zero-route recovery requires family identity")
    recipe = zero_route_recovery_recipe(family_join)
    validated_authorization = validate_zero_route_recovery_authorization(
        authorization,
        family_join=family_join,
    ) if isinstance(authorization, dict) else None
    if (
        clean.get("schema") != ZERO_ROUTE_RECOVERY_SCHEMA
        or clean.get("schema_version") != ZERO_ROUTE_RECOVERY_SCHEMA_VERSION
        or clean.get("trigger") != recipe["trigger"]
        or clean.get("sample_source") != recipe["sample_source"]
        or clean.get("capture_method") != recipe["capture_method"]
        or clean.get("selection_policy") != recipe["selection_policy"]
        or clean.get("candidate_rank_min") != recipe["candidate_rank_min"]
        or clean.get("candidate_rank_max") != recipe["candidate_rank_max"]
        or clean.get("selection_cap") != recipe["selection_cap"]
        or clean.get("target_sample_count") != recipe["target_sample_count"]
        or clean.get("identity_calibration_policy")
        != recipe["identity_calibration_policy"]
        or clean.get("block_namespace") != identity["block_namespace"]
        or clean.get("logical_layer") != identity["logical_layer"]
        or clean.get("expert") != identity["expert"]
        or any(
            isinstance(clean.get(field), bool)
            or not isinstance(clean.get(field), int)
            for field in integer_fields
        )
        or not 0
        <= clean["natural_sample_count"]
        < recipe["target_sample_count"]
        or clean["total_sample_count"]
        != recipe["target_sample_count"]
        or clean["total_sample_count"] != int(sample_count)
        or clean["forced_pass_count"] != 1
        or isinstance(clean.get("candidate_rows_observed"), bool)
        or not isinstance(clean.get("candidate_rows_observed"), int)
        or clean["candidate_rows_observed"] < 0
        or isinstance(clean.get("candidate_rows_selected"), bool)
        or not isinstance(clean.get("candidate_rows_selected"), int)
        or not 0
        <= clean["candidate_rows_selected"]
        <= min(
            clean["candidate_rows_observed"],
            recipe["selection_cap"],
        )
        or not isinstance(candidate_histogram, dict)
        or set(candidate_histogram)
        != {
            str(rank)
            for rank in range(
                recipe["candidate_rank_min"],
                recipe["candidate_rank_max"] + 1,
            )
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in candidate_histogram.values()
        )
        or sum(candidate_histogram.values())
        != clean["candidate_rows_selected"]
        or recovery_mode
        not in {
            ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR,
            ZERO_ROUTE_RECOVERY_MODE_IDENTITY,
            ZERO_ROUTE_RECOVERY_MODE_MIXED,
        }
        or validated_authorization is None
        or (
            expected_authorization is not None
            and validated_authorization != expected_authorization
        )
    ):
        raise ValueError("EXL3 zero-route recovery evidence has an invalid contract")
    selected = clean["candidate_rows_selected"]
    observed = clean["candidate_rows_observed"]
    if recovery_mode == ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR:
        gap_fields = ("min", "mean", "max")
        if (
            clean["identity_calibration_count"] != 0
            or clean["router_augmented_sample_count"]
            != recipe["target_sample_count"]
            - clean["natural_sample_count"]
            or clean["router_augmented_sample_count"] <= 0
            or clean["natural_sample_count"]
            + clean["router_augmented_sample_count"]
            != clean["total_sample_count"]
            or observed <= 0
            or selected <= 0
            or clean["router_augmented_sample_count"] != selected
            or not isinstance(candidate_gap, dict)
            or any(
                isinstance(candidate_gap.get(field), bool)
                or not isinstance(candidate_gap.get(field), (int, float))
                or not math.isfinite(float(candidate_gap[field]))
                or candidate_gap[field] < 0
                for field in gap_fields
            )
            or not candidate_gap["min"]
            <= candidate_gap["mean"]
            <= candidate_gap["max"]
        ):
            raise ValueError("EXL3 router-near recovery evidence is invalid")
    elif recovery_mode == ZERO_ROUTE_RECOVERY_MODE_MIXED:
        if (
            clean["identity_calibration_count"] <= 0
            or clean["natural_sample_count"]
            + clean["router_augmented_sample_count"]
            + clean["identity_calibration_count"]
            != clean["total_sample_count"]
            or clean["router_augmented_sample_count"] != selected
            or selected > observed
            or clean["natural_sample_count"] + selected <= 0
            or (
                selected > 0
                and (
                    not isinstance(candidate_gap, dict)
                    or any(
                        isinstance(candidate_gap.get(field), bool)
                        or not isinstance(candidate_gap.get(field), (int, float))
                        or not math.isfinite(float(candidate_gap[field]))
                        or candidate_gap[field] < 0
                        for field in ("min", "mean", "max")
                    )
                    or not candidate_gap["min"]
                    <= candidate_gap["mean"]
                    <= candidate_gap["max"]
                )
            )
            or (selected == 0 and candidate_gap is not None)
        ):
            raise ValueError("EXL3 mixed recovery evidence is invalid")
    elif (
        clean["natural_sample_count"] != 0
        or clean["router_augmented_sample_count"] != 0
        or clean["identity_calibration_count"]
        != recipe["target_sample_count"]
        or observed != 0
        or selected != 0
        or candidate_gap is not None
    ):
        raise ValueError("EXL3 identity-Hessian recovery evidence is invalid")
    return clean


def build_projection_record(
    *,
    module_full_name: str,
    layer_index: int | None,
    bits: int,
    codebook: str,
    sample_count: int,
    duration_seconds: float,
    encoded_bytes: int,
    device_names: list[str],
    quantizer_metrics: dict[str, Any],
    provenance: dict[str, Any] | None,
    route_evidence: dict[str, Any] | None = None,
    zero_route_recovery: dict[str, Any] | None = None,
    compact_provenance: bool = True,
) -> dict[str, Any]:
    """Construct one projection record without discarding raw error terms."""

    record = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_kind": "projection",
        "module": module_full_name,
        "processor_layer_index": layer_index,
        "bits": int(bits),
        "codebook": str(codebook),
        "sample_count": int(sample_count),
        "duration_seconds": float(duration_seconds),
        "encoded_bytes": int(encoded_bytes),
        "devices": [str(device) for device in device_names],
        "quantizer_metrics": deepcopy(quantizer_metrics),
        "provenance": (
            compact_projection_provenance(provenance)
            if compact_provenance
            else deepcopy(provenance) if provenance is not None else None
        ),
    }
    identity = routed_expert_identity(
        module_full_name,
        family_join=provenance.get("family_join") if isinstance(provenance, dict) else None,
    )
    if identity is not None:
        record.update(identity)
        recovery = None
        natural_sample_count = sample_count
        if zero_route_recovery is not None:
            recovery = validate_zero_route_recovery(
                zero_route_recovery,
                identity=identity,
                sample_count=sample_count,
                family_join=(
                    provenance.get("family_join")
                    if isinstance(provenance, dict)
                    else None
                ),
            )
            natural_sample_count = recovery["natural_sample_count"]
            record["zero_route_recovery"] = recovery
        if route_evidence is not None:
            record["route_evidence"] = validate_route_evidence(
                route_evidence,
                identity=identity,
                sample_count=natural_sample_count,
                allow_zero=recovery is not None,
            )
        elif route_evidence_required(provenance):
            raise ValueError(
                f"EXL3 natural-route evidence is required for `{module_full_name}`"
            )
    elif route_evidence is not None or zero_route_recovery is not None:
        raise ValueError(
            "EXL3 route/recovery evidence can only describe routed experts"
        )
    return _finite_json_value(record)


def _family_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    projection_bits = {
        record["projection"]: int(record["bits"]) for record in records
    }
    reconstructions = [
        record["quantizer_metrics"]["reconstruction"] for record in records
    ]
    error_sum_sq = sum(item["error_sum_sq"] for item in reconstructions)
    reference_sum_sq = sum(item["reference_sum_sq"] for item in reconstructions)
    element_count = sum(item["element_count"] for item in reconstructions)

    hessian_numerators = [
        record["quantizer_metrics"].get("hessian_weighted_error_numerator")
        for record in records
    ]
    hessian_denominators = [
        record["quantizer_metrics"].get("hessian_weighted_reference_denominator")
        for record in records
    ]
    hessian_complete = all(
        value is not None for value in hessian_numerators + hessian_denominators
    )
    hessian_numerator = sum(hessian_numerators) if hessian_complete else None
    hessian_denominator = sum(hessian_denominators) if hessian_complete else None

    route_evidence = [record.get("route_evidence") for record in records]
    if any(value is not None for value in route_evidence):
        if any(value is None for value in route_evidence) or any(
            _canonical_json_bytes(value) != _canonical_json_bytes(route_evidence[0])
            for value in route_evidence[1:]
        ):
            raise ValueError(
                "EXL3 expert-family projections have inconsistent route evidence"
            )

    zero_route_recovery = [
        record.get("zero_route_recovery") for record in records
    ]
    if any(value is not None for value in zero_route_recovery):
        if any(value is None for value in zero_route_recovery) or any(
            _canonical_json_bytes(value)
            != _canonical_json_bytes(zero_route_recovery[0])
            for value in zero_route_recovery[1:]
        ):
            raise ValueError(
                "EXL3 expert-family projections have inconsistent zero-route recovery"
            )

    family = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "record_kind": "expert_family",
        "block_namespace": first["block_namespace"],
        "logical_layer": first["logical_layer"],
        "expert": first["expert"],
        # Keep the base tier in the legacy scalar while recording exact
        # projection tiers explicitly. Uniform ledgers remain byte-compatible
        # apart from this additive field.
        "bits": min(projection_bits.values()),
        "projection_bits": projection_bits,
        "mixed_bits": len(set(projection_bits.values())) > 1,
        "codebook": first["codebook"],
        "projections": [record["projection"] for record in records],
        "projection_modules": [record["module"] for record in records],
        "sample_counts": [record["sample_count"] for record in records],
        "duration_seconds": sum(record["duration_seconds"] for record in records),
        "encoded_bytes": sum(record["encoded_bytes"] for record in records),
        "provenance": {
            "family_join": deepcopy(_family_join_provenance(first)),
            "projections": {
                record["projection"]: deepcopy(record.get("provenance"))
                for record in records
            },
        },
        "aggregate_metrics": {
            "error_sum_sq": error_sum_sq,
            "reference_sum_sq": reference_sum_sq,
            "element_count": element_count,
            "mse": error_sum_sq / max(element_count, 1),
            "nmse": error_sum_sq / max(reference_sum_sq, 1e-20),
            "relative_frobenius": math.sqrt(
                max(error_sum_sq / max(reference_sum_sq, 1e-20), 0.0)
            ),
            "max_abs_error": max(item["max_abs_error"] for item in reconstructions),
            "hessian_weighted_error_numerator": hessian_numerator,
            "hessian_weighted_reference_denominator": hessian_denominator,
            "hessian_weighted_relative_error": (
                hessian_numerator / max(hessian_denominator, 1e-8)
                if hessian_complete
                else None
            ),
            "hessian_metric_status": "ok" if hessian_complete else "incomplete",
        },
    }
    if route_evidence[0] is not None:
        family["route_evidence"] = deepcopy(route_evidence[0])
    if zero_route_recovery[0] is not None:
        family["zero_route_recovery"] = deepcopy(zero_route_recovery[0])
    return _finite_json_value(family)


def _family_join_provenance(record: dict[str, Any]) -> Any:
    provenance = record.get("provenance")
    if isinstance(provenance, dict) and "family_join" in provenance:
        return provenance["family_join"]
    return provenance


def derive_family_records(
    projection_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join complete w1/w2/w3 projection triples into expert-family records."""

    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in projection_records:
        required = ("block_namespace", "logical_layer", "expert", "projection")
        if any(field not in record for field in required):
            continue
        key = (
            record["block_namespace"],
            record["logical_layer"],
            record["expert"],
            record["codebook"],
            _canonical_json_bytes(_family_join_provenance(record)),
        )
        projections = groups.setdefault(key, {})
        projection = record["projection"]
        if projection in projections:
            raise ValueError(
                "duplicate EXL3 ledger projection for "
                f"{key[0]} layer {key[1]} expert {key[2]} {projection}"
            )
        projections[projection] = record

    families = []
    for projections in groups.values():
        if set(projections) != {"w1", "w2", "w3"}:
            continue
        families.append(
            _family_record([projections[name] for name in ("w1", "w2", "w3")])
        )
    return families


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("block_namespace", "unknown"),
        record.get("logical_layer", -1),
        record.get("expert", -1),
        0 if record["record_kind"] == "projection" else 1,
        record.get("projection", ""),
        record.get("bits", -1),
        record.get("module", ""),
    )


def _bind_record(record: dict[str, Any]) -> dict[str, Any]:
    clean = _finite_json_value(record)
    digest = hashlib.sha256(_canonical_json_bytes(clean)).hexdigest()
    return {**clean, "record_sha256": digest}


def _journal_file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _load_journal_index(path: Path) -> set[str]:
    digests: set[str] = set()
    try:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, 1):
                if not line.endswith(b"\n"):
                    raise ValueError(
                        "EXL3 error journal ends with a partial record"
                    )
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"EXL3 error journal record {line_number} is invalid"
                    ) from error
                if (
                    not isinstance(record, dict)
                    or record.get("record_kind") != "projection"
                ):
                    raise ValueError(
                        f"EXL3 error journal record {line_number} is not a projection"
                    )
                digest = record.get("record_sha256")
                unbound = {
                    key: value
                    for key, value in record.items()
                    if key != "record_sha256"
                }
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or hashlib.sha256(_canonical_json_bytes(unbound)).hexdigest()
                    != digest
                ):
                    raise ValueError(
                        f"EXL3 error journal record {line_number} failed its digest"
                    )
                digests.add(digest)
    except OSError as error:
        raise ValueError(f"cannot read EXL3 error journal: {path}") from error
    return digests


def append_exl3_error_journal(
    journal_path: str | os.PathLike[str],
    projection_record: dict[str, Any],
) -> str:
    """Append and fsync one bound projection before its tensors may commit."""

    if projection_record.get("record_kind") != "projection":
        raise ValueError("EXL3 error journal accepts projection records only")
    bound = _bind_record(projection_record)
    payload = _canonical_json_bytes(bound) + b"\n"
    path = Path(journal_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JOURNAL_INDEX_LOCK:
        existed = path.exists()
        identity = _journal_file_identity(path) if existed else None
        cached = _JOURNAL_INDEX.get(path)
        if cached is None or cached[0] != identity:
            digests = _load_journal_index(path) if existed else set()
            _JOURNAL_INDEX[path] = (identity, digests)
        else:
            digests = cached[1]
        if bound["record_sha256"] in digests:
            return bound["record_sha256"]

        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(
                    "short EXL3 error-journal write: "
                    f"expected {len(payload)}, wrote {written}"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not existed:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        digests.add(bound["record_sha256"])
        _JOURNAL_INDEX[path] = (_journal_file_identity(path), digests)
    return bound["record_sha256"]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_exl3_error_ledger(
    save_dir: str | os.PathLike[str],
    projection_records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Atomically persist projections, family joins, and a digest manifest."""

    projections = [
        compact_projection_record(record)
        for record in projection_records
        if record.get("record_kind") == "projection"
        and record.get("schema") == LEDGER_SCHEMA
        and record.get("schema_version") == LEDGER_SCHEMA_VERSION
    ]
    if not projections:
        return None

    families = derive_family_records(projections)
    records = projections + families
    records.sort(key=_record_sort_key)

    save_path = Path(save_dir)
    ledger_path = save_path / LEDGER_FILENAME
    manifest_path = save_path / LEDGER_MANIFEST_FILENAME
    save_path.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ledger_path.name}.", dir=save_path
    )
    ledger_digest = hashlib.sha256()
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                payload = _canonical_json_bytes(_bind_record(record)) + b"\n"
                handle.write(payload)
                ledger_digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, ledger_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    ledger_sha256 = ledger_digest.hexdigest()

    manifest = {
        "schema": LEDGER_SCHEMA,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ledger": LEDGER_FILENAME,
        "ledger_sha256": ledger_sha256,
        "projection_records": len(projections),
        "complete_family_records": len(families),
        "total_records": len(records),
    }
    _atomic_write(manifest_path, _canonical_json_bytes(manifest) + b"\n")
    return manifest


__all__ = [
    "JOURNAL_ENV",
    "LEDGER_FILENAME",
    "LEDGER_MANIFEST_FILENAME",
    "LEDGER_SCHEMA",
    "LEDGER_SCHEMA_VERSION",
    "PROJECTION_PROVENANCE_COMPACTION_CONTRACT",
    "ROUTE_EVIDENCE_SCHEMA",
    "ROUTE_EVIDENCE_SCHEMA_VERSION",
    "ZERO_ROUTE_RECOVERY_CAPTURE_METHOD",
    "ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX",
    "ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN",
    "ZERO_ROUTE_RECOVERY_IDENTITY_POLICY",
    "ZERO_ROUTE_RECOVERY_MODE_IDENTITY",
    "ZERO_ROUTE_RECOVERY_MODE_MIXED",
    "ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR",
    "ZERO_ROUTE_RECOVERY_SELECTION_CAP",
    "ZERO_ROUTE_RECOVERY_SELECTION_POLICY",
    "ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT",
    "ZERO_ROUTE_RECOVERY_AUTHORIZATION_KINDS",
    "ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA",
    "ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA_VERSION",
    "ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE",
    "ZERO_ROUTE_RECOVERY_SCHEMA",
    "ZERO_ROUTE_RECOVERY_SCHEMA_VERSION",
    "ZERO_ROUTE_RECOVERY_TRIGGER",
    "append_exl3_error_journal",
    "build_projection_record",
    "compact_projection_provenance",
    "compact_projection_record",
    "derive_family_records",
    "route_evidence_required",
    "routed_expert_identity",
    "validate_route_evidence",
    "validate_zero_route_recovery",
    "validate_zero_route_recovery_authorization",
    "write_exl3_error_ledger",
    "zero_route_recovery_enabled",
    "zero_route_recovery_recipe",
]
