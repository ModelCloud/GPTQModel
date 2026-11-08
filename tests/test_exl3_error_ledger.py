# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from gptqmodel.utils import exl3_error_ledger as ledger_module
from gptqmodel.utils.exl3_error_ledger import (
    LEDGER_FILENAME,
    LEDGER_MANIFEST_FILENAME,
    PROJECTION_PROVENANCE_COMPACTION_CONTRACT,
    ROUTE_EVIDENCE_SCHEMA,
    ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
    ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
    ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
    ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    ZERO_ROUTE_RECOVERY_MODE_IDENTITY,
    ZERO_ROUTE_RECOVERY_MODE_MIXED,
    ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR,
    ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
    ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
    ZERO_ROUTE_RECOVERY_SCHEMA,
    ZERO_ROUTE_RECOVERY_SELECTION_CAP,
    ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
    ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
    ZERO_ROUTE_RECOVERY_TRIGGER,
    append_exl3_error_journal,
    build_projection_record,
    compact_projection_provenance,
    compact_projection_record,
    derive_family_records,
    routed_expert_identity,
    write_exl3_error_ledger,
    zero_route_recovery_recipe,
)
from gptqmodel.looper.exllamav3_processor import EXL3Processor


def _metrics(scale: float) -> dict:
    return {
        "schema": "gptqmodel.exl3-trellis-error",
        "schema_version": 1,
        "quantizer_path": "hessian_ldlq",
        "reported_metric_kind": "hessian_weighted_relative_error",
        "reported_metric_value": 0.1 * scale,
        "hessian_weighted_error_numerator": 2.0 * scale,
        "hessian_weighted_reference_denominator": 20.0 * scale,
        "hessian_weighted_relative_error": 0.1,
        "hessian_metric_status": "ok",
        "selected_global_scale": 0.9,
        "scale_search_mse": 0.01,
        "apply_out_scales": True,
        "reconstruction": {
            "domain": "regularized_exl3_search_space",
            "shape": [16, 16],
            "element_count": 256,
            "error_sum_sq": 4.0 * scale,
            "reference_sum_sq": 40.0 * scale,
            "mse": 4.0 * scale / 256,
            "nmse": 0.1,
            "relative_frobenius": 0.1**0.5,
            "mean_abs_error": 0.01,
            "max_abs_error": 0.2 * scale,
            "reference_finite": True,
            "error_finite": True,
            "tile_shape": [16, 16],
            "tile_count": 1,
            "tile_sse_sum": 4.0 * scale,
            "tile_sse_max": 4.0 * scale,
            "tile_sse_percentiles": {
                "p50": 4.0 * scale,
                "p90": 4.0 * scale,
                "p99": 4.0 * scale,
                "p99_9": 4.0 * scale,
            },
            "worst_tiles": [{"row": 0, "column": 0, "sse": 4.0 * scale}],
        },
    }


def _route_evidence(scale: float = 1.0) -> dict:
    route_count = 1024
    selected_count = 8192
    gate_sum = 100.0 * scale
    gate_sq = 20.0 * scale
    total_gate_sum = 1000.0 * scale
    total_gate_sq = 200.0 * scale
    return {
        "schema": ROUTE_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "block_namespace": "base",
        "logical_layer": 7,
        "expert": 31,
        "router_calls": 8,
        "router_token_count": 1024,
        "router_selected_route_count": selected_count,
        "router_top_k": 8,
        "expert_route_count": route_count,
        "expert_gate_weight_sum": gate_sum,
        "expert_gate_squared_mass": gate_sq,
        "total_gate_weight_sum": total_gate_sum,
        "total_gate_squared_mass": total_gate_sq,
        "expert_route_fraction": route_count / selected_count,
        "expert_gate_weight_mass_fraction": gate_sum / total_gate_sum,
        "expert_gate_squared_mass_fraction": gate_sq / total_gate_sq,
        "expert_gate_weight_mean": gate_sum / route_count,
        "expert_gate_weight_rms": (gate_sq / route_count) ** 0.5,
        "router_weight_dtypes": ["torch.float32"],
        "mask_modes": ["all-valid"],
    }


def _zero_route_evidence() -> dict:
    evidence = _route_evidence()
    evidence.update(
        {
            "expert_route_count": 0,
            "expert_gate_weight_sum": 0.0,
            "expert_gate_squared_mass": 0.0,
            "expert_route_fraction": 0.0,
            "expert_gate_weight_mass_fraction": 0.0,
            "expert_gate_squared_mass_fraction": 0.0,
            "expert_gate_weight_mean": 0.0,
            "expert_gate_weight_rms": 0.0,
        }
    )
    return evidence


def _zero_route_recovery(
    family_join: dict,
    sample_count: int = 1024,
) -> dict:
    family_digest = hashlib.sha256(
        json.dumps(
            family_join,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema": ZERO_ROUTE_RECOVERY_SCHEMA,
        "schema_version": 1,
        "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
        "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
        "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
        "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
        "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
        "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
        "selection_cap": ZERO_ROUTE_RECOVERY_SELECTION_CAP,
        "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
        "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
        "block_namespace": "base",
        "logical_layer": 7,
        "expert": 31,
        "natural_sample_count": 0,
        "router_augmented_sample_count": 0,
        "identity_calibration_count": sample_count,
        "total_sample_count": sample_count,
        "forced_pass_count": 1,
        "recovery_mode": ZERO_ROUTE_RECOVERY_MODE_IDENTITY,
        "candidate_rows_observed": 0,
        "candidate_rows_selected": 0,
        "candidate_rank_histogram": {
            str(rank): 0
            for rank in range(
                ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX + 1,
            )
        },
        "candidate_score_gap": None,
        "authorization": {
            "schema": ZERO_ROUTE_RECOVERY_AUTHORIZATION_SCHEMA,
            "schema_version": 1,
            "kind": "content-bound-execution-upgrade",
            "recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
            "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
            "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
            "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
            "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
            "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
            "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
            "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
            "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
            "family_join_sha256": family_digest,
            "authorization_sha256": "a" * 64,
        },
    }


def _glm_recovery_recipe() -> dict:
    return {
        "trigger": "natural-route-count-below-1024",
        "sample_source": "same-fixed-calibration-selection",
        "capture_method": (
            "direct-expert-router-ranks-9-16-then-identity-residual"
        ),
        "selection_policy": "rank-ascending-then-fixed-replay-order-v1",
        "candidate_rank_min": 9,
        "candidate_rank_max": 16,
        "selection_cap": 1024,
        "target_sample_count": 1024,
        "identity_calibration_policy": (
            "normalized-2i-residual-to-effective-count-1024-v2"
        ),
    }


def _record(
    projection: str,
    scale: float = 1.0,
    provenance=None,
    route_evidence=None,
    zero_route_recovery=None,
) -> dict:
    return build_projection_record(
        module_full_name=f"model.layers.7.mlp.experts.31.{projection}",
        layer_index=7,
        bits=2,
        codebook="mcg",
        sample_count=1024,
        duration_seconds=1.25 * scale,
        encoded_bytes=128,
        device_names=["cuda:0"],
        quantizer_metrics=_metrics(scale),
        provenance=provenance or {"source_revision": "abc", "hessian_sha256": "def"},
        route_evidence=route_evidence,
        zero_route_recovery=zero_route_recovery,
    )


def test_routed_expert_identity_covers_base_and_mtp_namespaces():
    assert routed_expert_identity("model.layers.7.mlp.experts.31.gate_proj") == {
        "block_namespace": "base",
        "logical_layer": 7,
        "expert": 31,
        "projection": "w1",
    }
    assert routed_expert_identity("mtp.2.mlp.experts.255.down_proj") == {
        "block_namespace": "mtp",
        "logical_layer": 2,
        "expert": 255,
        "projection": "w2",
    }
    assert routed_expert_identity("model.layers.7.mlp.shared_experts.up_proj") is None


def _seeded_provenance() -> dict:
    family_join = {"source_revision": "abc", "corpus_sha256": "def"}
    files = [
        {"path": "01/23/checkpoint.json", "sha256": "1" * 64},
        {"path": "45/67/checkpoint.json", "sha256": "2" * 64},
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "family_join": family_join,
        "run": {
            "projection_checkpoint": {"root": "/current", "contract": "v1"},
            "projection_checkpoint_seed": {
                "contract": "seed-v1",
                "root": "/seed",
                "checkpoint_count": 2,
                "total_bytes": 1234,
                "inventory_sha256": inventory_sha256,
                "files": files,
                "family_join": family_join,
            },
        },
    }


def test_projection_provenance_compacts_repeated_seed_inventory_by_digest():
    provenance = _seeded_provenance()
    compact = compact_projection_provenance(provenance)
    seed = compact["run"]["projection_checkpoint_seed"]

    assert "files" not in seed
    assert "family_join" not in seed
    assert seed["inventory_sha256"] == provenance["run"][
        "projection_checkpoint_seed"
    ]["inventory_sha256"]
    assert seed["files_sha256"] == seed["inventory_sha256"]
    assert seed["family_join_sha256"] == hashlib.sha256(
        json.dumps(
            provenance["family_join"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert (
        seed["provenance_compaction_contract"]
        == PROJECTION_PROVENANCE_COMPACTION_CONTRACT
    )
    assert provenance["run"]["projection_checkpoint_seed"]["files"]
    assert compact_projection_provenance(compact) == compact


def test_projection_provenance_rejects_seed_inventory_digest_drift():
    provenance = _seeded_provenance()
    provenance["run"]["projection_checkpoint_seed"]["inventory_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="inventory digest differs"):
        compact_projection_provenance(provenance)


def test_projection_record_keeps_legacy_checkpoint_compatibility():
    provenance = _seeded_provenance()
    compact = _record("gate_proj", provenance=provenance)
    legacy = build_projection_record(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        bits=2,
        codebook="mcg",
        sample_count=1024,
        duration_seconds=1.25,
        encoded_bytes=128,
        device_names=["cuda:0"],
        quantizer_metrics=_metrics(1.0),
        provenance=provenance,
        compact_provenance=False,
    )

    assert "files" not in compact["provenance"]["run"][
        "projection_checkpoint_seed"
    ]
    assert legacy["provenance"]["run"]["projection_checkpoint_seed"]["files"]
    assert compact_projection_record(legacy) == compact


def test_zero_route_recovery_recipe_keeps_deepseek_default_and_binds_glm_window():
    default = zero_route_recovery_recipe(None)
    assert default["candidate_rank_min"] == 7
    assert default["candidate_rank_max"] == 12
    assert default["capture_method"] == ZERO_ROUTE_RECOVERY_CAPTURE_METHOD

    glm_recipe = _glm_recovery_recipe()
    assert zero_route_recovery_recipe(
        {"zero_route_recovery_recipe": glm_recipe}
    ) == glm_recipe


@pytest.mark.parametrize(
    "mutation",
    [
        {"candidate_rank_min": 0},
        {"candidate_rank_min": 17, "candidate_rank_max": 16},
        {"selection_cap": 1025},
        {"capture_method": ""},
    ],
)
def test_zero_route_recovery_recipe_rejects_invalid_model_binding(mutation):
    recipe = {**_glm_recovery_recipe(), **mutation}
    with pytest.raises(ValueError, match="recovery recipe is invalid"):
        zero_route_recovery_recipe({"zero_route_recovery_recipe": recipe})


def test_family_join_aggregates_raw_terms_only_for_exact_provenance():
    records = [
        _record("gate_proj", 1.0),
        _record("down_proj", 2.0),
        _record("up_proj", 3.0),
    ]
    families = derive_family_records(records)
    assert len(families) == 1
    family = families[0]
    assert family["projections"] == ["w1", "w2", "w3"]
    assert family["aggregate_metrics"]["error_sum_sq"] == pytest.approx(24.0)
    assert family["aggregate_metrics"]["reference_sum_sq"] == pytest.approx(240.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_error_numerator"
    ] == pytest.approx(12.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_reference_denominator"
    ] == pytest.approx(120.0)
    assert family["aggregate_metrics"][
        "hessian_weighted_relative_error"
    ] == pytest.approx(0.1)

    records[-1] = _record(
        "up_proj",
        3.0,
        provenance={"source_revision": "different", "hessian_sha256": "def"},
    )
    assert derive_family_records(records) == []

    common = {"source_revision": "abc", "corpus_sha256": "def"}
    records = [
        _record(
            projection,
            float(index),
            provenance={
                "family_join": common,
                "hessian_sha256": str(index) * 64,
            },
        )
        for index, projection in enumerate(
            ("gate_proj", "down_proj", "up_proj"), start=1
        )
    ]
    families = derive_family_records(records)
    assert len(families) == 1
    assert families[0]["provenance"]["family_join"] == common
    assert set(families[0]["provenance"]["projections"]) == {"w1", "w2", "w3"}


def test_family_join_records_projection_mixed_k2_k3_tiers():
    records = [
        _record("gate_proj"),
        _record("down_proj"),
        _record("up_proj"),
    ]
    records[1]["bits"] = 3

    family = derive_family_records(records)[0]

    assert family["bits"] == 2
    assert family["mixed_bits"] is True
    assert family["projection_bits"] == {"w1": 2, "w2": 3, "w3": 2}


def test_natural_route_evidence_is_required_and_shared_by_one_expert_family():
    provenance = {
        "family_join": {
            "source_revision": "abc",
            "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
        }
    }
    with pytest.raises(ValueError, match="natural-route evidence is required"):
        _record("gate_proj", provenance=provenance)

    evidence = _route_evidence()
    records = [
        _record(projection, provenance=provenance, route_evidence=evidence)
        for projection in ("gate_proj", "down_proj", "up_proj")
    ]
    family = derive_family_records(records)[0]
    assert family["route_evidence"] == evidence
    assert family["route_evidence"]["expert_gate_squared_mass"] == 20.0

    records[-1] = _record(
        "up_proj",
        provenance=provenance,
        route_evidence=_route_evidence(scale=2.0),
    )
    with pytest.raises(ValueError, match="inconsistent route evidence"):
        derive_family_records(records)


def test_zero_route_recovery_is_explicit_and_cannot_relabel_positive_routes():
    provenance = {
        "family_join": {
            "source_revision": "abc",
            "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
        }
    }
    recovery = _zero_route_recovery(provenance["family_join"])
    records = [
        _record(
            projection,
            provenance=provenance,
            route_evidence=_zero_route_evidence(),
            zero_route_recovery=recovery,
        )
        for projection in ("gate_proj", "down_proj", "up_proj")
    ]
    family = derive_family_records(records)[0]
    assert family["route_evidence"]["expert_route_count"] == 0
    assert family["zero_route_recovery"] == recovery
    assert family["sample_counts"] == [1024, 1024, 1024]

    with pytest.raises(ValueError, match="natural-route evidence has an invalid"):
        _record(
            "gate_proj",
            provenance=provenance,
            route_evidence=_zero_route_evidence(),
        )
    with pytest.raises(ValueError, match="natural-route evidence has an invalid"):
        _record(
            "gate_proj",
            provenance=provenance,
            route_evidence=_route_evidence(),
            zero_route_recovery=recovery,
        )
    malformed = {**recovery, "natural_sample_count": 1}
    with pytest.raises(ValueError, match="identity-Hessian recovery evidence"):
        _record(
            "gate_proj",
            provenance=provenance,
            route_evidence=_zero_route_evidence(),
            zero_route_recovery=malformed,
        )


def test_mixed_recovery_binds_empirical_rows_and_identity_residual() -> None:
    family_join = {
        "source_revision": "abc",
        "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
        "zero_route_recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
    }
    provenance = {"family_join": family_join}
    natural_count = 300
    router_count = 381
    identity_count = 1024 - natural_count - router_count
    evidence = _route_evidence()
    evidence.update(
        {
            "expert_route_count": natural_count,
            "expert_route_fraction": natural_count
            / evidence["router_selected_route_count"],
            "expert_gate_weight_mean": evidence["expert_gate_weight_sum"]
            / natural_count,
            "expert_gate_weight_rms": (
                evidence["expert_gate_squared_mass"] / natural_count
            )
            ** 0.5,
        }
    )
    recovery = _zero_route_recovery(family_join)
    recovery.update(
        {
            "natural_sample_count": natural_count,
            "router_augmented_sample_count": router_count,
            "identity_calibration_count": identity_count,
            "recovery_mode": ZERO_ROUTE_RECOVERY_MODE_MIXED,
            "candidate_rows_observed": router_count,
            "candidate_rows_selected": router_count,
            "candidate_rank_histogram": {
                "7": router_count,
                "8": 0,
                "9": 0,
                "10": 0,
                "11": 0,
                "12": 0,
            },
            "candidate_score_gap": {
                "min": 0.01,
                "mean": 0.02,
                "max": 0.03,
            },
        }
    )
    record = _record(
        "gate_proj",
        provenance=provenance,
        route_evidence=evidence,
        zero_route_recovery=recovery,
    )
    assert record["zero_route_recovery"]["identity_calibration_count"] == 343
    assert record["zero_route_recovery"]["candidate_rows_selected"] == 381

    malformed = {**recovery, "identity_calibration_count": 342}
    with pytest.raises(ValueError, match="mixed recovery evidence"):
        _record(
            "gate_proj",
            provenance=provenance,
            route_evidence=evidence,
            zero_route_recovery=malformed,
        )


@pytest.mark.parametrize(
    ("block_namespace", "logical_layer", "module_prefix"),
    [
        ("base", 7, "model.layers.7"),
        ("mtp", 1, "mtp.1"),
    ],
)
def test_exl3_census_tops_up_undercovered_learned_router_namespaces(
    block_namespace,
    logical_layer,
    module_prefix,
):
    processor = object.__new__(EXL3Processor)
    provenance = {
        "family_join": {
            "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
        }
    }
    processor.qcfg = SimpleNamespace(meta={"ds4rt_error_ledger": provenance})
    layer_module = torch.nn.Module()
    layer_module.mlp = torch.nn.Module()
    layer_module.mlp.gate = torch.nn.Module()
    layer_module.mlp.gate.e_score_correction_bias = torch.zeros(256)
    subset = {}
    processor.tasks = {}
    natural_count = 100
    mtp_evidence = _route_evidence()
    mtp_evidence.update(
        {
            "block_namespace": block_namespace,
            "logical_layer": logical_layer,
            "expert_route_count": natural_count,
            "expert_route_fraction": natural_count
            / mtp_evidence["router_selected_route_count"],
            "expert_gate_weight_mean": mtp_evidence["expert_gate_weight_sum"]
            / natural_count,
            "expert_gate_weight_rms": (
                mtp_evidence["expert_gate_squared_mass"] / natural_count
            )
            ** 0.5,
        }
    )
    for projection in ("gate_proj", "up_proj"):
        task_name = f"mlp.experts.31.{projection}"
        subset[task_name] = SimpleNamespace(
            full_name=f"{module_prefix}.mlp.experts.31.{projection}"
        )
        processor.tasks[task_name] = {
            "capture": SimpleNamespace(nsamples=natural_count),
            "route_evidence": dict(mtp_evidence),
        }

    assert processor.plan_subset_zero_route_recovery(
        subset=subset,
        layer_module=layer_module,
    ) == ()

    provenance["family_join"]["zero_route_recovery_contract"] = (
        ZERO_ROUTE_RECOVERY_SCHEMA
    )
    targets = processor.plan_subset_zero_route_recovery(
        subset=subset,
        layer_module=layer_module,
    )
    assert targets == tuple(sorted(subset))
    for task in processor.tasks.values():
        task["capture"].nsamples = ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT
        task["zero_route_recovery_capture"] = {
            "recovery_mode": ZERO_ROUTE_RECOVERY_MODE_ROUTER_NEAR,
            "router_augmented_sample_count": 924,
            "identity_calibration_count": 0,
            "candidate_rows_observed": 2000,
            "candidate_rows_selected": 924,
            "candidate_rank_histogram": {
                "7": 924,
                "8": 0,
                "9": 0,
                "10": 0,
                "11": 0,
                "12": 0,
            },
            "candidate_score_gap": {"min": 0.01, "mean": 0.02, "max": 0.03},
        }
    processor.finish_subset_zero_route_recovery(
        subset=subset,
        task_names=targets,
    )
    processor.validate_subset_capture_readiness(
        subset=subset,
        layer_module=layer_module,
    )
    assert {
        task["zero_route_recovery"]["router_augmented_sample_count"]
        for task in processor.tasks.values()
    } == {924}

    processor.tasks[targets[0]]["route_evidence"]["expert_route_count"] = 1024
    with pytest.raises(RuntimeError, match="relabeled as recovery"):
        processor.validate_subset_capture_readiness(
            subset=subset,
            layer_module=layer_module,
        )


def test_exl3_census_excludes_deterministic_hash_router_from_topup():
    processor = object.__new__(EXL3Processor)
    processor.qcfg = SimpleNamespace(
        meta={
            "ds4rt_error_ledger": {
                "family_join": {
                    "route_evidence_contract": ROUTE_EVIDENCE_SCHEMA,
                    "zero_route_recovery_contract": ZERO_ROUTE_RECOVERY_SCHEMA,
                }
            }
        }
    )
    layer_module = torch.nn.Module()
    layer_module.mlp = torch.nn.Module()
    layer_module.mlp.gate = torch.nn.Module()
    layer_module.mlp.gate.tid2eid = torch.arange(256)
    task_name = "mlp.experts.31.gate_proj"
    subset = {
        task_name: SimpleNamespace(
            full_name="model.layers.7.mlp.experts.31.gate_proj"
        )
    }
    evidence = _route_evidence()
    route_count = 100
    gate_sum = 10.0
    gate_sq = 2.0
    evidence.update(
        {
            "block_namespace": "base",
            "logical_layer": 7,
            "expert_route_count": route_count,
            "expert_gate_weight_sum": gate_sum,
            "expert_gate_squared_mass": gate_sq,
            "expert_route_fraction": route_count
            / evidence["router_selected_route_count"],
            "expert_gate_weight_mass_fraction": gate_sum
            / evidence["total_gate_weight_sum"],
            "expert_gate_squared_mass_fraction": gate_sq
            / evidence["total_gate_squared_mass"],
            "expert_gate_weight_mean": gate_sum / route_count,
            "expert_gate_weight_rms": (gate_sq / route_count) ** 0.5,
        }
    )
    processor.tasks = {
        task_name: {
            "capture": SimpleNamespace(nsamples=100),
            "route_evidence": evidence,
        }
    }

    assert processor.plan_subset_zero_route_recovery(
        subset=subset,
        layer_module=layer_module,
    ) == ()
    processor.validate_subset_capture_readiness(
        subset=subset,
        layer_module=layer_module,
    )


def test_ledger_is_canonical_content_bound_and_contains_family_record(tmp_path):
    records = [
        _record("up_proj", 3.0),
        _record("gate_proj", 1.0),
        _record("down_proj", 2.0),
    ]
    manifest = write_exl3_error_ledger(tmp_path, records)
    assert manifest is not None
    assert manifest["projection_records"] == 3
    assert manifest["complete_family_records"] == 1
    assert manifest["total_records"] == 4

    payload = (tmp_path / LEDGER_FILENAME).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["ledger_sha256"]
    assert json.loads((tmp_path / LEDGER_MANIFEST_FILENAME).read_text()) == manifest
    assert (tmp_path / LEDGER_FILENAME).stat().st_mode & 0o777 == 0o644
    assert (tmp_path / LEDGER_MANIFEST_FILENAME).stat().st_mode & 0o777 == 0o644

    rows = [json.loads(line) for line in payload.splitlines()]
    assert [row.get("projection", "family") for row in rows] == [
        "w1",
        "w2",
        "w3",
        "family",
    ]
    for row in rows:
        unbound = {key: value for key, value in row.items() if key != "record_sha256"}
        canonical = json.dumps(
            unbound,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == row["record_sha256"]


def test_ledger_normalizes_legacy_seed_provenance_before_publication(tmp_path):
    provenance = _seeded_provenance()
    records = [
        build_projection_record(
            module_full_name=f"model.layers.7.mlp.experts.31.{projection}",
            layer_index=7,
            bits=2,
            codebook="mcg",
            sample_count=1024,
            duration_seconds=float(index),
            encoded_bytes=128,
            device_names=["cuda:0"],
            quantizer_metrics=_metrics(float(index)),
            provenance=provenance,
            compact_provenance=False,
        )
        for index, projection in enumerate(
            ("gate_proj", "down_proj", "up_proj"), start=1
        )
    ]

    write_exl3_error_ledger(tmp_path, records)
    rows = [
        json.loads(line)
        for line in (tmp_path / LEDGER_FILENAME).read_bytes().splitlines()
    ]
    for row in rows:
        if row["record_kind"] != "projection":
            continue
        seed = row["provenance"]["run"]["projection_checkpoint_seed"]
        assert "files" not in seed
        assert "family_join" not in seed
        assert seed["files_sha256"] == seed["inventory_sha256"]


def test_projection_journal_fsyncs_individually_bound_records(tmp_path):
    journal = tmp_path / "in-progress.jsonl"
    expected = [_record("gate_proj", 1.0), _record("down_proj", 2.0)]
    digests = [append_exl3_error_journal(journal, record) for record in expected]

    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert [row["record_sha256"] for row in rows] == digests
    assert [row["projection"] for row in rows] == ["w1", "w2"]


def test_projection_journal_is_idempotent_across_process_index_rebuild(
    tmp_path, monkeypatch
):
    journal = tmp_path / "in-progress.jsonl"
    record = _record("gate_proj", 1.0)

    first = append_exl3_error_journal(journal, record)
    second = append_exl3_error_journal(journal, record)
    ledger_module._JOURNAL_INDEX.clear()
    original_load = ledger_module._load_journal_index
    load_calls = []

    def counted_load(path):
        load_calls.append(path)
        return original_load(path)

    monkeypatch.setattr(ledger_module, "_load_journal_index", counted_load)
    third = append_exl3_error_journal(journal, record)
    fourth = append_exl3_error_journal(journal, record)

    assert first == second == third == fourth
    assert load_calls == [journal.resolve()]
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert len(rows) == 1
    assert rows[0]["record_sha256"] == first


def test_projection_journal_rejects_partial_or_digest_corrupt_history(tmp_path):
    journal = tmp_path / "in-progress.jsonl"
    journal.write_bytes(b'{"partial":true}')
    ledger_module._JOURNAL_INDEX.clear()
    with pytest.raises(ValueError, match="partial record"):
        append_exl3_error_journal(journal, _record("gate_proj", 1.0))

    journal.write_text(
        json.dumps({**_record("gate_proj", 1.0), "record_sha256": "0" * 64})
        + "\n",
        encoding="utf-8",
    )
    ledger_module._JOURNAL_INDEX.clear()
    with pytest.raises(ValueError, match="failed its digest"):
        append_exl3_error_journal(journal, _record("down_proj", 2.0))


def test_ledger_rejects_non_finite_metrics():
    metrics = _metrics(1.0)
    metrics["reported_metric_value"] = float("nan")
    with pytest.raises(ValueError, match="non-finite EXL3 ledger value"):
        build_projection_record(
            module_full_name="model.layers.0.mlp.experts.0.gate_proj",
            layer_index=0,
            bits=2,
            codebook="mcg",
            sample_count=1,
            duration_seconds=1.0,
            encoded_bytes=1,
            device_names=["cuda:0"],
            quantizer_metrics=metrics,
            provenance=None,
        )
