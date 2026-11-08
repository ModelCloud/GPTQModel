# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
import json

import pytest

from gptqmodel.utils.exl3_inline_mixed import (
    INLINE_MIXED_LEGACY_K2_SCORE,
    INLINE_MIXED_META_KEY,
    INLINE_MIXED_SCHEMA,
    INLINE_MIXED_SCHEMA_VERSION,
    INLINE_MIXED_SCORE,
    InlineMixedPolicy,
    InlineMixedTierPlanStore,
    build_layer_tier_plan,
    inline_mixed_policy,
)


def _policy(tmp_path, *, extra=(1, 10)) -> InlineMixedPolicy:
    return InlineMixedPolicy(
        namespace="base",
        base_bits=2,
        upgrade_bits=3,
        extra_bits=Fraction(*extra),
        projection_ratio=(3, 5, 8),
        tier_plan_root=tmp_path,
    )


def _record(expert: int, projection: str, error: float, mass: float = 0.25):
    projection_name = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[
        projection
    ]
    return {
        "record_kind": "projection",
        "module": f"model.layers.0.mlp.experts.{expert}.{projection_name}",
        "block_namespace": "base",
        "logical_layer": 0,
        "expert": expert,
        "projection": projection,
        "bits": 2,
        "quantizer_metrics": {"hessian_weighted_relative_error": error},
        "route_evidence": {"expert_gate_squared_mass_fraction": mass},
    }


def test_policy_parses_exact_fraction_and_rejects_float_target(tmp_path):
    raw = {
        INLINE_MIXED_META_KEY: {
            "schema": INLINE_MIXED_SCHEMA,
            "schema_version": INLINE_MIXED_SCHEMA_VERSION,
            "namespace": "base",
            "base_bits": 2,
            "upgrade_bits": 3,
            "extra_bits": {"numerator": 1, "denominator": 10},
            "projection_ratio": {"w1": 3, "w3": 5, "w2": 8},
            "score_kind": INLINE_MIXED_SCORE,
            "tier_plan_root": str(tmp_path),
        }
    }
    parsed = inline_mixed_policy(raw)
    assert parsed is not None
    assert parsed.target_bpw == Fraction(21, 10)
    broken = json.loads(json.dumps(raw))
    broken[INLINE_MIXED_META_KEY]["extra_bits"] = 0.1
    with pytest.raises(ValueError, match="invalid contract"):
        inline_mixed_policy(broken)

    legacy = json.loads(json.dumps(raw))
    legacy[INLINE_MIXED_META_KEY]["score_kind"] = INLINE_MIXED_LEGACY_K2_SCORE
    parsed_legacy = inline_mixed_policy(legacy)
    assert parsed_legacy is not None
    assert parsed_legacy.score_kind == INLINE_MIXED_LEGACY_K2_SCORE


def test_21_bpw_358_quotas_are_exact_and_causally_distributed(tmp_path):
    policy = _policy(tmp_path)
    totals = policy.namespace_quotas(layer_count=43, experts_per_layer=256)
    assert totals == {"w1": 619, "w3": 1032, "w2": 1651}
    assert sum(totals.values()) == 3302
    accumulated = {projection: 0 for projection in totals}
    for layer in range(43):
        quota = policy.layer_quotas(
            layer_index=layer,
            layer_count=43,
            experts_per_layer=256,
        )
        for projection, value in quota.items():
            accumulated[projection] += value
    assert accumulated == totals


def test_glm_k325_358_quotas_use_only_target_layer_range(tmp_path):
    policy = InlineMixedPolicy(
        namespace="base",
        base_bits=3,
        upgrade_bits=4,
        extra_bits=Fraction(1, 4),
        projection_ratio=(3, 5, 8),
        tier_plan_root=tmp_path,
        logical_layer_start=3,
        logical_layer_count=42,
    )
    totals = policy.namespace_quotas(layer_count=42, experts_per_layer=288)
    assert totals == {"w1": 1701, "w3": 2835, "w2": 4536}
    assert sum(totals.values()) == 9072

    accumulated = {projection: 0 for projection in totals}
    for layer in range(3, 45):
        quota = policy.layer_quotas(
            layer_index=layer,
            layer_count=46,
            experts_per_layer=288,
        )
        assert quota["w2"] == 108
        assert quota["w1"] in {40, 41}
        assert quota["w3"] in {67, 68}
        for projection, value in quota.items():
            accumulated[projection] += value
    assert accumulated == totals

    with pytest.raises(ValueError, match="layer index"):
        policy.layer_quotas(
            layer_index=45,
            layer_count=46,
            experts_per_layer=288,
        )


def test_layer_plan_ranks_only_within_projection_class(tmp_path):
    policy = _policy(tmp_path, extra=(1, 6))
    records = [
        _record(expert, projection, error=(expert + 1) * multiplier)
        for expert in range(16)
        for projection, multiplier in (("w1", 1.0), ("w3", 2.0), ("w2", 3.0))
    ]
    plan = build_layer_tier_plan(
        policy=policy,
        layer_index=0,
        layer_count=1,
        candidate_records=records,
    )
    assert plan["quotas"] == {"w1": 2, "w3": 2, "w2": 4}
    selected = {(item["projection"], item["expert"]) for item in plan["selected"]}
    assert selected == {
        ("w1", 14),
        ("w1", 15),
        ("w3", 14),
        ("w3", 15),
        ("w2", 12),
        ("w2", 13),
        ("w2", 14),
        ("w2", 15),
    }


def test_tier_plan_store_is_idempotent_and_rejects_drift(tmp_path):
    policy = _policy(tmp_path, extra=(1, 6))
    records = [
        _record(expert, projection, error=float(expert + 1))
        for expert in range(16)
        for projection in ("w1", "w3", "w2")
    ]
    plan = build_layer_tier_plan(
        policy=policy,
        layer_index=0,
        layer_count=1,
        candidate_records=records,
    )
    store = InlineMixedTierPlanStore(policy)
    store.commit(plan)
    store.commit(plan)
    assert store.path(0).stat().st_mode & 0o777 == 0o644
    changed = json.loads(json.dumps(plan))
    changed["selected"][0]["expert"] = 0
    with pytest.raises(ValueError, match="invalid"):
        store.commit(changed)


def test_tier_plan_store_load_authenticates_complete_contract(tmp_path):
    policy = _policy(tmp_path, extra=(1, 6))
    records = [
        _record(expert, projection, error=float(expert + 1))
        for expert in range(16)
        for projection in ("w1", "w3", "w2")
    ]
    plan = build_layer_tier_plan(
        policy=policy,
        layer_index=0,
        layer_count=1,
        candidate_records=records,
    )
    store = InlineMixedTierPlanStore(policy)
    assert store.load(
        layer_index=0,
        layer_count=1,
        experts_per_layer=16,
    ) is None
    store.commit(plan)
    assert store.load(
        layer_index=0,
        layer_count=1,
        experts_per_layer=16,
    ) == plan

    path = store.path(0)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["selected"][0]["score"] += 1.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="failed validation"):
        store.load(
            layer_index=0,
            layer_count=1,
            experts_per_layer=16,
        )
