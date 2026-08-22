# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from torch import nn

import gptqmodel.utils.qvq_acceptance as acceptance
from gptqmodel.utils.qvq_acceptance import (
    PROJECTION_BOUNDARY,
    AcceptanceError,
    ProjectionCell,
    account_serialized_state,
    expected_cells,
    materialize_manifest,
    validate_acceptance_report,
    validate_manifest_disjointness,
)


class _Packed(nn.Module):
    def __init__(self, bits=2.0):
        super().__init__()
        self.bits = bits
        self.in_features = 16
        self.out_features = 16
        self.register_buffer("trellis", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("SU", torch.zeros(1))
        self.register_buffer("SV", torch.zeros(1))


def _census_model(module_factory):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _layer in range(36):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.mlp = nn.Module()
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, role, module_factory())
        for role in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, role, module_factory())
        model.model.layers.append(block)
    return model


def test_census_rejects_missing_module(monkeypatch):
    monkeypatch.setattr(acceptance, "QVQLinear", _Packed)
    model = _census_model(_Packed)
    del model.model.layers[9].mlp.down_proj
    with pytest.raises(AcceptanceError, match="missing"):
        acceptance.census_reloaded_model(model)


def test_census_rejects_dense_fallback(monkeypatch):
    monkeypatch.setattr(acceptance, "QVQLinear", _Packed)
    model = _census_model(_Packed)
    model.model.layers[3].self_attn.q_proj = nn.Linear(16, 16)
    with pytest.raises(AcceptanceError, match="not the exact packed"):
        acceptance.census_reloaded_model(model)


def test_census_rejects_higher_precision_fallback(monkeypatch):
    monkeypatch.setattr(acceptance, "QVQLinear", _Packed)
    model = _census_model(_Packed)
    model.model.layers[3].self_attn.q_proj = _Packed(bits=2.5)
    with pytest.raises(AcceptanceError, match="higher-precision"):
        acceptance.census_reloaded_model(model)


def _manifests(tmp_path):
    result = {}
    for index, split in enumerate(("calibration", "yaqa_tuning", "validation", "held_out_diagnostics", "diverse_32")):
        count = 32 if split == "diverse_32" else 1
        records = [{"identity": f"{split}:{row}", "content": f"content-{index}-{row}"} for row in range(count)]
        result[split] = materialize_manifest(split, records, tmp_path / f"{split}.json")
    return result


def test_manifest_rejects_content_leakage_even_with_distinct_identity(tmp_path):
    manifests = _manifests(tmp_path)
    manifests["validation"]["samples"][0]["content_sha256"] = manifests["calibration"]["samples"][0][
        "content_sha256"
    ]
    with pytest.raises(AcceptanceError, match="sample leakage"):
        validate_manifest_disjointness(manifests)


def test_diverse_manifest_requires_exactly_32(tmp_path):
    with pytest.raises(AcceptanceError, match="exactly 32"):
        materialize_manifest(
            "diverse_32", [{"identity": "one", "content": "one"}], tmp_path / "manifest.json"
        )


def test_accounting_includes_every_projection_auxiliary_tensor_and_bias():
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 4, 4)
    state = {
        f"{cell.name}.trellis": torch.zeros(1, dtype=torch.int32),
        f"{cell.name}.SU": torch.zeros(1, dtype=torch.float32),
        f"{cell.name}.SV": torch.zeros(1, dtype=torch.float32),
        f"{cell.name}.bank_ids": torch.zeros(1, dtype=torch.uint8),
        f"{cell.name}.bias": torch.zeros(1, dtype=torch.float16),
        f"{cell.name}.future_outliers": torch.zeros(1, dtype=torch.float32),
        "model.embed_tokens.weight": torch.zeros(3, dtype=torch.float16),
    }
    report = account_serialized_state(state, [cell], maximum_bpw=100)
    assert report["requested_projection_tensor_bytes"] == 19
    assert report["dense_non_target_tensor_bytes"] == 6
    assert set(report["per_module"][cell.name]["tensors"]) == set(state) - {"model.embed_tokens.weight"}


def test_accounting_rejects_bpw_above_limit():
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 1, 1)
    with pytest.raises(AcceptanceError, match="exceeds"):
        account_serialized_state({f"{cell.name}.trellis": torch.zeros(1, dtype=torch.int32)}, [cell])


def _complete_report(score=0.85, kl=0.1):
    metric = {
        "coverage_complete": True,
        "top1_agreement": score,
        "final_kl_nats": kl,
        "diverse_32": score,
    }
    return {
        "schema_version": 1,
        "artifact": {"fresh_reload_verified": True, "checkpoint_sha256": {"model.safetensors": "a" * 64}},
        "census": {"expected": 252, "actual": 252, "complete": True},
        "accounting": {"boundary": PROJECTION_BOUNDARY, "effective_bpw": 2.05, "maximum_bpw": 2.1},
        "manifests": {
            "pairwise_disjoint": True,
            "comparisons": [
                {"identity_overlap": [], "content_overlap": []}
                for _ in range(10)
            ],
        },
        "thresholds": {"top1_agreement_min": 0.85, "diverse_32_min": 0.85, "final_kl_max_nats": 0.2},
        "global": dict(metric),
        "cells": [{"layer": layer, "role": role, **metric} for layer, role in expected_cells()],
    }


def test_report_rejects_missing_layer_role_cell():
    report = _complete_report()
    report["cells"].pop()
    with pytest.raises(AcceptanceError, match="incomplete result coverage"):
        validate_acceptance_report(report)


@pytest.mark.parametrize(("field", "value"), [("top1_agreement", 0.849), ("diverse_32", 0.849)])
def test_report_rejects_score_threshold_failure(field, value):
    report = _complete_report()
    report["cells"][100][field] = value
    with pytest.raises(AcceptanceError, match="below threshold"):
        validate_acceptance_report(report)


def test_report_rejects_final_kl_as_percentage_semantics():
    report = _complete_report()
    report["thresholds"].pop("final_kl_max_nats")
    report["thresholds"]["final_kl_min"] = 0.85
    with pytest.raises(AcceptanceError, match="explicit finite nonnegative KL threshold"):
        validate_acceptance_report(report)


def test_report_rejects_incomplete_coverage_and_schema():
    report = _complete_report()
    report["schema_version"] = 2
    with pytest.raises(AcceptanceError, match="schema"):
        validate_acceptance_report(report)
    report = _complete_report()
    report["global"]["coverage_complete"] = False
    with pytest.raises(AcceptanceError, match="coverage"):
        validate_acceptance_report(report)


def test_materialized_manifest_is_stable(tmp_path):
    records = [{"identity": "source:7", "content": {"b": 2, "a": 1}}]
    first = materialize_manifest("validation", records, tmp_path / "first.json")
    second = materialize_manifest("validation", records, tmp_path / "second.json")
    assert first == second
    assert json.loads((tmp_path / "first.json").read_text()) == first
