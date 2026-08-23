# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import gptqmodel.utils.qvq_acceptance as acceptance
import scripts.accept_qwen3_8b_qvq as acceptance_script
from gptqmodel.utils.qvq_acceptance import (
    MANIFEST_SPLITS,
    PROJECTION_BOUNDARY,
    QWEN3_DENSE_ARTIFACT_SHA256,
    QWEN3_DENSE_CONFIG_SHA256,
    QWEN3_MODEL_CONFIG,
    QWEN3_PINNED_REVISION,
    AcceptanceError,
    ProjectionCell,
    account_serialized_checkpoint,
    account_serialized_state,
    expected_cells,
    expected_projection_dimensions,
    expected_projection_name,
    expected_projection_names,
    hash_qvq_module_payloads,
    materialize_manifest,
    seal_acceptance_observation,
    select_diverse_32,
    validate_acceptance_report,
    validate_manifest_disjointness,
    validate_qwen3_model_artifact,
)
from scripts.accept_qwen3_8b_qvq import build_parser as build_acceptance_parser

FROZEN_SPLITS = Path(__file__).parent / "data" / "qwen3_8b_qvq_acceptance"


class _Packed(nn.Module):
    def __init__(self, bits=2.0, in_features=16, out_features=16):
        super().__init__()
        self.bits = bits
        self.bank_count = 2
        self.v2b2_p32 = True
        self._bank_ids_loaded = True
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("trellis", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("SU", torch.zeros(1))
        self.register_buffer("SV", torch.zeros(1))
        self.register_buffer("bank_ids", torch.zeros(1, dtype=torch.uint8))
        self.register_buffer("bank_alt_id", torch.zeros(1, dtype=torch.uint8))


def _census_model(module_factory):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _layer in range(36):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.mlp = nn.Module()
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(
                block.self_attn,
                role,
                module_factory(
                    in_features=expected_projection_dimensions(role)[0],
                    out_features=expected_projection_dimensions(role)[1],
                ),
            )
        for role in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                block.mlp,
                role,
                module_factory(
                    in_features=expected_projection_dimensions(role)[0],
                    out_features=expected_projection_dimensions(role)[1],
                ),
            )
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
    model.model.layers[3].self_attn.q_proj = _Packed(
        bits=2.5, in_features=4096, out_features=4096
    )
    with pytest.raises(AcceptanceError, match="higher-precision"):
        acceptance.census_reloaded_model(model)


def test_census_separates_wrapper_runtime_name_from_serialized_identity(monkeypatch):
    monkeypatch.setattr(acceptance, "QVQLinear", _Packed)
    wrapper = nn.Module()
    wrapper.model = _census_model(_Packed)
    cells = acceptance.census_reloaded_model(wrapper)
    assert len(cells) == 252
    assert cells[0].name == "model.layers.0.self_attn.q_proj"
    assert cells[0].runtime_name == "model.model.layers.0.self_attn.q_proj"


def test_census_rejects_role_dimension_drift(monkeypatch):
    monkeypatch.setattr(acceptance, "QVQLinear", _Packed)
    model = _census_model(_Packed)
    model.model.layers[7].self_attn.k_proj.out_features = 4096
    with pytest.raises(AcceptanceError, match="dimension mismatch"):
        acceptance.census_reloaded_model(model)


def _manifests(tmp_path):
    result = {}
    for index, split in enumerate(MANIFEST_SPLITS):
        count = {"diverse_pool_512": 512, "diverse_32": 32}.get(split, 1)
        records = [
            {"identity": f"{split}:{row}", "content": f"content-{index}-{row}"}
            for row in range(count)
        ]
        result[split] = materialize_manifest(split, records, tmp_path / f"{split}.json")
    return result


def test_manifest_rejects_content_leakage_even_with_distinct_identity(tmp_path):
    manifests = _manifests(tmp_path)
    manifests["validation"]["samples"][0]["content_sha256"] = manifests["calibration"][
        "samples"
    ][0]["content_sha256"]
    with pytest.raises(AcceptanceError, match="sample leakage"):
        validate_manifest_disjointness(manifests)


def test_diverse_manifest_requires_exactly_32(tmp_path):
    with pytest.raises(AcceptanceError, match="exactly 32"):
        materialize_manifest(
            "diverse_32",
            [{"identity": "one", "content": "one"}],
            tmp_path / "manifest.json",
        )


def test_diverse_pool_selection_is_deterministic_and_content_bound():
    pool = [
        {"identity": f"pool:{index:03d}", "content": "x" * (512 - index)}
        for index in range(512)
    ]
    selected = select_diverse_32(pool)
    assert selected == select_diverse_32(list(reversed(pool)))
    assert len(selected) == 32
    drifted = list(selected)
    drifted[0] = pool[0]
    with pytest.raises(AcceptanceError, match="not the deterministic"):
        acceptance.validate_diverse_selection(pool, drifted)


def test_committed_frozen_pool_and_selection_are_content_bound_and_disjoint():
    acceptance_script._verify_all_manifest_sources(FROZEN_SPLITS)
    evidence = acceptance_script._check_manifests(FROZEN_SPLITS)
    assert evidence["counts"]["diverse_pool_512"] == 512
    assert evidence["counts"]["diverse_32"] == 32
    assert evidence["diverse_selection"]["verified"] is True
    assert len(evidence["comparisons"]) == 14


def test_payload_hashes_measure_tensor_bytes_and_fail_on_drift():
    module = _Packed()
    before = hash_qvq_module_payloads([("module", module)])
    assert before == hash_qvq_module_payloads([("module", module)])
    module.trellis[0] = 1
    after = hash_qvq_module_payloads([("module", module)])
    assert before["module_sha256"]["module"] != after["module_sha256"]["module"]
    assert before["aggregate_sha256"] != after["aggregate_sha256"]


def test_accounting_includes_every_projection_auxiliary_tensor_and_bias():
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 16, 16)
    state = {
        f"{cell.name}.trellis": torch.zeros((1, 16), dtype=torch.int32),
        f"{cell.name}.SU": torch.zeros(16, dtype=torch.float32),
        f"{cell.name}.SV": torch.zeros(16, dtype=torch.float32),
        f"{cell.name}.bank_ids": torch.zeros(1, dtype=torch.uint8),
        f"{cell.name}.bank_alt_id": torch.zeros(1, dtype=torch.uint8),
        f"{cell.name}.bias": torch.zeros(1, dtype=torch.float16),
        f"{cell.name}.future_outliers": torch.zeros(1, dtype=torch.float32),
        "model.embed_tokens.weight": torch.zeros(3, dtype=torch.float16),
    }
    report = account_serialized_state(state, [cell], maximum_bpw=100)
    assert report["requested_projection_tensor_bytes"] == 200
    assert report["dense_non_target_tensor_bytes"] == 6
    assert set(report["per_module"][cell.name]["tensors"]) == set(state) - {
        "model.embed_tokens.weight"
    }


def test_accounting_rejects_bpw_above_limit():
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 16, 16)
    with pytest.raises(AcceptanceError, match="exceeds"):
        account_serialized_state(
            {
                f"{cell.name}.trellis": torch.zeros((1, 16), dtype=torch.int32),
                f"{cell.name}.SU": torch.zeros(16),
                f"{cell.name}.SV": torch.zeros(16),
                f"{cell.name}.bank_ids": torch.zeros(1, dtype=torch.uint8),
                f"{cell.name}.bank_alt_id": torch.zeros(1, dtype=torch.uint8),
            },
            [cell],
            maximum_bpw=1,
        )


@pytest.mark.parametrize(("shape", "dtype"), [((2, 16), torch.int32), ((1, 16), torch.int64)])
def test_accounting_rejects_wrong_required_packed_tensor_shape_or_dtype(shape, dtype):
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 16, 16)
    state = {
        f"{cell.name}.trellis": torch.zeros(shape, dtype=dtype),
        f"{cell.name}.SU": torch.zeros(16, dtype=torch.float32),
        f"{cell.name}.SV": torch.zeros(16, dtype=torch.float32),
        f"{cell.name}.bank_ids": torch.zeros(1, dtype=torch.uint8),
        f"{cell.name}.bank_alt_id": torch.ones(1, dtype=torch.uint8),
    }
    with pytest.raises(AcceptanceError, match="packed tensor metadata mismatch"):
        account_serialized_state(state, [cell], maximum_bpw=100)


def test_accounting_reads_exact_saved_safetensors_extents(tmp_path, monkeypatch):
    cells = [
        ProjectionCell(
            layer,
            role,
            expected_projection_name(layer, role),
            *expected_projection_dimensions(role),
        )
        for layer, role in expected_cells()
    ]
    metadata = {"model.embed_tokens.weight": {"bytes": 16, "dtype": "F16", "shape": [8]}}
    for cell in cells:
        tile_count = (cell.in_features // 16) * (cell.out_features // 16)
        for suffix, shape, dtype, size in (
            ("trellis", [tile_count, 16], "I32", tile_count * 64),
            ("SU", [cell.in_features], "F32", cell.in_features * 4),
            ("SV", [cell.out_features], "F32", cell.out_features * 4),
            ("bank_ids", [tile_count], "U8", tile_count),
            ("bank_alt_id", [1], "U8", 1),
        ):
            metadata[f"{cell.name}.{suffix}"] = {"bytes": size, "dtype": dtype, "shape": shape}
    tensor_bytes = sum(item["bytes"] for item in metadata.values())
    monkeypatch.setattr(acceptance, "_serialized_tensor_metadata", lambda _path: (metadata, {"model.safetensors": tensor_bytes + 8}))
    report = account_serialized_checkpoint(tmp_path, cells, maximum_bpw=100)
    assert report["requested_projection_tensor_bytes"] == tensor_bytes - 16
    assert report["dense_non_target_tensor_bytes"] == 16
    assert report["container_header_and_padding_bytes"] > 0


def test_accounting_rejects_shards_without_authoritative_index(tmp_path):
    cell = ProjectionCell(0, "q_proj", "model.layers.0.self_attn.q_proj", 16, 16)
    save_file(
        {f"{cell.name}.trellis": torch.zeros(1, dtype=torch.int32)},
        tmp_path / "model-00001.safetensors",
    )
    save_file(
        {
            f"{cell.name}.SU": torch.zeros(1),
            f"{cell.name}.SV": torch.zeros(1),
        },
        tmp_path / "model-00002.safetensors",
    )
    with pytest.raises(AcceptanceError, match="lacks model.safetensors.index.json"):
        account_serialized_checkpoint(tmp_path, [cell], maximum_bpw=100)


def test_accounting_contextualizes_malformed_shard_index(tmp_path):
    save_file({"tensor": torch.zeros(1)}, tmp_path / "model-00001.safetensors")
    save_file({"other": torch.zeros(1)}, tmp_path / "model-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="failed to parse safetensors index"):
        account_serialized_checkpoint(tmp_path, [], maximum_bpw=100)


def _complete_report(score=0.85, kl=0.1):
    metric = {
        "coverage_complete": True,
        "sample_count": 1056,
        "diverse_32_sample_count": 32,
        "top1_agreement": score,
        "final_kl_nats": kl,
        "diverse_32": score,
    }
    target_tensors = {}
    per_module = {}
    for module_name in expected_projection_names():
        role = module_name.rsplit(".", 1)[-1]
        in_features, out_features = expected_projection_dimensions(role)
        tile_count = (in_features // 16) * (out_features // 16)
        tensors = {
            f"{module_name}.trellis": {"bytes": tile_count * 64, "shape": [tile_count, 16], "dtype": "I32"},
            f"{module_name}.SU": {"bytes": in_features * 4, "shape": [in_features], "dtype": "F32"},
            f"{module_name}.SV": {"bytes": out_features * 4, "shape": [out_features], "dtype": "F32"},
            f"{module_name}.bank_ids": {"bytes": tile_count, "shape": [tile_count], "dtype": "U8"},
            f"{module_name}.bank_alt_id": {"bytes": 1, "shape": [1], "dtype": "U8"},
        }
        target_tensors.update({key: value["bytes"] for key, value in tensors.items()})
        module_bytes = sum(item["bytes"] for item in tensors.values())
        per_module[module_name] = {
            "bytes": module_bytes,
            "dense_weight_count": in_features * out_features,
            "tensors": tensors,
        }
    target_bytes = sum(record["bytes"] for record in per_module.values())
    dense_weights = sum(record["dense_weight_count"] for record in per_module.values())
    comparisons = [
        {"left": left, "right": right, "identity_overlap": [], "content_overlap": []}
        for left_index, left in enumerate(MANIFEST_SPLITS)
        for right in MANIFEST_SPLITS[left_index + 1 :]
        if (left, right) != ("diverse_pool_512", "diverse_32")
    ]
    manifest_hashes = {split: "d" * 64 for split in MANIFEST_SPLITS}
    content_hashes = {split: "e" * 64 for split in MANIFEST_SPLITS}
    quantization_streams = {
        quant_name: {
            "manifest_verified": True,
            "content_sha256": content_hashes[manifest_name],
            "identity_manifest_sha256": manifest_hashes[manifest_name],
        }
        for quant_name, manifest_name in (
            ("calibration", "calibration"),
            ("yaqa", "yaqa_tuning"),
            ("validation", "validation"),
        )
    }
    checkpoint_config_sha256 = "c" * 64
    payload_hashes = {
        "scheme": acceptance.QVQ_PAYLOAD_HASH_SCHEME,
        "module_count": 252,
        "module_sha256": {name: "f" * 64 for name in expected_projection_names()},
        "module_tensor_counts": {name: 5 for name in expected_projection_names()},
        "aggregate_sha256": "a" * 64,
    }
    run_id = "run-acceptance-test"
    producer_process_id = "process-quantizer"
    start = seal_acceptance_observation(
        {
            "stage": "quantization_start",
            "run_id": run_id,
            "process_id": producer_process_id,
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
        }
    )
    end = seal_acceptance_observation(
        {
            "stage": "quantization_end",
            "run_id": run_id,
            "process_id": producer_process_id,
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
            "previous_observation_sha256": start["observation_sha256"],
        }
    )
    pre_save = seal_acceptance_observation(
        {
            "stage": "pre_save_in_memory",
            "run_id": run_id,
            "process_id": producer_process_id,
            "dense_source_end_sha256": end["observation_sha256"],
            "payload": payload_hashes,
        }
    )
    fresh_process = seal_acceptance_observation(
        {
            "stage": "fresh_process_reload",
            "run_id": run_id,
            "process_id": "process-fresh",
            "previous_observation_sha256": pre_save["observation_sha256"],
            "payload": payload_hashes,
        }
    )
    evaluation_reload = seal_acceptance_observation(
        {
            "stage": "acceptance_evaluation_reload",
            "run_id": run_id,
            "process_id": "process-evaluation",
            "previous_observation_sha256": fresh_process["observation_sha256"],
            "payload": payload_hashes,
        }
    )
    return {
        "schema_version": 3,
        "artifact": {
            "fresh_reload_verified": True,
            "checkpoint_sha256": {
                "config.json": checkpoint_config_sha256,
                "model.safetensors": "a" * 64,
            },
            "dense_model_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
            "dense_model_identity": {
                "config": dict(QWEN3_MODEL_CONFIG),
                "config_sha256": QWEN3_DENSE_CONFIG_SHA256,
                "decoder_layers": list(range(36)),
                "revision": QWEN3_PINNED_REVISION,
                "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
            },
            "checkpoint_model_identity": {
                "config": dict(QWEN3_MODEL_CONFIG),
                "config_sha256": checkpoint_config_sha256,
                "decoder_layers": list(range(36)),
                "revision": None,
            },
            "packed_payload_parity": {
                "verified": True,
                "pre_save": pre_save,
                "fresh_process": fresh_process,
                "evaluation_reload": evaluation_reload,
            },
            "dense_source_binding": {
                "run_id": run_id,
                "producer_process_id": producer_process_id,
                "start": start,
                "end": end,
            },
        },
        "census": {"expected": 252, "actual": 252, "complete": True},
        "accounting": {
            "boundary": PROJECTION_BOUNDARY,
            "effective_bpw": 8 * target_bytes / dense_weights,
            "maximum_bpw": 2.1,
            "requested_projection_tensor_bytes": target_bytes,
            "requested_projection_dense_weight_count": dense_weights,
            "target_tensors": target_tensors,
            "dense_non_target_tensors": {"model.embed_tokens.weight": 16},
            "dense_non_target_tensor_bytes": 16,
            "per_module": per_module,
        },
        "manifests": {
            "pairwise_disjoint": True,
            "counts": {
                "calibration": 512,
                "yaqa_tuning": 512,
                "validation": 512,
                "held_out_diagnostics": 512,
                "diverse_pool_512": 512,
                "diverse_32": 32,
            },
            "comparisons": comparisons,
            "diverse_selection": {
                "verified": True,
                "scheme": acceptance.DIVERSE_SELECTION_SCHEME,
                "pool_count": 512,
                "selected_count": 32,
                "selected_identities": [f"pool:{index}" for index in range(32)],
                "selected_content_sha256": [f"{index:064x}" for index in range(32)],
            },
            "manifest_sha256": manifest_hashes,
            "content_jsonl_sha256": content_hashes,
            "quantization_streams": quantization_streams,
        },
        "thresholds": {
            "top1_agreement_min": 0.85,
            "diverse_32_min": 0.85,
            "final_kl_max_nats": 0.2,
        },
        "global": dict(metric),
        "cells": [
            {
                "layer": layer,
                "role": role,
                "module": expected_projection_name(layer, role),
                **metric,
            }
            for layer, role in expected_cells()
        ],
    }


def test_report_rejects_missing_layer_role_cell():
    report = _complete_report()
    report["cells"].pop()
    with pytest.raises(AcceptanceError, match="incomplete result coverage"):
        validate_acceptance_report(report)


def test_report_rejects_noncanonical_or_inconsistent_accounting():
    report = _complete_report()
    first_name = expected_projection_names()[0]
    report["accounting"]["per_module"]["not.canonical"] = report["accounting"][
        "per_module"
    ].pop(first_name)
    with pytest.raises(AcceptanceError, match="exact canonical 252"):
        validate_acceptance_report(report)

    report = _complete_report()
    first_name = expected_projection_names()[0]
    report["accounting"]["per_module"][first_name]["dense_weight_count"] += 1
    report["accounting"]["requested_projection_dense_weight_count"] += 1
    with pytest.raises(AcceptanceError, match="denominator is not pinned"):
        validate_acceptance_report(report)


@pytest.mark.parametrize(("field", "value"), [("shape", [1, 16]), ("dtype", "I64")])
def test_report_rejects_wrong_canonical_packed_tensor_shape_or_dtype(field, value):
    report = _complete_report()
    module_name = expected_projection_names()[0]
    report["accounting"]["per_module"][module_name]["tensors"][f"{module_name}.trellis"][field] = value
    with pytest.raises(AcceptanceError, match="packed tensor metadata mismatch"):
        validate_acceptance_report(report)


def test_report_rejects_asserted_or_drifted_payload_parity():
    report = _complete_report()
    report["artifact"]["packed_payload_parity"] = {"verified": True}
    with pytest.raises(AcceptanceError, match="missing|structurally sealed"):
        validate_acceptance_report(report)

    report = _complete_report()
    report["artifact"]["packed_payload_parity"]["fresh_process"]["payload"]["aggregate_sha256"] = "0" * 64
    with pytest.raises(AcceptanceError, match="structurally sealed"):
        validate_acceptance_report(report)

    report = _complete_report()
    first_name = expected_projection_names()[0]
    report["accounting"]["per_module"][first_name]["bytes"] += 1
    with pytest.raises(AcceptanceError, match="byte subtotal disagrees"):
        validate_acceptance_report(report)


@pytest.mark.parametrize("failure", ["missing", "digest_mismatch"])
def test_report_rejects_missing_or_mismatched_quantization_dense_source_binding(failure):
    report = _complete_report()
    if failure == "missing":
        report["artifact"].pop("dense_source_binding")
        message = "lacks quantization-start/end"
    else:
        end = report["artifact"]["dense_source_binding"]["end"]
        end["artifact_sha256"] = {**QWEN3_DENSE_ARTIFACT_SHA256, "config.json": "0" * 64}
        end.update(seal_acceptance_observation(end))
        message = "mismatched, or unpinned"
    with pytest.raises(AcceptanceError, match=message):
        validate_acceptance_report(report)


@pytest.mark.parametrize("failure", ["same_process", "broken_chain"])
def test_report_rejects_unindependent_or_unchained_reload_parity(failure):
    report = _complete_report()
    parity = report["artifact"]["packed_payload_parity"]
    fresh = parity["fresh_process"]
    if failure == "same_process":
        fresh["process_id"] = parity["pre_save"]["process_id"]
        message = "independently measured"
    else:
        fresh["previous_observation_sha256"] = "0" * 64
        message = "observation chain"
    parity["fresh_process"] = seal_acceptance_observation(fresh)
    with pytest.raises(AcceptanceError, match=message):
        validate_acceptance_report(report)


@pytest.mark.parametrize(("top1", "diverse"), [(0.85, 0.1), (0.1, 0.85)])
def test_report_accepts_either_score_alternative(top1, diverse):
    report = _complete_report()
    report["global"]["top1_agreement"] = top1
    report["global"]["diverse_32"] = diverse
    report["cells"][100]["top1_agreement"] = top1
    report["cells"][100]["diverse_32"] = diverse
    validate_acceptance_report(report)


def test_report_rejects_when_both_score_alternatives_fail():
    report = _complete_report()
    report["cells"][100]["top1_agreement"] = 0.849
    report["cells"][100]["diverse_32"] = 0.849
    with pytest.raises(AcceptanceError, match="fails both score alternatives"):
        validate_acceptance_report(report)


def test_report_rejects_final_kl_as_percentage_semantics():
    report = _complete_report()
    report["thresholds"].pop("final_kl_max_nats")
    report["thresholds"]["final_kl_min"] = 0.85
    with pytest.raises(
        AcceptanceError, match="explicit finite nonnegative KL threshold"
    ):
        validate_acceptance_report(report)


def test_report_rejects_incomplete_coverage_and_schema():
    report = _complete_report()
    report["schema_version"] = 4
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


def test_report_rejects_duplicate_or_wrong_manifest_pair():
    report = _complete_report()
    report["manifests"]["comparisons"][9] = dict(report["manifests"]["comparisons"][0])
    with pytest.raises(AcceptanceError, match="exact unique canonical split pairs"):
        validate_acceptance_report(report)


def test_gate_parser_has_no_report_only_acceptance_mode():
    with pytest.raises(SystemExit):
        build_acceptance_parser().parse_args(["gate", "--report", "report.json"])


def test_gate_recomputes_and_content_binds_submitted_report(tmp_path, monkeypatch):
    report = _complete_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    calls = []

    def fake_evaluate(args):
        calls.append(args)
        args.output.write_text(json.dumps(report), encoding="utf-8")
        return 0

    monkeypatch.setattr(acceptance_script, "_evaluate", fake_evaluate)
    assert acceptance_script._gate(SimpleNamespace(report=report_path)) == 0
    assert len(calls) == 1

    changed = _complete_report()
    changed["global"]["final_kl_nats"] = 0.11
    report_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        AcceptanceError, match="does not exactly match artifact-recomputed"
    ):
        acceptance_script._gate(SimpleNamespace(report=report_path))


def test_quantization_stream_verification_rejects_content_mutation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    manifests = tmp_path / "splits"
    dense = tmp_path / "dense"
    checkpoint.mkdir()
    manifests.mkdir()
    dense.mkdir()
    datasets = {}
    for quant_name, file_stem in (
        ("calibration", "calibration"),
        ("yaqa", "yaqa_tuning"),
        ("validation", "validation"),
    ):
        source = manifests / f"{file_stem}.jsonl"
        manifest = manifests / f"{file_stem}.manifest.json"
        source.write_text('{"identity":"x","content":"sample"}\n', encoding="utf-8")
        manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
        datasets[quant_name] = {
            "source": str(source.resolve()),
            "row_start": 0,
            "rows": 512,
            "content_sha256": acceptance_script._sha256_file(source),
            "identity_manifest": str(manifest.resolve()),
            "identity_manifest_sha256": acceptance_script._sha256_file(manifest),
        }
    run = {
        "model": str(dense.resolve()),
        "layer_scope": "all",
        "quantize_config": {
            "bits": 2,
            "format": "qvq_v2b2_p32",
            "group_size": -1,
            "rounding": "yaqa",
            "sym": True,
            "pack_dtype": "int32",
            "bank_count": 2,
        },
        "datasets": datasets,
        "packed_payload_parity": {
            "verified": True,
            "pre_save": {"aggregate_sha256": "a" * 64},
            "fresh_process": {"aggregate_sha256": "a" * 64},
        },
    }
    start = seal_acceptance_observation(
        {
            "stage": "quantization_start",
            "run_id": "run",
            "process_id": "producer",
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
        }
    )
    end = seal_acceptance_observation(
        {
            "stage": "quantization_end",
            "run_id": "run",
            "process_id": "producer",
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
            "previous_observation_sha256": start["observation_sha256"],
        }
    )
    run["dense_source_binding"] = {
        "run_id": "run",
        "producer_process_id": "producer",
        "start": start,
        "end": end,
    }
    pre_save = seal_acceptance_observation(
        {
            "stage": "pre_save_in_memory",
            "run_id": "run",
            "process_id": "producer",
            "dense_source_end_sha256": end["observation_sha256"],
            "payload": {"aggregate_sha256": "a" * 64},
        }
    )
    fresh_process = seal_acceptance_observation(
        {
            "stage": "fresh_process_reload",
            "run_id": "run",
            "process_id": "fresh",
            "previous_observation_sha256": pre_save["observation_sha256"],
            "payload": {"aggregate_sha256": "a" * 64},
        }
    )
    run["packed_payload_parity"] = {
        "verified": True,
        "pre_save": pre_save,
        "fresh_process": fresh_process,
    }
    (checkpoint / "qvq_quantize_run.json").write_text(json.dumps(run), encoding="utf-8")
    acceptance_script._verify_quantization_streams(checkpoint, manifests, dense)

    (manifests / "calibration.jsonl").write_text(
        '{"content":"changed"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="content hash"):
        acceptance_script._verify_quantization_streams(checkpoint, manifests, dense)


def test_local_qwen3_target_has_exact_pinned_identity():
    evidence = validate_qwen3_model_artifact(
        Path("/monster/data/model/Qwen3-8B"), require_pinned_dense=True
    )
    assert evidence["revision"] == QWEN3_PINNED_REVISION
    assert evidence["decoder_layers"] == list(range(36))
    assert evidence["config"]["architectures"] == ["Qwen3ForCausalLM"]
    assert evidence["artifact_sha256"] == QWEN3_DENSE_ARTIFACT_SHA256
    assert "not Qwen3-8B-Instruct" in evidence["instruction_identity"]


@pytest.mark.parametrize("failure", ["missing", "extra", "mismatch"])
def test_pinned_dense_hash_authority_fails_closed(failure):
    hashes = dict(QWEN3_DENSE_ARTIFACT_SHA256)
    if failure == "missing":
        hashes.pop("model-00005-of-00005.safetensors")
        message = "file set mismatch"
    elif failure == "extra":
        hashes["model-00006-of-00006.safetensors"] = "0" * 64
        message = "file set mismatch"
    else:
        hashes["model-00003-of-00005.safetensors"] = "0" * 64
        message = "SHA-256 mismatch"
    with pytest.raises(AcceptanceError, match=message):
        acceptance._validate_pinned_dense_hashes(hashes)


def test_model_identity_contextualizes_malformed_config_json(tmp_path):
    (tmp_path / "config.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(AcceptanceError, match=r"failed to parse Qwen3 config\.json"):
        validate_qwen3_model_artifact(tmp_path, require_pinned_dense=False)
