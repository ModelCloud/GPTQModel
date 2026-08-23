# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import gptqmodel.utils.qvq_acceptance as acceptance
import gptqmodel.utils.qvq_acceptance_controller as controller_module
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
    validate_manifest_disjointness,
    validate_qwen3_model_artifact,
)
from gptqmodel.utils.qvq_acceptance import (
    validate_acceptance_report as _validate_acceptance_report,
)
from gptqmodel.utils.qvq_acceptance_controller import (
    AcceptanceController,
    controller_authority_receipt,
)
from scripts.accept_qwen3_8b_qvq import build_parser as build_acceptance_parser

FROZEN_SPLITS = Path(__file__).parent / "data" / "qwen3_8b_qvq_acceptance"


@pytest.fixture(autouse=True)
def _trusted_test_verifier(tmp_path, monkeypatch):
    private = tmp_path / "verifier-private.pem"
    public = tmp_path / "verifier-public.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)], check=True)
    subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)], check=True)
    private.chmod(0o600)
    python = Path(sys.executable).resolve()
    openssl = Path(subprocess.check_output(["which", "openssl"], text=True).strip()).resolve()
    scripts = {
        "quantization_producer": Path.cwd() / "scripts" / "qvq_quantize.py",
        "fresh_process_reload": Path.cwd() / "scripts" / "accept_qwen3_8b_qvq.py",
        "acceptance_evaluation": Path.cwd() / "scripts" / "accept_qwen3_8b_qvq.py",
    }
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({
        "schema": "qvq-acceptance-trust-v1",
        "verifier_public_key": str(public),
        "verifier_public_key_sha256": acceptance_script._sha256_file(public),
        "python_executable": str(python),
        "python_executable_sha256": acceptance_script._sha256_file(python),
        "openssl_executable": str(openssl),
        "openssl_executable_sha256": acceptance_script._sha256_file(openssl),
        "acceptance_policy": str(Path.cwd() / "configs" / "qwen3_8b_qvq_acceptance_policy.json"),
        "acceptance_policy_sha256": acceptance_script._sha256_file(
            Path.cwd() / "configs" / "qwen3_8b_qvq_acceptance_policy.json"
        ),
        "quant_config": str(Path.cwd() / "configs" / "qwen3_8b_qvq_w2_acceptance.json"),
        "quant_config_sha256": acceptance_script._sha256_file(
            Path.cwd() / "configs" / "qwen3_8b_qvq_w2_acceptance.json"
        ),
        "stage_scripts": {
            stage: {"path": str(path), "sha256": acceptance_script._sha256_file(path)}
            for stage, path in scripts.items()
        },
    }), encoding="utf-8")
    trust.chmod(0o600)
    monkeypatch.setenv(controller_module.VERIFIER_PRIVATE_KEY_ENV, str(private))
    monkeypatch.setenv(controller_module.TRUST_CONFIG_ENV, str(trust))
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for split, stem in (("calibration", "calibration"), ("yaqa_tuning", "yaqa_tuning"),
                        ("validation", "validation")):
        records = [
            {"identity": f"{split}:{index}", "content": {"split": split, "index": index}}
            for index in range(512)
        ]
        source = input_root / f"{stem}.jsonl"
        source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        acceptance.materialize_manifest(split, records, input_root / f"{stem}.manifest.json", expected_count=512)
    monkeypatch.setenv("GPTQMODEL_QVQ_TEST_INPUT_ROOT", str(input_root))


def validate_acceptance_report(report):
    _validate_acceptance_report(
        report,
        controller_authority=report.get("_test_controller_authority"),
    )


def _resign_controller_report(report):
    transcript = report["artifact"]["acceptance_controller"]
    previous = None
    for record in transcript["processes"]:
        record["event_sha256"] = hashlib.sha256(acceptance.canonical_content(record["event"])).hexdigest()
        record["acknowledgement"]["acknowledged_event_sha256"] = record["event_sha256"]
        record["previous_record_sha256"] = previous
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        record["record_sha256"] = hashlib.sha256(acceptance.canonical_content(unsigned)).hexdigest()
        previous = record["record_sha256"]
    transcript.pop("controller_signature_ed25519", None)
    private = Path(os.environ[controller_module.VERIFIER_PRIVATE_KEY_ENV]).read_bytes()
    transcript["controller_signature_ed25519"] = controller_module._sign(
        private, acceptance.canonical_content(transcript)
    )
    report["_test_controller_authority"] = controller_authority_receipt(transcript)


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


def _complete_report(score=0.85, kl=0.1, diverse_score=None):
    diverse_score = score if diverse_score is None else diverse_score
    metric = {
        "coverage_complete": True,
        "sample_count": 1056,
        "diverse_32_sample_count": 32,
        "top1_agreement": score,
        "final_kl_nats": kl,
        "diverse_32": diverse_score,
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
        "aggregate_sha256": "",
    }
    payload_hashes["aggregate_sha256"] = acceptance.qwen3_payload_aggregate(
        payload_hashes["module_sha256"], payload_hashes["module_tensor_counts"]
    )
    controller = AcceptanceController()
    producer_instance, reload_instance, evaluation_instance = (secrets.token_hex(32) for _ in range(3))
    producer_nonce, reload_nonce, evaluation_nonce = (secrets.token_hex(32) for _ in range(3))
    start = seal_acceptance_observation(
        {
            "stage": "quantization_start",
            "controller_run_nonce": controller.run_nonce,
            "process_instance_id": producer_instance,
            "producer_stage_nonce": producer_nonce,
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
        }
    )
    end = seal_acceptance_observation(
        {
            "stage": "quantization_end",
            "controller_run_nonce": controller.run_nonce,
            "process_instance_id": producer_instance,
            "producer_stage_nonce": producer_nonce,
            "artifact_sha256": dict(QWEN3_DENSE_ARTIFACT_SHA256),
            "previous_observation_sha256": start["observation_sha256"],
        }
    )
    dense_binding = {
        "controller_run_nonce": controller.run_nonce,
        "producer_process_instance_id": producer_instance,
        "producer_stage_nonce": producer_nonce,
        "start": start,
        "end": end,
    }
    test_root = Path(os.environ["GPTQMODEL_QVQ_TEST_INPUT_ROOT"]).resolve()
    producer_datasets = controller_module.controller_dataset_evidence({
        "--calibration-dataset": str(test_root / "calibration.jsonl"),
        "--yaqa-dataset": str(test_root / "yaqa_tuning.jsonl"),
        "--validation-dataset": str(test_root / "validation.jsonl"),
    })
    measurements = (
        {"dense_source_binding": dense_binding, "pre_save": {"dense_source_end_sha256": end["observation_sha256"], "payload": payload_hashes},
         "quantize_config": {"bits": 2, "format": "qvq_v2b2_p32", "group_size": -1, "rounding": "yaqa", "sym": True, "pack_dtype": "int32", "bank_count": 2},
         "quant_config_authority_sha256": acceptance._sha256_file(Path.cwd() / "configs" / "qwen3_8b_qvq_w2_acceptance.json"),
         "layer_scope": "all", "datasets": producer_datasets},
        {"payload": payload_hashes},
        {"payload": payload_hashes},
    )
    instances = (producer_instance, reload_instance, evaluation_instance)
    nonces = (producer_nonce, reload_nonce, evaluation_nonce)
    stages = ("quantization_producer", "fresh_process_reload", "acceptance_evaluation")
    producer_datasets = measurements[0]["datasets"]
    records = []
    for index, (stage, instance, nonce, measurement) in enumerate(zip(stages, instances, nonces, measurements)):
        python = controller_module.trusted_python_executable()
        accept_script = str(Path.cwd() / "scripts" / "accept_qwen3_8b_qvq.py")
        if index == 0:
            command = [
                python, str(Path.cwd() / "scripts" / "qvq_quantize.py"),
                "--model", "/monster/data/model/Qwen3-8B", "--output", str(test_root / "checkpoint"),
                "--quant-config", str(Path.cwd() / "configs" / "qwen3_8b_qvq_w2_acceptance.json"),
                "--calibration-dataset", str(test_root / "calibration.jsonl"), "--calibration-rows", "512",
                "--yaqa-dataset", str(test_root / "yaqa_tuning.jsonl"), "--yaqa-rows", "512",
                "--validation-dataset", str(test_root / "validation.jsonl"), "--validation-rows", "512",
                "--device", "cuda:0", "--verify-qwen3-acceptance-payload-parity",
            ]
        elif index == 1:
            command = [python, accept_script, "payload-hashes", "--checkpoint", str(test_root / "checkpoint"),
                       "--device", "cuda:0", "--output", str(test_root / "reload.json")]
        else:
            command = [
                python, accept_script, "evaluate", "--dense-model", "/monster/data/model/Qwen3-8B",
                "--revision", QWEN3_PINNED_REVISION, "--checkpoint", str(test_root / "checkpoint"),
                "--manifest-dir", str(test_root), "--validation-jsonl", str(test_root / "validation.jsonl"),
                "--held-out-diagnostics-jsonl", str(test_root / "held_out_diagnostics.jsonl"),
                "--diverse-jsonl", str(test_root / "diverse_32.jsonl"), "--device", "cuda:0",
                "--maximum-bpw", "2.1", "--score-min", "0.85", "--final-kl-max-nats", "0.1",
                "--output", str(test_root / "draft.json"),
            ]
        execution_command = [command[0], f"/proc/self/fd/{50 + index}", *command[2:]]
        python_identity = os.stat(command[0], follow_symlinks=False)
        event = {
            "run_nonce": controller.run_nonce,
            "controller_instance_id": controller.controller_instance_id,
            "controller_pid": controller.controller_pid,
            "stage": stage,
            "stage_nonce": nonce,
            "process_instance_id": instance,
            "measurement": measurement,
        }
        record = {
            "stage": stage,
            "stage_nonce": nonce,
            "process_instance_id": instance,
            "pid": (700, 700, 701)[index],
            "parent_pid": controller.controller_pid,
            "os_process": {
                "pid": (700, 700, 701)[index], "ppid": controller.controller_pid,
                "start_time_ticks": 1000 + index,
                "executable": controller_module.trusted_python_executable(),
                "executable_sha256": controller_module.load_trust_config()["python_executable_sha256"],
                "executable_device": python_identity.st_dev, "executable_inode": python_identity.st_ino,
                "cmdline_sha256": hashlib.sha256(
                    b"\0".join(item.encode() for item in execution_command) + b"\0"
                ).hexdigest(),
            },
            "argv": command,
            "argv_sha256": hashlib.sha256(b"\0".join(item.encode() for item in command)).hexdigest(),
            "execution_argv": execution_command,
            "execution_argv_sha256": hashlib.sha256(
                b"\0".join(item.encode() for item in execution_command)
            ).hexdigest(),
            "spawned_monotonic_ns": index * 10 + 1,
            "event_received_monotonic_ns": index * 10 + 2,
            "acknowledgement_sent_monotonic_ns": index * 10 + 4,
            "exited_monotonic_ns": index * 10 + 5,
            "exit_code": 0,
            "pidfd_bound_through_wait": True,
            "event": event,
            "event_sha256": hashlib.sha256(acceptance.canonical_content(event)).hexdigest(),
            "previous_record_sha256": records[-1]["record_sha256"] if records else None,
            "controller_datasets": producer_datasets if index == 0 else None,
            "controller_live_payload": payload_hashes if index == 0 else None,
        }
        record["acknowledgement"] = {
            "acknowledged_event_sha256": record["event_sha256"],
            **(
                {"validation_policy": "qwen3-producer-pre-save-v1", "validated_before_save_monotonic_ns": 3}
                if index == 0 else {}
            ),
        }
        record["record_sha256"] = hashlib.sha256(acceptance.canonical_content(record)).hexdigest()
        records.append(record)
    controller._records = records
    transcript = controller.signed_transcript()
    pre_save = {**measurements[0]["pre_save"], "stage": stages[0], "stage_nonce": nonces[0], "process_instance_id": instances[0]}
    fresh_process = {**measurements[1], "stage": stages[1], "stage_nonce": nonces[1], "process_instance_id": instances[1]}
    evaluation_reload = {**measurements[2], "stage": stages[2], "stage_nonce": nonces[2], "process_instance_id": instances[2]}
    report = {
        "schema_version": 4,
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
                "controller_run_nonce": controller.run_nonce,
                "pre_save": pre_save,
                "fresh_process": fresh_process,
                "evaluation_reload": evaluation_reload,
            },
            "dense_source_binding": {
                **dense_binding,
            },
            "acceptance_controller": transcript,
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
            "final_kl_max_nats": 0.1,
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
    claim = {key: report[key] for key in ("accounting", "manifests", "thresholds", "global", "cells")}
    records[2]["event"]["measurement"]["acceptance_evidence_sha256"] = hashlib.sha256(
        acceptance.canonical_content(claim)
    ).hexdigest()
    records[2]["event_sha256"] = hashlib.sha256(
        acceptance.canonical_content(records[2]["event"])
    ).hexdigest()
    records[2]["acknowledgement"]["acknowledged_event_sha256"] = records[2]["event_sha256"]
    unsigned_record = dict(records[2])
    unsigned_record.pop("record_sha256")
    records[2]["record_sha256"] = hashlib.sha256(acceptance.canonical_content(unsigned_record)).hexdigest()
    report["artifact"]["acceptance_controller"] = controller.signed_transcript()
    report["_test_controller_authority"] = controller_authority_receipt(
        report["artifact"]["acceptance_controller"]
    )
    return report


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
    with pytest.raises(AcceptanceError, match="controller-bound"):
        validate_acceptance_report(report)

    report = _complete_report()
    report["artifact"]["packed_payload_parity"]["fresh_process"]["payload"]["aggregate_sha256"] = "0" * 64
    with pytest.raises(AcceptanceError, match="authority|controller-bound|signature|noncanonical"):
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


def test_controller_provenance_accepts_pid_reuse_with_distinct_process_instances():
    report = _complete_report()
    processes = report["artifact"]["acceptance_controller"]["processes"]
    assert processes[0]["pid"] == processes[1]["pid"]
    assert processes[0]["process_instance_id"] != processes[1]["process_instance_id"]
    validate_acceptance_report(report)


def test_controller_observes_process_facts_from_linux_proc():
    observed = controller_module._observe_linux_process(os.getpid())
    assert observed["pid"] == os.getpid()
    assert observed["ppid"] == os.getppid()
    assert observed["start_time_ticks"] > 0
    assert observed["executable_sha256"] == acceptance_script._sha256_file(Path(observed["executable"]))


def test_proc_stat_parser_handles_spaces_and_parentheses_in_comm():
    remainder = ["S", "42", *(["0"] * 17), "999"]
    pid, comm, ppid, start = controller_module._parse_linux_proc_stat(
        f"123 (evil ) name (x)) {' '.join(remainder)}"
    )
    assert (pid, comm, ppid, start) == (123, "evil ) name (x)", 42, 999)


def test_controller_issues_unpredictable_run_stage_and_process_instance_identities(tmp_path):
    first = AcceptanceController()
    second = AcceptanceController()
    assert first.run_nonce != second.run_nonce
    assert first.controller_instance_id != second.controller_instance_id
    assert len({first.run_nonce, second.run_nonce, first.controller_instance_id, second.controller_instance_id}) == 4


def test_controller_fails_closed_without_or_with_mismatched_trust_root(tmp_path, monkeypatch):
    monkeypatch.delenv(controller_module.VERIFIER_PRIVATE_KEY_ENV)
    with pytest.raises(RuntimeError, match="signing authority is absent"):
        AcceptanceController()
    attacker = tmp_path / "attacker.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(attacker)], check=True)
    attacker.chmod(0o600)
    monkeypatch.setenv(controller_module.VERIFIER_PRIVATE_KEY_ENV, str(attacker))
    with pytest.raises(RuntimeError, match="does not match the pinned public trust root"):
        AcceptanceController()


@pytest.mark.parametrize("mode", [0o700, 0o644])
def test_controller_rejects_nonexact_private_key_modes(mode):
    private = Path(os.environ[controller_module.VERIFIER_PRIVATE_KEY_ENV])
    private.chmod(mode)
    with pytest.raises(RuntimeError, match="mode must be exactly 0600"):
        AcceptanceController()


def test_controller_rejects_private_key_symlink_and_path_replacement(tmp_path, monkeypatch):
    private = Path(os.environ[controller_module.VERIFIER_PRIVATE_KEY_ENV])
    link = tmp_path / "private-link.pem"
    link.symlink_to(private)
    monkeypatch.setenv(controller_module.VERIFIER_PRIVATE_KEY_ENV, str(link))
    with pytest.raises(RuntimeError, match="opened safely|traverses a symlink"):
        AcceptanceController()
    monkeypatch.setenv(controller_module.VERIFIER_PRIVATE_KEY_ENV, str(private))
    controller = AcceptanceController()
    replacement = tmp_path / "replacement.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(replacement)], check=True)
    replacement.chmod(0o600)
    private.unlink()
    replacement.rename(private)
    with pytest.raises(RuntimeError, match="path changed"):
        controller.signed_transcript(require_complete=False)


def test_controller_rejects_same_inode_private_key_mutation():
    private = Path(os.environ[controller_module.VERIFIER_PRIVATE_KEY_ENV])
    controller = AcceptanceController()
    replacement = subprocess.check_output(["openssl", "genpkey", "-algorithm", "ED25519"])
    private.write_bytes(replacement)
    private.chmod(0o600)
    with pytest.raises(RuntimeError, match="path changed"):
        controller.signed_transcript(require_complete=False)


@pytest.mark.parametrize("target", ["trust", "private", "script"])
def test_controller_rejects_parent_directory_symlink_traversal(tmp_path, monkeypatch, target):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    if target == "trust":
        original = Path(os.environ[controller_module.TRUST_CONFIG_ENV])
        copied = actual / "trust.json"
        copied.write_bytes(original.read_bytes())
        copied.chmod(0o600)
        monkeypatch.setenv(controller_module.TRUST_CONFIG_ENV, str(link / "trust.json"))
    elif target == "private":
        original = Path(os.environ[controller_module.VERIFIER_PRIVATE_KEY_ENV])
        copied = actual / "private.pem"
        copied.write_bytes(original.read_bytes())
        copied.chmod(0o600)
        monkeypatch.setenv(controller_module.VERIFIER_PRIVATE_KEY_ENV, str(link / "private.pem"))
    else:
        trust_path = Path(os.environ[controller_module.TRUST_CONFIG_ENV])
        trust = json.loads(trust_path.read_text())
        copied = actual / "qvq_quantize.py"
        source = Path(trust["stage_scripts"]["quantization_producer"]["path"])
        copied.write_bytes(source.read_bytes())
        copied.chmod(0o755)
        trust["stage_scripts"]["quantization_producer"] = {
            "path": str(link / "qvq_quantize.py"), "sha256": acceptance_script._sha256_file(copied)
        }
        trust_path.write_text(json.dumps(trust), encoding="utf-8")
        trust_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="traverses a symlink"):
        AcceptanceController()


def test_external_trust_fingerprint_mismatch_and_repo_key_replacement(tmp_path, monkeypatch):
    trust_path = Path(os.environ[controller_module.TRUST_CONFIG_ENV])
    trust = json.loads(trust_path.read_text())
    trust["verifier_public_key_sha256"] = "0" * 64
    trust_path.write_text(json.dumps(trust), encoding="utf-8")
    trust_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="identity is absent or mismatched"):
        AcceptanceController()
    repo_key = Path.cwd() / "configs" / "attacker-verifier-public.pem"
    repo_key.write_text(Path(trust["verifier_public_key"]).read_text(), encoding="ascii")
    try:
        trust["verifier_public_key"] = str(repo_key)
        trust["verifier_public_key_sha256"] = acceptance_script._sha256_file(repo_key)
        trust_path.write_text(json.dumps(trust), encoding="utf-8")
        trust_path.chmod(0o600)
        with pytest.raises(RuntimeError, match="outside the candidate repository"):
            AcceptanceController()
    finally:
        repo_key.unlink()


def test_controller_production_api_rejects_command_substitution():
    controller = AcceptanceController()
    with pytest.raises(ValueError, match="trusted interpreter|required executable"):
        controller.spawn_stage(
            "fresh_process_reload", [sys.executable, "-c", "raise SystemExit(0)"], cwd=Path.cwd()
        )


@pytest.mark.parametrize("attack", ["argv0", "extra_flag", "duplicate_flag"])
def test_controller_rejects_resigned_noncanonical_producer_argv(attack):
    report = _complete_report()
    record = report["artifact"]["acceptance_controller"]["processes"][0]
    if attack == "argv0":
        record["argv"][0] = "/bin/true"
        record["os_process"]["executable"] = "/bin/true"
        record["os_process"]["executable_sha256"] = acceptance_script._sha256_file(Path("/bin/true"))
    elif attack == "extra_flag":
        record["argv"].append("--attacker-extra")
    else:
        record["argv"][4:4] = ["--model", "/monster/data/model/Qwen3-8B"]
    record["argv_sha256"] = hashlib.sha256(b"\0".join(item.encode() for item in record["argv"])).hexdigest()
    record["os_process"]["cmdline_sha256"] = hashlib.sha256(
        b"\0".join(item.encode() for item in record["argv"]) + b"\0"
    ).hexdigest()
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError, match="canonical required command"):
        validate_acceptance_report(report)


def test_controller_rejects_well_formed_fake_manifest_hashes():
    report = _complete_report()
    transcript = report["artifact"]["acceptance_controller"]
    fake = json.loads(json.dumps(transcript["controller_datasets"]))
    fake["calibration"]["content_sha256"] = "9" * 64
    transcript["controller_datasets"] = fake
    transcript["processes"][0]["controller_datasets"] = fake
    transcript["processes"][0]["event"]["measurement"]["datasets"] = fake
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError, match="manifest observations mismatch"):
        validate_acceptance_report(report)


def test_controller_dataset_snapshot_survives_verify_use_mutation():
    root = Path(os.environ["GPTQMODEL_QVQ_TEST_INPUT_ROOT"])
    values = {
        "--calibration-dataset": str(root / "calibration.jsonl"),
        "--yaqa-dataset": str(root / "yaqa_tuning.jsonl"),
        "--validation-dataset": str(root / "validation.jsonl"),
    }
    _evidence, snapshots = controller_module._controller_dataset_bundle(values)
    mapping, descriptors = controller_module._sealed_snapshot_descriptors(snapshots)
    try:
        source = root / "calibration.jsonl"
        original = snapshots[str(source)]
        source.write_text('{"identity":"attacker","content":"changed"}\n', encoding="utf-8")
        descriptor = mapping[str(source)]
        assert os.read(descriptor, len(original) + 1) == original
        with pytest.raises(OSError):
            os.write(descriptor, b"attacker")
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_descriptor_owned_file_survives_path_replacement(tmp_path):
    path = tmp_path / "trusted.bin"
    path.write_bytes(b"reviewed")
    trusted = controller_module._open_trusted_file(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"attacker")
    replacement.replace(path)
    try:
        os.lseek(trusted.descriptor, 0, os.SEEK_SET)
        assert os.read(trusted.descriptor, 32) == b"reviewed"
        assert trusted.content == b"reviewed"
    finally:
        trusted.close()


def test_verified_policy_and_stage_are_consumed_from_retained_descriptors(tmp_path):
    trust_path = Path(os.environ[controller_module.TRUST_CONFIG_ENV])
    trust = json.loads(trust_path.read_text())
    script = tmp_path / "producer.py"
    script.write_text("print('descriptor-owned')\n", encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(Path(trust["acceptance_policy"]).read_bytes())
    trust["acceptance_policy"] = str(policy_path)
    trust["acceptance_policy_sha256"] = acceptance_script._sha256_file(policy_path)
    quant_path = tmp_path / "quant.json"
    quant_path.write_bytes(Path(trust["quant_config"]).read_bytes())
    trust["quant_config"] = str(quant_path)
    trust["quant_config_sha256"] = acceptance_script._sha256_file(quant_path)
    trust["stage_scripts"]["quantization_producer"] = {
        "path": str(script), "sha256": acceptance_script._sha256_file(script)
    }
    trust_path.write_text(json.dumps(trust), encoding="utf-8")
    trust_path.chmod(0o600)
    resources = controller_module._TrustedResources()
    policy_before = controller_module._acceptance_policy(resources)
    for path, content in (
        (script, b"raise SystemExit('attacker')\n"),
        (policy_path, b'{"schema":"attacker"}'),
        (quant_path, b'{"format":"attacker"}'),
    ):
        replacement = path.with_suffix(path.suffix + ".replacement")
        replacement.write_bytes(content)
        replacement.replace(path)
    try:
        assert controller_module._acceptance_policy(resources) == policy_before
        assert json.loads(resources.files["quant_config"].content)["format"] == "qvq_v2b2_p32"
        python_file = resources.files["python_executable"]
        script_file = resources.files["quantization_producer"]
        result = subprocess.run(
            [trust["python_executable"], script_file.fd_path],
            executable=python_file.fd_path,
            pass_fds=(python_file.descriptor, script_file.descriptor),
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == "descriptor-owned"
    finally:
        resources.close()


def test_controller_rejects_rename_capable_trusted_parent(tmp_path):
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    policy = unsafe / "policy.json"
    original = Path.cwd() / "configs" / "qwen3_8b_qvq_acceptance_policy.json"
    policy.write_bytes(original.read_bytes())
    trust_path = Path(os.environ[controller_module.TRUST_CONFIG_ENV])
    trust = json.loads(trust_path.read_text())
    trust["acceptance_policy"] = str(policy)
    trust["acceptance_policy_sha256"] = acceptance_script._sha256_file(policy)
    trust_path.write_text(json.dumps(trust), encoding="utf-8")
    trust_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="unsafe parent directory"):
        AcceptanceController()


def test_controller_rejects_dataset_dotdot_alias():
    report = _complete_report()
    command = report["artifact"]["acceptance_controller"]["processes"][0]["argv"]
    index = command.index("--calibration-dataset") + 1
    source = Path(command[index])
    command[index] = str(source.parent / "alias" / ".." / source.name)
    with pytest.raises(ValueError, match="normalized path"):
        controller_module.validate_required_command("quantization_producer", command)


def test_live_producer_ipc_keeps_reader_open_until_validation_and_ack(monkeypatch):
    module_names = [f"module-{index:03d}" for index in range(252)]
    cells = [(index, "role") for index in range(252)]
    monkeypatch.setattr(acceptance, "expected_cells", lambda: cells)
    monkeypatch.setattr(acceptance, "expected_projection_name", lambda layer, _role: module_names[layer])
    monkeypatch.setattr(
        acceptance,
        "canonical_packed_tensor_schema",
        lambda layer, _role: {
            f"{module_names[layer]}.{name}": {"dtype": "U8", "shape": [1]}
            for name in ("SU", "SV", "bank_alt_id", "bank_ids", "trellis")
        },
    )
    parent, child = socket.socketpair()
    prefix = "GPTQMODEL_QVQ_CONTROLLER_"
    authority = {
        "RUN_NONCE": secrets.token_hex(32), "CONTROLLER_INSTANCE_ID": secrets.token_hex(32),
        "CONTROLLER_PID": str(os.getpid()), "STAGE": "quantization_producer",
        "STAGE_NONCE": secrets.token_hex(32), "PROCESS_INSTANCE_ID": secrets.token_hex(32),
        "EVENT_FD": str(child.detach()),
    }
    for key, value in authority.items():
        monkeypatch.setenv(prefix + key, value)
    saved = threading.Event()
    error = []

    def producer():
        try:
            records = (
                ({"module": module, "tensor": tensor, "dtype": "torch.uint8", "shape": [1], "byte_count": 1}, b"x")
                for module in module_names
                for tensor in ("SU", "SV", "bank_alt_id", "bank_ids", "trellis")
            )
            controller_module.emit_controller_measurement(
                "quantization_producer", {"live": True}, live_payload_records=records
            )
            saved.set()
        except (OSError, RuntimeError, ValueError) as caught:
            error.append(caught)

    worker = threading.Thread(target=producer)
    worker.start()
    with parent.makefile("rb") as reader:
        event = json.loads(reader.readline())
        assert not saved.is_set()
        live = controller_module._receive_live_payload(reader)
        assert live["module_count"] == 252
        parent.sendall(controller_module._canonical({
            "acknowledged_event_sha256": controller_module._sha256(event),
            "validation_policy": "qwen3-producer-pre-save-v1",
        }) + b"\n")
    worker.join(timeout=5)
    parent.close()
    assert not error
    assert saved.is_set()


def test_live_payload_rejects_extra_tensor(monkeypatch):
    import io

    monkeypatch.setattr(acceptance, "expected_cells", lambda: [(0, "role")])
    monkeypatch.setattr(acceptance, "expected_projection_name", lambda _layer, _role: "module")
    monkeypatch.setattr(
        acceptance,
        "canonical_packed_tensor_schema",
        lambda _layer, _role: {
            f"module.{name}": {"dtype": "U8", "shape": [1]}
            for name in ("SU", "SV", "bank_alt_id", "bank_ids", "trellis")
        },
    )
    stream = bytearray()
    for name in ("SU", "SV", "bank_alt_id", "bank_ids", "trellis", "unexpected"):
        header = controller_module._canonical(
            {"module": "module", "tensor": name, "dtype": "torch.uint8", "shape": [1], "byte_count": 1}
        )
        stream.extend(struct.pack("<Q", len(header)) + header + b"x")
    stream.extend(struct.pack("<Q", 0))
    with pytest.raises(RuntimeError, match="extra tensor"):
        controller_module._receive_live_payload(io.BytesIO(stream))


def test_process_observation_rejects_mixed_start_identity(monkeypatch):
    original = Path.read_text
    calls = 0

    def changing_stat(path, *args, **kwargs):
        nonlocal calls
        value = original(path, *args, **kwargs)
        if path.name == "stat":
            calls += 1
            if calls == 2:
                opening, closing = value.find("("), value.rfind(")")
                fields = value[closing + 2:].split()
                fields[19] = str(int(fields[19]) + 1)
                return f"{value[:opening]}({value[opening + 1:closing]}) {' '.join(fields)}"
        return value

    monkeypatch.setattr(Path, "read_text", changing_stat)
    with pytest.raises(RuntimeError, match="mixed or reused"):
        controller_module._observe_linux_process(os.getpid())


def test_controller_rejects_fabricated_252_digest_payload_without_live_bytes():
    report = _complete_report()
    transcript = report["artifact"]["acceptance_controller"]
    fabricated = {
        "scheme": acceptance.QVQ_PAYLOAD_HASH_SCHEME,
        "module_count": 252,
        "module_sha256": {name: "7" * 64 for name in expected_projection_names()},
        "module_tensor_counts": {name: 5 for name in expected_projection_names()},
    }
    fabricated["aggregate_sha256"] = acceptance.qwen3_payload_aggregate(
        fabricated["module_sha256"], fabricated["module_tensor_counts"]
    )
    transcript["processes"][0]["event"]["measurement"]["pre_save"]["payload"] = fabricated
    transcript["processes"][1]["event"]["measurement"]["payload"] = fabricated
    transcript["processes"][2]["event"]["measurement"]["payload"] = fabricated
    report["artifact"]["packed_payload_parity"]["pre_save"]["payload"] = fabricated
    report["artifact"]["packed_payload_parity"]["fresh_process"]["payload"] = fabricated
    report["artifact"]["packed_payload_parity"]["evaluation_reload"]["payload"] = fabricated
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError, match="controller-hashed live tensor bytes"):
        validate_acceptance_report(report)


@pytest.mark.parametrize("attack", ["revision", "threshold", "checkpoint_path"])
def test_controller_rejects_resigned_semantic_or_cross_stage_substitution(attack):
    report = _complete_report()
    records = report["artifact"]["acceptance_controller"]["processes"]
    record = records[2]
    flag = {"revision": "--revision", "threshold": "--maximum-bpw", "checkpoint_path": "--checkpoint"}[attack]
    value = {"revision": "attacker-revision", "threshold": "2.09", "checkpoint_path": "/tmp/other-checkpoint"}[attack]
    record["argv"][record["argv"].index(flag) + 1] = value
    record["argv_sha256"] = hashlib.sha256(b"\0".join(item.encode() for item in record["argv"])).hexdigest()
    record["os_process"]["cmdline_sha256"] = hashlib.sha256(
        b"\0".join(item.encode() for item in record["argv"]) + b"\0"
    ).hexdigest()
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError, match="canonical required command|split across"):
        validate_acceptance_report(report)


@pytest.mark.parametrize("field", ["aggregate", "module"])
def test_controller_rejects_internally_inconsistent_payload_aggregate(field):
    report = _complete_report()
    payload = report["artifact"]["packed_payload_parity"]["pre_save"]["payload"]
    if field == "aggregate":
        payload["aggregate_sha256"] = "0" * 64
    else:
        payload["module_sha256"][expected_projection_names()[0]] = "0" * 64
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError, match="noncanonical hash scheme"):
        validate_acceptance_report(report)


def test_gate_rejects_complete_attacker_key_final_checkpoint_fabrication(tmp_path):
    report = _complete_report()
    transcript = dict(report["artifact"]["acceptance_controller"])
    transcript.pop("controller_signature_ed25519")
    attacker_private = tmp_path / "attacker.pem"
    attacker_public = tmp_path / "attacker.pub"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(attacker_private)], check=True)
    subprocess.run(["openssl", "pkey", "-in", str(attacker_private), "-pubout", "-out", str(attacker_public)], check=True)
    attacker_public_bytes = attacker_public.read_bytes()
    transcript["verifier_public_key_sha256"] = hashlib.sha256(attacker_public_bytes).hexdigest()
    transcript["controller_signature_ed25519"] = controller_module._sign(
        attacker_private.read_bytes(), acceptance.canonical_content(transcript)
    )
    report["artifact"]["acceptance_controller"] = transcript
    attacker_receipt = {
        "schema": "qvq-acceptance-controller-trust-root-v4",
        "controller_instance_id": transcript["controller_instance_id"],
        "run_nonce": transcript["run_nonce"],
        "verifier_public_key_sha256": hashlib.sha256(attacker_public_bytes).hexdigest(),
        "transcript_sha256": hashlib.sha256(acceptance.canonical_content(transcript)).hexdigest(),
    }
    with pytest.raises(AcceptanceError, match="trust root|authority receipt"):
        _validate_acceptance_report(report, controller_authority=attacker_receipt)


@pytest.mark.parametrize("attack", ["command", "parent", "os_parent", "injected", "split_identity"])
def test_controller_rejects_signed_record_fact_substitution(attack):
    report = _complete_report()
    transcript = report["artifact"]["acceptance_controller"]
    records = transcript["processes"]
    if attack == "command":
        records[1]["argv"][2] = "evaluate"
        records[1]["argv_sha256"] = hashlib.sha256(
            b"\0".join(item.encode() for item in records[1]["argv"])
        ).hexdigest()
        records[1]["os_process"]["cmdline_sha256"] = hashlib.sha256(
            b"\0".join(item.encode() for item in records[1]["argv"]) + b"\0"
        ).hexdigest()
    elif attack == "parent":
        records[0]["parent_pid"] += 1
    elif attack == "os_parent":
        records[0]["os_process"]["ppid"] += 1
    elif attack == "injected":
        records.append(dict(records[-1]))
    else:
        report["artifact"]["dense_source_binding"]["producer_stage_nonce"] = records[1]["stage_nonce"]
    _resign_controller_report(report)
    with pytest.raises(AcceptanceError):
        validate_acceptance_report(report)


def test_invalid_producer_payload_is_rejected_before_acknowledgement():
    report = _complete_report()
    event = report["artifact"]["acceptance_controller"]["processes"][0]["event"]
    event["measurement"]["pre_save"]["payload"]["module_sha256"].pop(expected_projection_names()[0])
    with pytest.raises(AcceptanceError, match="pre-save controller validation"):
        acceptance.validate_producer_pre_save_measurement(
            event["measurement"], event=event, controller_datasets=event["measurement"]["datasets"]
        )


def test_controller_rejects_fabricated_final_checkpoint_only_report():
    report = _complete_report()
    with pytest.raises(AcceptanceError, match="independently supplied controller authority"):
        _validate_acceptance_report(report)
    report["artifact"].pop("acceptance_controller")
    with pytest.raises(AcceptanceError, match="controller transcript"):
        validate_acceptance_report(report)


def test_controller_rejects_arbitrary_textual_process_ids_and_nonces():
    report = _complete_report()
    process = report["artifact"]["acceptance_controller"]["processes"][1]
    process["process_instance_id"] = "pid:123"
    process["stage_nonce"] = "chosen-by-caller"
    with pytest.raises(AcceptanceError, match="authority|signature|256-bit"):
        validate_acceptance_report(report)


def test_controller_rejects_replayed_stage_evidence():
    report = _complete_report()
    processes = report["artifact"]["acceptance_controller"]["processes"]
    processes[2]["event"] = dict(processes[1]["event"])
    with pytest.raises(AcceptanceError, match="authority|signature|replayed"):
        validate_acceptance_report(report)


@pytest.mark.parametrize("missing", ["spawn", "exit"])
def test_controller_rejects_missing_spawn_or_exit_records(missing):
    report = _complete_report()
    processes = report["artifact"]["acceptance_controller"]["processes"]
    if missing == "spawn":
        processes.pop(1)
    else:
        processes[1].pop("exit_code")
    with pytest.raises(AcceptanceError, match="authority|signature|spawn/exit"):
        validate_acceptance_report(report)


def test_controller_rejects_self_authored_evaluation_observation():
    report = _complete_report()
    report["artifact"]["packed_payload_parity"]["evaluation_reload"] = {
        "stage": "acceptance_evaluation",
        "stage_nonce": secrets.token_hex(32),
        "process_instance_id": secrets.token_hex(32),
        "payload": report["artifact"]["packed_payload_parity"]["fresh_process"]["payload"],
    }
    with pytest.raises(AcceptanceError, match="self-authored|controller-bound"):
        validate_acceptance_report(report)


@pytest.mark.parametrize(("top1", "diverse"), [(0.85, 0.1), (0.1, 0.85)])
def test_report_accepts_either_score_alternative(top1, diverse):
    report = _complete_report(score=top1, diverse_score=diverse)
    validate_acceptance_report(report)


def test_report_rejects_when_both_score_alternatives_fail():
    report = _complete_report(score=0.849, diverse_score=0.849)
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
    report["schema_version"] = 5
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


def test_controller_signature_content_binds_submitted_report():
    report = _complete_report()
    validate_acceptance_report(report)
    report["global"]["final_kl_nats"] = 0.09
    with pytest.raises(AcceptanceError, match="metrics were not emitted"):
        validate_acceptance_report(report)


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
