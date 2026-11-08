# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

from gptqmodel.utils.exl3_projection_checkpoint import (
    CHECKPOINT_CONTRACT,
    EXL3ProjectionCheckpointStore,
    build_projection_request,
    checkpoint_root_from_provenance,
)


def _request(scale: float = 1.0) -> dict:
    return build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=torch.arange(32, dtype=torch.float32).reshape(8, 4) * scale,
        hessian=torch.eye(8, dtype=torch.float32),
        sample_count=1024,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
        },
        family_join={"source_revision": "abc", "corpus_sha256": "def"},
        route_evidence={"expert_route_count": 1024},
    )


def _inline_request(
    role: str,
    *,
    base_bits: int = 2,
    candidate_request_sha256: str | None = None,
) -> dict:
    upgrade_bits = base_bits + 1
    inline = {
        "role": role,
        "policy_sha256": "a" * 64,
        "base_bits": base_bits,
        "upgrade_bits": upgrade_bits,
    }
    bits = base_bits
    if role == f"selected_k{upgrade_bits}":
        bits = upgrade_bits
        inline.update(
            {
                "candidate_request_sha256": candidate_request_sha256,
                "tier_plan_sha256": "b" * 64,
            }
        )
    return build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=torch.arange(32, dtype=torch.float32).reshape(8, 4),
        hessian=torch.eye(8, dtype=torch.float32),
        sample_count=1024,
        quantizer_contract={
            "bits": bits,
            "codebook": "mcg",
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
            "inline_mixed": inline,
        },
        family_join={"source_revision": "abc", "corpus_sha256": "def"},
        route_evidence={"expert_route_count": 1024},
    )


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.arange(32, dtype=torch.int16).reshape(1, 1, 32),
        "suh": torch.ones(16, dtype=torch.float16),
        "svh": torch.ones(16, dtype=torch.float16),
        "mcg": torch.tensor([-877912083], dtype=torch.int32),
    }


def _result() -> dict:
    return {
        "duration_seconds": 12.5,
        "proxy_error": 0.125,
        "device_names": ["cuda:0"],
        "quantizer_metrics": {"reported_metric_kind": "test"},
        "ledger_record": {"record_kind": "projection", "module": "test"},
    }


def test_projection_checkpoint_round_trips_and_is_idempotent(tmp_path) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    tensors = _tensors()
    result = _result()

    assert store.load(request) is None
    assert store.commit(request, tensors, result) == result
    first_files = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }
    assert store.commit(request, tensors, result) == result
    second_files = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files
    assert {
        path.stat().st_mode & 0o777
        for path in store.root.rglob("*")
        if path.is_file()
    } == {0o644}

    loaded_tensors, loaded_result = store.load(request)
    assert loaded_result == result
    assert set(loaded_tensors) == set(tensors)
    assert all(
        torch.equal(loaded_tensors[name], tensor) for name, tensor in tensors.items()
    )
    stored_request, committed_tensors, committed_result = store.load_committed(
        request["request_sha256"]
    )
    assert stored_request == request
    assert committed_result == result
    assert all(
        torch.equal(committed_tensors[name], tensor)
        for name, tensor in tensors.items()
    )
    assert store.load(_request(scale=2.0)) is None


def test_projection_checkpoint_module_request_reservation_rejects_drift(
    tmp_path,
) -> None:
    root = tmp_path / "checkpoints"
    original = _request()
    store = EXL3ProjectionCheckpointStore(root)
    store.commit(original, _tensors(), _result())

    resumed_store = EXL3ProjectionCheckpointStore(root)
    resumed_store.reserve_module_request(original)
    resumed_store.reserve_module_request(original)
    with pytest.raises(ValueError, match="immutable module request drift"):
        resumed_store.reserve_module_request(_request(scale=2.0))


def test_projection_checkpoint_module_request_reservation_rejects_existing_drift(
    tmp_path,
) -> None:
    root = tmp_path / "checkpoints"
    original = _request()
    drifted = _request(scale=2.0)
    EXL3ProjectionCheckpointStore(root).commit(original, _tensors(), _result())
    # Direct stores are also used as bounded remote scratch, so committing a
    # second request remains possible there.  A coordinator reservation must
    # reject that polluted state before any new quantization begins.
    EXL3ProjectionCheckpointStore(root).commit(
        drifted, _tensors(), {**_result(), "drifted": True}
    )

    with pytest.raises(ValueError, match="contains immutable module request drift"):
        EXL3ProjectionCheckpointStore(root).reserve_module_request(original)


def test_projection_checkpoint_reserves_inline_k2_candidate_and_selected_k3(
    tmp_path,
) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    candidate = _inline_request("candidate_k2")
    selected = _inline_request(
        "selected_k3", candidate_request_sha256=candidate["request_sha256"]
    )

    store.reserve_module_request(candidate)
    store.reserve_module_request(selected)
    store.reserve_module_request(candidate)
    store.reserve_module_request(selected)

    with pytest.raises(ValueError, match="uniform and inline"):
        store.reserve_module_request(_request())


def test_projection_checkpoint_reserves_inline_k3_candidate_and_selected_k4(
    tmp_path,
) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    candidate = _inline_request("candidate_k3", base_bits=3)
    selected = _inline_request(
        "selected_k4",
        base_bits=3,
        candidate_request_sha256=candidate["request_sha256"],
    )

    store.reserve_module_request(candidate)
    store.reserve_module_request(selected)
    store.reserve_module_request(candidate)
    store.reserve_module_request(selected)


def test_projection_checkpoint_load_committed_rejects_stored_request_tampering(
    tmp_path,
) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    store.commit(request, _tensors(), _result())
    manifest_path = next(store.root.rglob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["request"]["sample_count"] += 1
    body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    from gptqmodel.utils.exl3_projection_checkpoint import (
        canonical_json_bytes,
        sha256_bytes,
    )

    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(body))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content validation"):
        store.load_committed(request["request_sha256"])


def test_projection_checkpoint_rejects_corrupt_payload_and_manifest(tmp_path) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    store.commit(request, _tensors(), _result())
    manifest_path = next(store.root.rglob("*.json"))
    tensor_path = next(store.root.rglob("*.safetensors"))

    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="content validation"):
        store.load(request)

    store = EXL3ProjectionCheckpointStore(tmp_path / "other")
    store.commit(request, _tensors(), _result())
    manifest_path = next(store.root.rglob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["result"]["proxy_error"] = 9.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content validation"):
        store.load(request)


def test_projection_checkpoint_contract_is_explicit_in_run_provenance(tmp_path) -> None:
    root = tmp_path / "checkpoint-root"
    provenance = {
        "run": {
            "projection_checkpoint": {
                "contract": CHECKPOINT_CONTRACT,
                "root": str(root),
            }
        }
    }
    assert checkpoint_root_from_provenance(provenance) == root.resolve()

    provenance["run"]["projection_checkpoint"]["contract"] = "wrong"
    with pytest.raises(ValueError, match="run contract"):
        checkpoint_root_from_provenance(provenance)


def test_projection_checkpoint_prune_bounds_remote_scratch_and_preserves_retry(
    tmp_path,
) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    requests = [_request(scale=float(index + 1)) for index in range(3)]
    for index, request in enumerate(requests):
        store.commit(request, _tensors(), {**_result(), "sequence": index})
        manifest_path, tensor_path = store._paths(request["request_sha256"])
        timestamp = 1_000_000_000 + index
        os.utime(manifest_path, ns=(timestamp, timestamp))
        os.utime(tensor_path, ns=(timestamp, timestamp))

    summary = store.prune(
        max_entries=2,
        preserve_request_sha256=(requests[0]["request_sha256"],),
    )
    assert summary["retained_entries"] == 2
    assert summary["removed_entries"] == 1
    assert summary["removed_bytes"] > 0
    assert summary["removed_orphans"] == 0
    assert store.load(requests[0]) is not None
    assert store.load(requests[1]) is None
    assert store.load(requests[2]) is not None
    assert len(list(store.root.rglob("*.json"))) == 2
    assert len(list(store.root.rglob("*.safetensors"))) == 2


def test_projection_checkpoint_prune_removes_only_validated_tensor_orphans(
    tmp_path,
) -> None:
    store = EXL3ProjectionCheckpointStore(tmp_path / "checkpoints")
    request = _request()
    _manifest_path, tensor_path = store._paths(request["request_sha256"])
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"interrupted-commit")
    summary = store.prune(max_entries=2)
    assert summary == {
        "retained_entries": 0,
        "removed_entries": 0,
        "removed_bytes": len(b"interrupted-commit"),
        "removed_orphans": 1,
    }
    assert not tensor_path.exists()

    unexpected = store.root / "not-a-checkpoint"
    unexpected.write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe entry"):
        store.prune(max_entries=2)
    assert unexpected.read_text(encoding="utf-8") == "do not delete"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_packed_replay_matches_exllamav3_runtime_reconstruction() -> None:
    from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3
    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
        quantize_exl3,
        reconstruct_exl3_tensors,
    )

    torch.manual_seed(787)
    device = torch.device("cuda:0")
    weight = torch.randn((128, 128), dtype=torch.float32, device=device) * 0.02
    activations = torch.randn((1024, 128), dtype=torch.float32, device=device)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    quant_args = {
        "K": 2,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 787,
        "mcg": True,
    }
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        quant_args,
        return_weight_q=False,
    )
    replay = reconstruct_exl3_tensors(tensors, device=device, dtype=torch.float16)
    runtime = LinearEXL3(
        config=None,
        in_features=128,
        out_features=128,
        suh=tensors["suh"],
        svh=tensors["svh"],
        trellis=tensors["trellis"],
        mcg=tensors["mcg"],
        out_dtype=torch.float16,
    )
    try:
        reference = runtime.get_weight_tensor()
        assert torch.equal(replay.view(torch.int16), reference.view(torch.int16))
    finally:
        runtime.unload()
