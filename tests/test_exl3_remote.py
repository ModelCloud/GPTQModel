# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import hmac
import io
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import gptqmodel.utils.exl3_remote as exl3_remote
import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
)
from gptqmodel.utils.exl3_projection_checkpoint import (
    EXL3ProjectionCheckpointStore,
    build_projection_request,
)
from gptqmodel.utils.exl3_remote import (
    CoordinatorSlot,
    REMOTE_CONTRACT,
    REMOTE_RESULT_SCHEMA,
    REMOTE_SCHEDULER,
    EXL3RemoteClient,
    RemoteEndpoint,
    decode_tensor_envelope,
    encode_tensor_envelope,
    execute_remote_projection,
    exl3_quantization_failure_message,
    remote_client_from_provenance,
    validate_exl3_hessian_metrics,
    validate_remote_output_tensors,
)


def _endpoint(name: str) -> RemoteEndpoint:
    return RemoteEndpoint(
        name=name,
        url=f"http://{name}:17841",
        preflight_sha256=(name.encode().hex() + "0" * 64)[:64],
        image_digest="sha256:" + (name.encode().hex() + "1" * 64)[:64],
    )


def _coordinator(device: str) -> CoordinatorSlot:
    index = device.split(":", 1)[1]
    return CoordinatorSlot(
        device=device,
        gpu_uuid=f"GPU-coordinator-{index}",
        preflight_sha256="a" * 64,
        image_digest="sha256:" + "b" * 64,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(787)
    weight = torch.randn((128, 128), generator=generator, dtype=torch.float32) * 0.02
    activations = torch.randn((1024, 128), generator=generator, dtype=torch.float32)
    hessian = activations.T @ activations
    return weight, hessian


def _request(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    endpoint: RemoteEndpoint,
) -> dict:
    return build_projection_request(
        module_full_name="model.layers.7.mlp.experts.31.gate_proj",
        layer_index=7,
        input_weight=weight,
        hessian=hessian,
        sample_count=1024,
        quantizer_contract={
            "bits": 2,
            "codebook": "mcg",
            "hessian_capture": "raw-xtx-sum-fp32-v1",
            "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
            "apply_out_scales": None,
            "sigma_reg": 0.025,
            "seed": 787,
            "execution": EXL3RemoteClient.execution_contract(endpoint),
        },
        family_join={"source_revision": "test-source"},
        route_evidence=None,
    )


def _packed_tensors() -> dict[str, torch.Tensor]:
    return {
        "trellis": torch.zeros((8, 8, 32), dtype=torch.int16),
        "suh": torch.ones(128, dtype=torch.float16),
        "svh": torch.ones(128, dtype=torch.float16),
        "mcg": torch.tensor([-877912083], dtype=torch.int32),
    }


def test_quantization_failure_message_identifies_hessian_and_minor() -> None:
    hessian = torch.tensor([[2.0, 0.5], [0.5, -1.0]], dtype=torch.float32)
    message = exl3_quantization_failure_message(
        error=RuntimeError(
            "linalg.cholesky: the leading minor of order 2 is not positive-definite"
        ),
        module_full_name="model.layers.0.mlp.experts.29.down_proj",
        request_sha256="a" * 64,
        hessian=hessian,
        sample_count=1024,
        sigma_reg=0.025,
        raw_hessian=torch.eye(2, dtype=torch.float32) * 4.0,
    )

    diagnostics = json.loads(message.removeprefix("EXL3 quantization failed: "))
    assert diagnostics["module_full_name"].endswith("experts.29.down_proj")
    assert diagnostics["request_sha256"] == "a" * 64
    assert diagnostics["cholesky_leading_minor"] == 2
    assert diagnostics["hessian"] == {
        "state": "quantizer-owned-current-matrix",
        "shape": [2, 2],
        "dtype": "torch.float32",
        "sample_count": 1024,
        "sigma_reg": 0.025,
        "nonfinite_count": 0,
        "symmetry_max_abs": 0.0,
        "diagonal_min": -1.0,
        "diagonal_mean": 0.5,
        "diagonal_max": 2.0,
        "symmetrized_cholesky": {
            "attempted": True,
            "succeeded": False,
            "leading_minor": 2,
        },
    }
    assert diagnostics["raw_hessian"] == {
        "state": "unmodified-raw-xtx-sum",
        "shape": [2, 2],
        "dtype": "torch.float32",
        "sample_count": 1024,
        "nonfinite_count": 0,
        "symmetry_max_abs": 0.0,
        "diagonal_min": 4.0,
        "diagonal_mean": 4.0,
        "diagonal_max": 4.0,
    }


def test_quantization_failure_message_preserves_error_when_diagnostics_fail(
    monkeypatch,
) -> None:
    def fail_diagnostics(**_kwargs):
        raise KeyError("diagnostic-field")

    monkeypatch.setattr(
        exl3_remote,
        "_exl3_quantization_failure_message",
        fail_diagnostics,
    )
    message = exl3_quantization_failure_message(
        error=RuntimeError("original cholesky failure"),
        module_full_name="model.layers.0.mlp.experts.29.down_proj",
        request_sha256="b" * 64,
        hessian=torch.eye(2),
        sample_count=1024,
        sigma_reg=0.025,
    )
    diagnostics = json.loads(message.removeprefix("EXL3 quantization failed: "))
    assert diagnostics["error"] == "original cholesky failure"
    assert diagnostics["diagnostic_error_type"] == "KeyError"
    assert diagnostics["hessian"]["state"] == "diagnostic-collection-failed"


def test_tensor_envelope_is_content_bound() -> None:
    payload = encode_tensor_envelope(
        {"schema": "test", "value": 7},
        {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)},
    )
    manifest, tensors = decode_tensor_envelope(payload)
    assert manifest == {"schema": "test", "value": 7}
    assert torch.equal(
        tensors["weight"], torch.arange(16, dtype=torch.float32).reshape(4, 4)
    )

    corrupt = bytearray(payload)
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="content validation"):
        decode_tensor_envelope(bytes(corrupt))


def test_scheduler_reuses_next_free_slot_and_durably_resumes(tmp_path) -> None:
    endpoints = [_endpoint("spark-d"), _endpoint("spark-a"), _endpoint("spark-c")]
    assignment_store = tmp_path / "assignments"
    forward = EXL3RemoteClient(
        endpoints=endpoints,
        token=b"secret",
        coordinator_slots=[_coordinator("cuda:0"), _coordinator("cuda:1")],
        timeout_seconds=1,
        assignment_store_path=assignment_store,
    )
    keys = [f"model.layers.17.mlp.experts.{expert}.gate_proj" for expert in range(9)]
    leases = [forward.acquire_slot(key) for key in keys[:5]]
    assert [lease.slot for lease in leases] == [
        _coordinator("cuda:0"),
        _coordinator("cuda:1"),
        _endpoint("spark-a"),
        _endpoint("spark-c"),
        _endpoint("spark-d"),
    ]
    staged = [forward.acquire_slot(key) for key in keys[5:8]]
    assert [lease.slot for lease in staged] == [
        _endpoint("spark-a"),
        _endpoint("spark-c"),
        _endpoint("spark-d"),
    ]
    leases[0].release()
    replacement = forward.acquire_slot(keys[8])
    assert replacement.slot == _coordinator("cuda:0")
    assert replacement.new_assignment is True
    replacement.release()
    for lease in leases[1:]:
        lease.release()
    for lease in staged:
        lease.release()

    records = [json.loads(path.read_text()) for path in assignment_store.rglob("*.json")]
    assert len(records) == len(keys)
    assert {record["assignment_key"] for record in records} == set(keys)
    assert {record["scheduler"] for record in records} == {REMOTE_SCHEDULER}

    reverse = EXL3RemoteClient(
        endpoints=list(reversed(endpoints)),
        token=b"secret",
        coordinator_slots=[_coordinator("cuda:1"), _coordinator("cuda:0")],
        timeout_seconds=1,
        assignment_store_path=assignment_store,
    )
    resumed = reverse.acquire_slot(keys[0])
    assert resumed.slot == _coordinator("cuda:0")
    assert resumed.new_assignment is False
    resumed.release()

    endpoint = endpoints[0]
    assert forward.execution_contract(endpoint) == {
        "kind": "remote_worker",
        "remote_contract": REMOTE_CONTRACT,
        "name": endpoint.name,
        "preflight_sha256": endpoint.preflight_sha256,
        "image_digest": endpoint.image_digest,
    }
    local = _coordinator("cuda:1")
    assert forward.execution_contract(local) == {
        "kind": "coordinator",
        "remote_contract": REMOTE_CONTRACT,
        "device": "cuda:1",
        "gpu_uuid": "GPU-coordinator-1",
        "preflight_sha256": "a" * 64,
        "image_digest": "sha256:" + "b" * 64,
    }


def test_provenance_requires_token_and_preserves_retry_contract(
    tmp_path, monkeypatch
) -> None:
    endpoint = _endpoint("spark-a")
    provenance = {
        "run": {
            "remote_workers": {
                "contract": REMOTE_CONTRACT,
                "scheduler": REMOTE_SCHEDULER,
                "token_env": "TEST_EXL3_TOKEN",
                "assignment_store": str(tmp_path / "assignments.json"),
                "coordinator_slots": [
                    {
                        "device": "cuda:0",
                        "gpu_uuid": "GPU-coordinator-0",
                        "preflight_sha256": "a" * 64,
                        "image_digest": "sha256:" + "b" * 64,
                    }
                ],
                "timeout_seconds": 42,
                "max_attempts": 3,
                "orchestration_workers": 4,
                "endpoints": [endpoint.__dict__],
            }
        }
    }
    monkeypatch.delenv("TEST_EXL3_TOKEN", raising=False)
    with pytest.raises(ValueError, match="is unset"):
        remote_client_from_provenance(provenance)
    monkeypatch.setenv("TEST_EXL3_TOKEN", "secret")
    client = remote_client_from_provenance(provenance)
    assert client is not None
    assert client.max_attempts == 3
    assert client.timeout_seconds == 42

    invalid = json.loads(json.dumps(provenance))
    invalid["run"]["remote_workers"]["max_attempts"] = 0
    with pytest.raises(ValueError, match="run contract"):
        remote_client_from_provenance(invalid)


def test_output_geometry_is_bound_to_request() -> None:
    endpoint = _endpoint("spark-a")
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    tensors = _packed_tensors()
    validate_remote_output_tensors(request, tensors)
    tensors["trellis"] = torch.zeros((8, 8, 16), dtype=torch.int16)
    with pytest.raises(ValueError, match="geometry"):
        validate_remote_output_tensors(request, tensors)


def test_hessian_metrics_bind_fp64_congruence_contract() -> None:
    metrics = {
        "quantizer_path": "hessian_ldlq",
        "hessian_metric_status": "ok",
        "hessian_sample_count": 2821,
        "hessian_regularization_sigma": 0.025,
        "hessian_numerical_contract": EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.000180678,
        "hessian_symmetry_restoration": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "hessian_symmetry_correction_max_abs": 1e-12,
        "selected_global_scale": 0.9926824431197947,
        "scale_search_mse": 0.10599474608898163,
    }
    validate_exl3_hessian_metrics(
        metrics,
        sample_count=2821,
        sigma_reg=0.025,
    )

    metrics["hessian_numerical_contract"] = "legacy-fp32"
    with pytest.raises(RuntimeError, match="numerical contract"):
        validate_exl3_hessian_metrics(
            metrics,
            sample_count=2821,
            sigma_reg=0.025,
        )


@pytest.mark.parametrize("invalid_mse", [-1.0, float("nan"), float("inf")])
def test_hessian_metrics_reject_invalid_scale_search_mse(invalid_mse: float) -> None:
    metrics = {
        "quantizer_path": "hessian_ldlq",
        "hessian_metric_status": "ok",
        "hessian_sample_count": 2821,
        "hessian_regularization_sigma": 0.025,
        "hessian_numerical_contract": EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.000180678,
        "hessian_symmetry_restoration": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "hessian_symmetry_correction_max_abs": 1e-12,
        "selected_global_scale": 0.9926824431197947,
        "scale_search_mse": invalid_mse,
    }

    with pytest.raises(RuntimeError, match="numerical contract"):
        validate_exl3_hessian_metrics(
            metrics,
            sample_count=2821,
            sigma_reg=0.025,
        )


def test_retry_stays_on_the_assigned_worker_and_retains_history(monkeypatch) -> None:
    endpoint = _endpoint("spark-a")
    client = EXL3RemoteClient(
        endpoints=[endpoint],
        token=b"secret",
        coordinator_slots=[],
        timeout_seconds=1,
        max_attempts=2,
    )
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    result = {
        "duration_seconds": 1.0,
        "proxy_error": 0.125,
        "device_names": ["remote:spark-a/cuda:0"],
        "quantizer_metrics": {"reported_metric_kind": "test"},
        "worker": {
            "name": endpoint.name,
            "preflight_sha256": endpoint.preflight_sha256,
            "image_digest": endpoint.image_digest,
        },
    }
    response = encode_tensor_envelope(
        {
            "schema": REMOTE_RESULT_SCHEMA,
            "contract": REMOTE_CONTRACT,
            "request_sha256": request["request_sha256"],
            "checkpoint_hit": True,
            "result": result,
        },
        _packed_tensors(),
    )
    calls: list[str] = []

    monkeypatch.setattr(client, "qualify", lambda value: None)

    def request_once_then_succeed(value, _request):
        calls.append(value.name)
        if len(calls) == 1:
            raise RuntimeError("simulated lost response")
        return response

    monkeypatch.setattr(client, "_request", request_once_then_succeed)
    tensors, returned, transport = client.quantize(
        endpoint=endpoint,
        request_manifest=request,
        input_weight=weight,
        hessian=hessian,
    )
    assert calls == ["spark-a", "spark-a"]
    assert returned == result
    assert set(tensors) == {"trellis", "suh", "svh", "mcg"}
    assert transport["attempts"] == 2
    assert transport["worker_checkpoint_hit"] is True
    assert transport["retry_errors"] == [
        {
            "attempt": 1,
            "error_type": "RuntimeError",
            "message": "simulated lost response",
        }
    ]


def test_client_allows_two_transports_to_stage_for_one_worker(monkeypatch) -> None:
    endpoint = _endpoint("spark-a")
    client = EXL3RemoteClient(
        endpoints=[endpoint],
        token=b"secret",
        coordinator_slots=[],
        timeout_seconds=1,
        max_attempts=1,
    )
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    result = {
        "duration_seconds": 1.0,
        "proxy_error": 0.125,
        "device_names": ["remote:spark-a/cuda:0"],
        "quantizer_metrics": {"reported_metric_kind": "test"},
        "worker": {
            "name": endpoint.name,
            "preflight_sha256": endpoint.preflight_sha256,
            "image_digest": endpoint.image_digest,
        },
    }
    response = encode_tensor_envelope(
        {
            "schema": REMOTE_RESULT_SCHEMA,
            "contract": REMOTE_CONTRACT,
            "request_sha256": request["request_sha256"],
            "checkpoint_hit": False,
            "result": result,
        },
        _packed_tensors(),
    )
    barrier = threading.Barrier(2)
    monkeypatch.setattr(client, "qualify", lambda value: None)

    def staged_request(_endpoint, _request):
        barrier.wait(timeout=2)
        return response

    monkeypatch.setattr(client, "_request", staged_request)

    def quantize_once():
        return client.quantize(
            endpoint=endpoint,
            request_manifest=request,
            input_weight=weight,
            hessian=hessian,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result(timeout=5) for future in [
            executor.submit(quantize_once),
            executor.submit(quantize_once),
        ]]
    assert all(returned == result for _, returned, _ in results)


def test_request_surfaces_authenticated_worker_error(monkeypatch) -> None:
    endpoint = _endpoint("spark-a")
    token = b"secret"
    client = EXL3RemoteClient(
        endpoints=[endpoint],
        token=token,
        coordinator_slots=[],
        timeout_seconds=1,
    )
    payload = json.dumps(
        {"message": "checkpoint request mismatch", "status": "error"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    headers = {
        "Content-Length": str(len(payload)),
        "X-DS4RT-Signature": hmac.new(token, payload, hashlib.sha256).hexdigest(),
    }

    def _reject(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            endpoint.url,
            400,
            "Bad Request",
            headers,
            io.BytesIO(payload),
        )

    monkeypatch.setattr("urllib.request.urlopen", _reject)

    with pytest.raises(
        RuntimeError,
        match="HTTP 400.*checkpoint request mismatch",
    ):
        client._request(
            endpoint,
            urllib.request.Request(endpoint.url),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_worker_quantization_resumes_from_its_packed_checkpoint(tmp_path) -> None:
    endpoint = _endpoint("spark-a")
    identity = {
        "contract": REMOTE_CONTRACT,
        "name": endpoint.name,
        "preflight_sha256": endpoint.preflight_sha256,
        "image_digest": endpoint.image_digest,
    }
    weight, hessian = _inputs()
    request = _request(weight, hessian, endpoint)
    store = EXL3ProjectionCheckpointStore(tmp_path / "worker-checkpoints")

    first_tensors, first_result, first_hit = execute_remote_projection(
        request=request,
        tensors={"input_weight": weight, "hessian": hessian},
        device="cuda:0",
        worker_identity=identity,
        checkpoint_store=store,
    )
    second_tensors, second_result, second_hit = execute_remote_projection(
        request=request,
        tensors={"input_weight": weight, "hessian": hessian},
        device="cuda:0",
        worker_identity=identity,
        checkpoint_store=store,
    )
    assert first_hit is False
    assert second_hit is True
    assert first_result == second_result
    assert set(first_tensors) == {"trellis", "suh", "svh", "mcg"}
    assert all(
        torch.equal(first_tensors[name], second_tensors[name])
        for name in first_tensors
    )
