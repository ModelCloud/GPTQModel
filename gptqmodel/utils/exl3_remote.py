# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Authenticated tensor protocol for deterministic remote EXL3 work."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from ..exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    quantize_exl3,
)
from .exl3_projection_checkpoint import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    EXL3ProjectionCheckpointStore,
    canonical_json_bytes,
    tensor_identity,
)

REMOTE_CONTRACT = "ds4rt.exl3-remote-worker-v1"
REMOTE_SCHEDULER = "dynamic-pipelined-slot-projection-v2"
REMOTE_PIPELINE_DEPTH = 2
REMOTE_POSTPROCESS_ALLOWANCE = 1
REMOTE_ASSIGNMENT_SCHEMA = "ds4rt.exl3-dynamic-projection-assignments"
REMOTE_ASSIGNMENT_SCHEMA_VERSION = 1
EXL3_HESSIAN_CAPTURE_CONTRACT = "raw-xtx-sum-fp32-v1"
REMOTE_REQUEST_SCHEMA = "ds4rt.exl3-remote-request"
REMOTE_RESULT_SCHEMA = "ds4rt.exl3-remote-result"
ENVELOPE_MAGIC = b"DS4EXL3\x00"
MAX_HEADER_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024 * 1024
REMOTE_CHECKPOINT_RETENTION = REMOTE_PIPELINE_DEPTH + 2
_CHOLESKY_MINOR_PATTERN = re.compile(r"leading minor of order (\d+)")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hmac(token: bytes, payload: bytes) -> str:
    return hmac.new(token, payload, hashlib.sha256).hexdigest()


def _finite_scalar(value: torch.Tensor) -> float | None:
    scalar = float(value.item())
    return scalar if math.isfinite(scalar) else None


@torch.no_grad()
def _hessian_matrix_diagnostics(hessian: torch.Tensor) -> dict[str, Any]:
    """Collect bounded, non-mutating diagnostics for one Hessian matrix."""

    rows = int(hessian.shape[0]) if hessian.ndim == 2 else 0
    columns = int(hessian.shape[1]) if hessian.ndim == 2 else 0
    diagnostics: dict[str, Any] = {
        "shape": [rows, columns],
        "dtype": str(hessian.dtype),
        "nonfinite_count": 0,
        "symmetry_max_abs": 0.0,
        "diagonal_min": None,
        "diagonal_mean": None,
        "diagonal_max": None,
    }
    if not rows or rows != columns:
        return diagnostics

    chunk_rows = min(rows, 1024)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        block = hessian[start:stop]
        diagnostics["nonfinite_count"] += int(
            (~torch.isfinite(block)).sum().item()
        )
        difference = block - hessian[:, start:stop].transpose(0, 1)
        block_max = _finite_scalar(difference.abs().max())
        if block_max is not None:
            diagnostics["symmetry_max_abs"] = max(
                diagnostics["symmetry_max_abs"],
                block_max,
            )

    diagonal = torch.diagonal(hessian)
    diagnostics.update(
        {
            "diagonal_min": _finite_scalar(diagonal.min()),
            "diagonal_mean": _finite_scalar(diagonal.mean()),
            "diagonal_max": _finite_scalar(diagonal.max()),
        }
    )
    return diagnostics


@torch.no_grad()
def _symmetrized_cholesky_diagnostics(hessian: torch.Tensor) -> dict[str, Any]:
    """Test the mathematically equivalent symmetric matrix after a failure."""

    rows = int(hessian.shape[0]) if hessian.ndim == 2 else 0
    columns = int(hessian.shape[1]) if hessian.ndim == 2 else 0
    diagnostics: dict[str, Any] = {
        "attempted": False,
        "succeeded": None,
        "leading_minor": None,
    }
    if not rows or rows != columns or not torch.isfinite(hessian).all():
        return diagnostics

    diagnostics["attempted"] = True
    try:
        symmetric = (hessian + hessian.transpose(0, 1)) * 0.5
        factor, info = torch.linalg.cholesky_ex(symmetric, check_errors=False)
        info_value = int(info.item())
        diagnostics["succeeded"] = info_value == 0
        diagnostics["leading_minor"] = info_value or None
        del factor, symmetric
    except Exception as probe_error:
        diagnostics["error_type"] = type(probe_error).__name__
        diagnostics["error"] = str(probe_error)
    return diagnostics


@torch.no_grad()
def _exl3_quantization_failure_message(
    *,
    error: Exception,
    module_full_name: str,
    request_sha256: str | None,
    hessian: torch.Tensor,
    sample_count: int,
    sigma_reg: float,
    raw_hessian: torch.Tensor | None = None,
) -> str:
    """Create bounded diagnostics from EXL3's current Hessian on failure."""

    error_text = str(error)
    minor_match = _CHOLESKY_MINOR_PATTERN.search(error_text)
    current_diagnostics = _hessian_matrix_diagnostics(hessian)
    current_diagnostics.update(
        {
            # quantize_exl3 mutates the raw X^T X input during finalization. At
            # Cholesky time this is the regularized, Hadamard-transformed matrix.
            "state": "quantizer-owned-current-matrix",
            "sample_count": int(sample_count),
            "sigma_reg": float(sigma_reg),
            "symmetrized_cholesky": _symmetrized_cholesky_diagnostics(hessian),
        }
    )
    diagnostics = {
        "schema": "gptqmodel.exl3-quantization-failure",
        "schema_version": 1,
        "module_full_name": module_full_name,
        "request_sha256": request_sha256,
        "error_type": type(error).__name__,
        "error": error_text,
        "cholesky_leading_minor": (
            int(minor_match.group(1)) if minor_match is not None else None
        ),
        "hessian": current_diagnostics,
        "raw_hessian": (
            {
                **_hessian_matrix_diagnostics(raw_hessian),
                "state": "unmodified-raw-xtx-sum",
                "sample_count": int(sample_count),
            }
            if raw_hessian is not None
            else None
        ),
    }
    return "EXL3 quantization failed: " + json.dumps(
        diagnostics,
        sort_keys=True,
        separators=(",", ":"),
    )


def exl3_quantization_failure_message(
    *,
    error: Exception,
    module_full_name: str,
    request_sha256: str | None,
    hessian: torch.Tensor,
    sample_count: int,
    sigma_reg: float,
    raw_hessian: torch.Tensor | None = None,
) -> str:
    """Never let diagnostic collection mask the original quantizer failure."""

    try:
        return _exl3_quantization_failure_message(
            error=error,
            module_full_name=module_full_name,
            request_sha256=request_sha256,
            hessian=hessian,
            sample_count=sample_count,
            sigma_reg=sigma_reg,
            raw_hessian=raw_hessian,
        )
    except Exception as diagnostic_error:
        fallback = {
            "schema": "gptqmodel.exl3-quantization-failure",
            "schema_version": 1,
            "module_full_name": module_full_name,
            "request_sha256": request_sha256,
            "error_type": type(error).__name__,
            "error": str(error),
            "diagnostic_error_type": type(diagnostic_error).__name__,
            "diagnostic_error": str(diagnostic_error),
            "hessian": {
                "state": "diagnostic-collection-failed",
                "sample_count": int(sample_count),
                "sigma_reg": float(sigma_reg),
            },
        }
        return "EXL3 quantization failed: " + json.dumps(
            fallback,
            sort_keys=True,
            separators=(",", ":"),
        )


def encode_tensor_envelope(
    manifest: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> bytes:
    host_tensors = {
        str(name): tensor.detach().contiguous().to(device="cpu")
        for name, tensor in tensors.items()
    }
    if not host_tensors:
        raise ValueError("EXL3 remote tensor envelope cannot be empty")
    tensor_payload = save_safetensors(host_tensors)
    header = {
        "manifest": manifest,
        "tensor_bytes": len(tensor_payload),
        "tensor_sha256": _sha256(tensor_payload),
        "tensors": {name: tensor_identity(tensor) for name, tensor in host_tensors.items()},
    }
    header_payload = canonical_json_bytes(header)
    if len(header_payload) > MAX_HEADER_BYTES:
        raise ValueError("EXL3 remote tensor header is too large")
    return ENVELOPE_MAGIC + struct.pack(">Q", len(header_payload)) + header_payload + tensor_payload


def decode_tensor_envelope(
    payload: bytes,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if len(payload) > max_body_bytes:
        raise ValueError("EXL3 remote tensor envelope exceeds the size limit")
    prefix_bytes = len(ENVELOPE_MAGIC) + 8
    if len(payload) < prefix_bytes or payload[: len(ENVELOPE_MAGIC)] != ENVELOPE_MAGIC:
        raise ValueError("EXL3 remote tensor envelope has invalid magic")
    header_size = struct.unpack(">Q", payload[len(ENVELOPE_MAGIC) : prefix_bytes])[0]
    if header_size <= 0 or header_size > MAX_HEADER_BYTES:
        raise ValueError("EXL3 remote tensor envelope has invalid header size")
    header_end = prefix_bytes + header_size
    if header_end > len(payload):
        raise ValueError("EXL3 remote tensor envelope is truncated")
    try:
        header = json.loads(payload[prefix_bytes:header_end])
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("EXL3 remote tensor header is invalid JSON") from error
    tensor_payload = payload[header_end:]
    if (
        not isinstance(header, dict)
        or not isinstance(header.get("manifest"), dict)
        or header.get("tensor_bytes") != len(tensor_payload)
        or header.get("tensor_sha256") != _sha256(tensor_payload)
    ):
        raise ValueError("EXL3 remote tensor envelope failed content validation")
    try:
        tensors = load_safetensors(tensor_payload)
    except Exception as error:
        raise ValueError("EXL3 remote tensor payload is invalid") from error
    specs = header.get("tensors")
    if not isinstance(specs, dict) or set(specs) != set(tensors):
        raise ValueError("EXL3 remote tensor envelope has an inconsistent tensor set")
    for name, tensor in tensors.items():
        if specs[name] != tensor_identity(tensor):
            raise ValueError(f"EXL3 remote tensor `{name}` failed identity validation")
    return header["manifest"], tensors


def validate_projection_request(request: dict[str, Any]) -> None:
    if (
        not isinstance(request, dict)
        or request.get("schema") != CHECKPOINT_SCHEMA
        or request.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or not isinstance(request.get("request_sha256"), str)
    ):
        raise ValueError("EXL3 remote projection request is malformed")
    clean = deepcopy(request)
    claimed = clean.pop("request_sha256")
    if claimed != _sha256(canonical_json_bytes(clean)):
        raise ValueError("EXL3 remote projection request digest is invalid")


def validate_remote_inputs(
    request: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    validate_projection_request(request)
    if set(tensors) != {"input_weight", "hessian"}:
        raise ValueError("EXL3 remote request requires input_weight and hessian")
    weight = tensors["input_weight"]
    hessian = tensors["hessian"]
    contract = request.get("quantizer_contract")
    if (
        tensor_identity(weight) != request.get("input_weight")
        or tensor_identity(hessian) != request.get("hessian")
        or weight.dtype != torch.float32
        or hessian.dtype != torch.float32
        or weight.ndim != 2
        or hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape[0] != weight.shape[0]
        or weight.shape[0] % 128
        or weight.shape[1] % 128
        or not isinstance(contract, dict)
        or isinstance(contract.get("bits"), bool)
        or not isinstance(contract.get("bits"), int)
        or contract["bits"] < 1
        or contract["bits"] > 8
        or contract.get("codebook") not in {"mcg", "mul1"}
        or contract.get("hessian_capture") != EXL3_HESSIAN_CAPTURE_CONTRACT
        or contract.get("hessian_numerical") != EXL3_HESSIAN_NUMERICAL_CONTRACT
        or contract.get("hessian_symmetry") != EXL3_HESSIAN_SYMMETRY_CONTRACT
        or contract.get("apply_out_scales") not in {None, True, False}
        or isinstance(contract.get("sigma_reg"), bool)
        or not isinstance(contract.get("sigma_reg"), (int, float))
        or not isinstance(contract.get("seed"), int)
        or isinstance(contract.get("seed"), bool)
        or not isinstance(request.get("sample_count"), int)
        or request["sample_count"] <= 0
    ):
        raise ValueError("EXL3 remote request tensor/quantizer contract is invalid")
    return weight, hessian, contract


def validate_remote_output_tensors(
    request: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> None:
    """Reject packed results that cannot represent the requested projection."""

    validate_projection_request(request)
    weight_spec = request.get("input_weight")
    contract = request.get("quantizer_contract")
    if not isinstance(weight_spec, dict) or not isinstance(contract, dict):
        raise TypeError("EXL3 remote output request is incomplete")
    shape = weight_spec.get("shape")
    bits = contract.get("bits")
    codebook = contract.get("codebook")
    expected_names = {"trellis", "suh", "svh", codebook}
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or isinstance(bits, bool)
        or not isinstance(bits, int)
        or codebook not in {"mcg", "mul1"}
        or set(tensors) != expected_names
    ):
        raise ValueError("EXL3 remote output tensor contract is invalid")
    rows, columns = shape
    trellis = tensors["trellis"]
    suh = tensors["suh"]
    svh = tensors["svh"]
    marker = tensors[codebook]
    if (
        trellis.dtype != torch.int16
        or list(trellis.shape) != [rows // 16, columns // 16, bits * 16]
        or suh.dtype != torch.float16
        or list(suh.shape) != [rows]
        or svh.dtype != torch.float16
        or list(svh.shape) != [columns]
        or marker.dtype != torch.int32
        or marker.numel() != 1
    ):
        raise ValueError("EXL3 remote output tensor geometry is inconsistent")


def validate_exl3_hessian_metrics(
    metrics: dict[str, Any],
    *,
    sample_count: int,
    sigma_reg: float,
) -> None:
    """Require the complete production Hessian numerical contract."""

    diagonal_addend = metrics.get("hessian_regularization_diagonal_addend")
    symmetry_correction = metrics.get("hessian_symmetry_correction_max_abs")
    selected_global_scale = metrics.get("selected_global_scale")
    scale_search_mse = metrics.get("scale_search_mse")
    if (
        metrics.get("quantizer_path") != "hessian_ldlq"
        or metrics.get("hessian_metric_status") != "ok"
        or metrics.get("hessian_sample_count") != sample_count
        or metrics.get("hessian_regularization_sigma") != sigma_reg
        or metrics.get("hessian_numerical_contract")
        != EXL3_HESSIAN_NUMERICAL_CONTRACT
        or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
        or metrics.get("hessian_storage_dtype") != "torch.float32"
        or metrics.get("hessian_regularization_placement")
        != "before-fp64-congruence"
        or metrics.get("hessian_symmetry_restoration")
        != EXL3_HESSIAN_SYMMETRY_CONTRACT
        or isinstance(diagonal_addend, bool)
        or not isinstance(diagonal_addend, (int, float))
        or not math.isfinite(diagonal_addend)
        or diagonal_addend <= 0.0
        or isinstance(symmetry_correction, bool)
        or not isinstance(symmetry_correction, (int, float))
        or not math.isfinite(symmetry_correction)
        or symmetry_correction < 0.0
        or isinstance(selected_global_scale, bool)
        or not isinstance(selected_global_scale, (int, float))
        or not math.isfinite(selected_global_scale)
        or selected_global_scale <= 0.0
        or isinstance(scale_search_mse, bool)
        or not isinstance(scale_search_mse, (int, float))
        or not math.isfinite(scale_search_mse)
        or scale_search_mse < 0.0
    ):
        observed = {
            key: metrics.get(key)
            for key in (
                "quantizer_path",
                "hessian_metric_status",
                "hessian_sample_count",
                "hessian_regularization_sigma",
                "hessian_numerical_contract",
                "hessian_transform_compute_dtype",
                "hessian_storage_dtype",
                "hessian_regularization_placement",
                "hessian_regularization_diagonal_addend",
                "hessian_symmetry_restoration",
                "hessian_symmetry_correction_max_abs",
                "selected_global_scale",
                "scale_search_mse",
            )
        }
        raise RuntimeError(
            "EXL3 quantizer returned an invalid Hessian numerical contract: "
            f"expected_sample_count={sample_count}, expected_sigma_reg={sigma_reg}, "
            f"observed={observed!r}"
        )


def execute_remote_projection(
    *,
    request: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    device: torch.device | str,
    worker_identity: dict[str, Any],
    checkpoint_store: EXL3ProjectionCheckpointStore,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], bool]:
    """Run or resume one fail-closed projection on a Spark worker."""

    weight, hessian, contract = validate_remote_inputs(request, tensors)
    expected_execution = {
        "kind": "remote_worker",
        "remote_contract": REMOTE_CONTRACT,
        "name": worker_identity.get("name"),
        "preflight_sha256": worker_identity.get("preflight_sha256"),
        "image_digest": worker_identity.get("image_digest"),
    }
    if contract.get("execution") != expected_execution:
        raise ValueError("EXL3 projection was assigned to a different worker")
    existing = checkpoint_store.load(request)
    if existing is not None:
        out_tensors, result = existing
        validate_remote_output_tensors(request, out_tensors)
        checkpoint_store.prune(
            max_entries=REMOTE_CHECKPOINT_RETENTION,
            preserve_request_sha256=(request["request_sha256"],),
        )
        return out_tensors, result, True

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("EXL3 remote execution requires a CUDA/HIP device")
    quant_args: dict[str, Any] = {
        "K": contract["bits"],
        "devices": [target],
        "apply_out_scales": contract["apply_out_scales"],
        "sigma_reg": float(contract["sigma_reg"]),
        "seed": contract["seed"],
        contract["codebook"]: True,
    }
    started = time.perf_counter()
    device_hessian = hessian.to(device=target)
    try:
        _weight_q, proxy_error, out_tensors = quantize_exl3(
            weight.to(device=target),
            {
                "H": device_hessian,
                "count": request["sample_count"],
                "finalized": False,
            },
            quant_args,
            return_weight_q=False,
        )
    except Exception as error:
        raise RuntimeError(
            exl3_quantization_failure_message(
                error=error,
                module_full_name=request["module"],
                request_sha256=request["request_sha256"],
                hessian=device_hessian,
                sample_count=request["sample_count"],
                sigma_reg=float(contract["sigma_reg"]),
                raw_hessian=hessian,
            )
        ) from error
    del _weight_q
    duration = time.perf_counter() - started
    metrics = quant_args.get("error_metrics")
    if not isinstance(metrics, dict) or quant_args.get("q_fallback") is not False:
        raise RuntimeError("EXL3 remote worker rejected fallback/incomplete metrics")
    validate_exl3_hessian_metrics(
        metrics,
        sample_count=request["sample_count"],
        sigma_reg=float(contract["sigma_reg"]),
    )
    if isinstance(proxy_error, torch.Tensor):
        proxy_error = proxy_error.item()
    if (
        isinstance(proxy_error, bool)
        or not isinstance(proxy_error, (int, float))
        or not math.isfinite(proxy_error)
    ):
        raise RuntimeError("EXL3 remote worker produced a non-finite proxy error")
    worker_name = worker_identity.get("name")
    if not isinstance(worker_name, str) or not worker_name:
        raise ValueError("EXL3 remote worker identity has no stable name")
    result = {
        "duration_seconds": duration,
        "proxy_error": proxy_error,
        "device_names": [f"remote:{worker_name}/{target}"],
        "quantizer_metrics": metrics,
        "worker": deepcopy(worker_identity),
    }
    validate_remote_output_tensors(request, out_tensors)
    checkpoint_store.commit(request, out_tensors, result)
    loaded = checkpoint_store.load(request)
    if loaded is None:
        raise RuntimeError("EXL3 remote worker result did not commit")
    checkpoint_store.prune(
        max_entries=REMOTE_CHECKPOINT_RETENTION,
        preserve_request_sha256=(request["request_sha256"],),
    )
    return loaded[0], loaded[1], False


@dataclass(frozen=True)
class RemoteEndpoint:
    name: str
    url: str
    preflight_sha256: str
    image_digest: str


@dataclass(frozen=True)
class CoordinatorSlot:
    """One explicitly qualified coordinator GPU execution slot."""

    device: str
    gpu_uuid: str
    preflight_sha256: str
    image_digest: str


ExecutionSlot = CoordinatorSlot | RemoteEndpoint


@dataclass
class ExecutionSlotLease:
    """One exclusive physical-slot claim for a projection execution."""

    slot: ExecutionSlot
    assignment_key: str
    wait_seconds: float
    new_assignment: bool
    _client: "EXL3RemoteClient" = field(repr=False)
    _slot_id: str = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def release(self) -> None:
        """Release this lease exactly once; safe from error-cleanup paths."""

        with self._release_lock:
            if self._released:
                return
            self._client._release_slot(self._slot_id)
            self._released = True


class EXL3RemoteClient:
    """Dynamically dispatch durable projections with bounded Spark staging."""

    def __init__(
        self,
        *,
        endpoints: list[RemoteEndpoint],
        token: bytes,
        coordinator_slots: list[CoordinatorSlot],
        timeout_seconds: float,
        max_attempts: int = 2,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        assignment_store_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not endpoints or not token:
            raise ValueError("EXL3 remote client requires endpoints and an auth token")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
            or isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 10
            or isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes <= 0
        ):
            raise ValueError("EXL3 remote client has invalid resource limits")
        names = [endpoint.name for endpoint in endpoints]
        if len(set(names)) != len(names):
            raise ValueError("EXL3 remote client has duplicate endpoint names")
        devices = [slot.device for slot in coordinator_slots]
        if (
            len(set(devices)) != len(devices)
            or any(
                not isinstance(device, str)
                or re.fullmatch(r"cuda:\d+", device) is None
                for device in devices
            )
        ):
            raise ValueError("EXL3 remote client has invalid coordinator slots")
        self.endpoints = tuple(sorted(endpoints, key=lambda endpoint: endpoint.name))
        self.coordinator_slots = tuple(
            sorted(
                coordinator_slots,
                key=lambda slot: int(slot.device.split(":", 1)[1]),
            )
        )
        self.token = token
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max_attempts
        self.max_body_bytes = int(max_body_bytes)
        self._slots: tuple[ExecutionSlot, ...] = (
            *self.coordinator_slots,
            *self.endpoints,
        )
        self._slot_by_id = {
            self._slot_id(slot): slot
            for slot in self._slots
        }
        if len(self._slot_by_id) != len(self._slots):
            raise ValueError("EXL3 remote client has duplicate execution slots")
        self._slot_order = {
            slot_id: index for index, slot_id in enumerate(self._slot_by_id)
        }
        self._slot_contracts = {
            slot_id: self.execution_contract(slot)
            for slot_id, slot in self._slot_by_id.items()
        }
        self._slot_capacities = {
            slot_id: (
                1
                if isinstance(slot, CoordinatorSlot)
                else REMOTE_PIPELINE_DEPTH
            )
            for slot_id, slot in self._slot_by_id.items()
        }
        self._topology_sha256 = _sha256(canonical_json_bytes(self._slot_contracts))
        self.assignment_store_path = None
        if assignment_store_path is not None:
            raw_assignment_path = Path(assignment_store_path).expanduser()
            if raw_assignment_path.is_symlink():
                raise ValueError("EXL3 assignment store cannot be a symbolic link")
            self.assignment_store_path = raw_assignment_path.resolve()
        self._slot_condition = threading.Condition()
        self._busy_slots: dict[str, int] = {}
        self._assignments = self._load_assignments()
        self._identity_lock = threading.Lock()
        self._qualified: set[str] = set()

    @staticmethod
    def _slot_id(slot: ExecutionSlot) -> str:
        if isinstance(slot, CoordinatorSlot):
            return f"coordinator:{slot.device}"
        return f"remote_worker:{slot.name}"

    def _assignment_record(self, assignment_key: str, slot_id: str) -> dict[str, Any]:
        body = {
            "schema": REMOTE_ASSIGNMENT_SCHEMA,
            "schema_version": REMOTE_ASSIGNMENT_SCHEMA_VERSION,
            "scheduler": REMOTE_SCHEDULER,
            "topology_sha256": self._topology_sha256,
            "assignment_key": assignment_key,
            "assignment_key_sha256": _sha256(assignment_key.encode()),
            "slot_id": slot_id,
            "execution": self._slot_contracts[slot_id],
        }
        return {
            **body,
            "record_sha256": _sha256(canonical_json_bytes(body)),
        }

    def _load_assignments(self) -> dict[str, str]:
        root = self.assignment_store_path
        if root is None or not root.exists():
            return {}
        if root.is_symlink() or not root.is_dir():
            raise ValueError("EXL3 assignment store is not a regular directory")
        assignments: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise ValueError("EXL3 assignment store contains a symbolic link")
                continue
            if path.name.startswith("."):
                # A killed atomic writer may leave only an unpublished temp file.
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ValueError("EXL3 assignment store contains an invalid entry")
            try:
                record = json.loads(path.read_bytes())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("EXL3 assignment record is not valid JSON") from error
            if not isinstance(record, dict):
                raise ValueError("EXL3 assignment record is malformed")
            body = {
                key: value
                for key, value in record.items()
                if key != "record_sha256"
            }
            assignment_key = record.get("assignment_key")
            slot_id = record.get("slot_id")
            key_sha256 = (
                _sha256(assignment_key.encode())
                if isinstance(assignment_key, str)
                else None
            )
            expected_relative = (
                Path(key_sha256[:2]) / key_sha256[2:4] / f"{key_sha256}.json"
                if key_sha256 is not None
                else None
            )
            if (
                record.get("schema") != REMOTE_ASSIGNMENT_SCHEMA
                or record.get("schema_version") != REMOTE_ASSIGNMENT_SCHEMA_VERSION
                or record.get("scheduler") != REMOTE_SCHEDULER
                or record.get("topology_sha256") != self._topology_sha256
                or not isinstance(assignment_key, str)
                or not assignment_key
                or len(assignment_key) > 4096
                or record.get("assignment_key_sha256") != key_sha256
                or path.relative_to(root) != expected_relative
                or not isinstance(slot_id, str)
                or slot_id not in self._slot_by_id
                or record.get("execution") != self._slot_contracts[slot_id]
                or record.get("record_sha256") != _sha256(canonical_json_bytes(body))
                or assignment_key in assignments
            ):
                raise ValueError("EXL3 assignment store failed identity validation")
            assignments[assignment_key] = slot_id
        return assignments

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_assignment_directory(self, path: Path) -> None:
        missing = []
        current = path
        while not current.exists():
            missing.append(current)
            current = current.parent
        if current.is_symlink() or not current.is_dir():
            raise ValueError("EXL3 assignment-store parent is invalid")
        for directory in reversed(missing):
            directory.mkdir()
            self._fsync_directory(directory.parent)

    def _persist_assignment(self, assignment_key: str, slot_id: str) -> None:
        root = self.assignment_store_path
        if root is None:
            raise ValueError("EXL3 dynamic scheduling requires an assignment store")
        key_sha256 = _sha256(assignment_key.encode())
        path = root / key_sha256[:2] / key_sha256[2:4] / f"{key_sha256}.json"
        self._ensure_assignment_directory(path.parent)
        if path.is_symlink():
            raise ValueError("EXL3 assignment record cannot be a symbolic link")
        if path.exists():
            raise ValueError("EXL3 assignment record appeared concurrently")
        payload = canonical_json_bytes(
            self._assignment_record(assignment_key, slot_id)
        ) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, path)
            self._fsync_directory(path.parent)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def acquire_slot(self, assignment_key: str) -> ExecutionSlotLease:
        """Wait for a physical slot or bounded Spark staging position."""

        if (
            not isinstance(assignment_key, str)
            or not assignment_key
            or len(assignment_key) > 4096
        ):
            raise ValueError("EXL3 projection assignment key is invalid")
        wait_started = time.perf_counter()
        new_assignment = False
        with self._slot_condition:
            while True:
                slot_id = self._assignments.get(assignment_key)
                if slot_id is None:
                    available = [
                        candidate
                        for candidate in self._slot_by_id
                        if self._busy_slots.get(candidate, 0)
                        < self._slot_capacities[candidate]
                    ]
                    slot_id = (
                        min(
                            available,
                            key=lambda candidate: (
                                self._busy_slots.get(candidate, 0)
                                / self._slot_capacities[candidate],
                                self._slot_order[candidate],
                            ),
                        )
                        if available
                        else None
                    )
                    if slot_id is not None:
                        # This commit is the ownership barrier. A killed run will
                        # either see the old complete file or the new complete file.
                        self._persist_assignment(assignment_key, slot_id)
                        self._assignments[assignment_key] = slot_id
                        new_assignment = True
                if (
                    slot_id is not None
                    and self._busy_slots.get(slot_id, 0)
                    < self._slot_capacities[slot_id]
                ):
                    self._busy_slots[slot_id] = self._busy_slots.get(slot_id, 0) + 1
                    return ExecutionSlotLease(
                        slot=self._slot_by_id[slot_id],
                        assignment_key=assignment_key,
                        wait_seconds=time.perf_counter() - wait_started,
                        new_assignment=new_assignment,
                        _client=self,
                        _slot_id=slot_id,
                    )
                self._slot_condition.wait()

    def _release_slot(self, slot_id: str) -> None:
        with self._slot_condition:
            active = self._busy_slots.get(slot_id, 0)
            if active <= 0:
                raise RuntimeError(f"EXL3 execution slot `{slot_id}` is not leased")
            if active == 1:
                del self._busy_slots[slot_id]
            else:
                self._busy_slots[slot_id] = active - 1
            self._slot_condition.notify_all()

    @staticmethod
    def execution_contract(slot: ExecutionSlot) -> dict[str, Any]:
        """Return the immutable slot identity bound into a projection request."""

        if isinstance(slot, CoordinatorSlot):
            return {
                "kind": "coordinator",
                "remote_contract": REMOTE_CONTRACT,
                "device": slot.device,
                "gpu_uuid": slot.gpu_uuid,
                "preflight_sha256": slot.preflight_sha256,
                "image_digest": slot.image_digest,
            }
        endpoint = slot
        return {
            "kind": "remote_worker",
            "remote_contract": REMOTE_CONTRACT,
            "name": endpoint.name,
            "preflight_sha256": endpoint.preflight_sha256,
            "image_digest": endpoint.image_digest,
        }

    def _read_response(self, endpoint: RemoteEndpoint, response) -> bytes:
        payload = response.read(self.max_body_bytes + 1)
        signature = response.headers.get("X-DS4RT-Signature")
        content_length = response.headers.get("Content-Length")
        if len(payload) > self.max_body_bytes:
            raise RuntimeError(f"EXL3 worker `{endpoint.name}` response is too large")
        if content_length is not None:
            try:
                expected_length = int(content_length)
            except ValueError as error:
                raise RuntimeError(
                    f"EXL3 worker `{endpoint.name}` returned invalid content length"
                ) from error
            if expected_length != len(payload):
                raise RuntimeError(
                    f"EXL3 worker `{endpoint.name}` response is truncated"
                )
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, _hmac(self.token, payload)
        ):
            raise RuntimeError(f"EXL3 worker `{endpoint.name}` response signature is invalid")
        return payload

    def _request(self, endpoint: RemoteEndpoint, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return self._read_response(endpoint, response)
        except urllib.error.HTTPError as error:
            try:
                payload = self._read_response(endpoint, error)
            except RuntimeError as response_error:
                raise RuntimeError(
                    f"EXL3 worker `{endpoint.name}` returned an invalid error response"
                ) from response_error
            finally:
                error.close()
            try:
                error_record = json.loads(payload)
            except (UnicodeError, json.JSONDecodeError) as decode_error:
                raise RuntimeError(
                    f"EXL3 worker `{endpoint.name}` returned an invalid error record"
                ) from decode_error
            message = error_record.get("message") if isinstance(error_record, dict) else None
            if (
                not isinstance(error_record, dict)
                or error_record.get("status") != "error"
                or not isinstance(message, str)
                or not message
            ):
                raise RuntimeError(
                    f"EXL3 worker `{endpoint.name}` returned an inconsistent error record"
                )
            raise RuntimeError(
                f"EXL3 worker `{endpoint.name}` rejected request "
                f"(HTTP {error.code}): {message}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(f"EXL3 worker `{endpoint.name}` request failed") from error

    def qualify(self, endpoint: RemoteEndpoint) -> None:
        with self._identity_lock:
            if endpoint.name in self._qualified:
                return
        auth_payload = b"GET /v1/identity"
        request = urllib.request.Request(
            endpoint.url.rstrip("/") + "/v1/identity",
            headers={"X-DS4RT-Signature": _hmac(self.token, auth_payload)},
            method="GET",
        )
        payload = self._request(endpoint, request)
        try:
            identity = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"EXL3 worker `{endpoint.name}` identity is invalid JSON"
            ) from error
        if (
            not isinstance(identity, dict)
            or identity.get("contract") != REMOTE_CONTRACT
            or identity.get("name") != endpoint.name
            or identity.get("preflight_sha256") != endpoint.preflight_sha256
            or identity.get("image_digest") != endpoint.image_digest
        ):
            raise RuntimeError(f"EXL3 worker `{endpoint.name}` identity drifted")
        with self._identity_lock:
            self._qualified.add(endpoint.name)

    def quantize(
        self,
        *,
        endpoint: RemoteEndpoint,
        request_manifest: dict[str, Any],
        input_weight: torch.Tensor,
        hessian: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        # Multiple request threads may upload/decode concurrently. The worker's
        # quantize_lock remains the single-GPU execution barrier, so the second
        # request stages CPU/network work instead of overlapping trellis kernels.
        self.qualify(endpoint)
        manifest = {
            "schema": REMOTE_REQUEST_SCHEMA,
            "contract": REMOTE_CONTRACT,
            "request": request_manifest,
        }
        payload = encode_tensor_envelope(
            manifest,
            {"input_weight": input_weight, "hessian": hessian},
        )
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                endpoint.url.rstrip("/") + "/v1/exl3/quantize",
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(payload)),
                    "X-DS4RT-Signature": _hmac(self.token, payload),
                },
                method="POST",
            )
            try:
                response_payload = self._request(endpoint, request)
                response_manifest, tensors = decode_tensor_envelope(
                    response_payload,
                    max_body_bytes=self.max_body_bytes,
                )
                result = response_manifest.get("result")
                worker = result.get("worker") if isinstance(result, dict) else None
                if (
                    response_manifest.get("schema") != REMOTE_RESULT_SCHEMA
                    or response_manifest.get("contract") != REMOTE_CONTRACT
                    or response_manifest.get("request_sha256")
                    != request_manifest.get("request_sha256")
                    or not isinstance(response_manifest.get("checkpoint_hit"), bool)
                    or not isinstance(result, dict)
                    or not isinstance(worker, dict)
                    or worker.get("name") != endpoint.name
                    or worker.get("preflight_sha256") != endpoint.preflight_sha256
                    or worker.get("image_digest") != endpoint.image_digest
                ):
                    raise RuntimeError(
                        f"EXL3 worker `{endpoint.name}` result is inconsistent"
                    )
                validate_remote_output_tensors(request_manifest, tensors)
                return tensors, result, {
                    "attempts": attempt,
                    "retry_errors": errors,
                    "worker_checkpoint_hit": response_manifest["checkpoint_hit"],
                }
            except RuntimeError as error:
                errors.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        f"EXL3 worker `{endpoint.name}` failed after "
                        f"{self.max_attempts} attempts"
                    ) from error
        raise AssertionError("unreachable EXL3 remote retry state")


def remote_client_from_provenance(
    provenance: dict[str, Any] | None,
) -> EXL3RemoteClient | None:
    if not isinstance(provenance, dict):
        return None
    run = provenance.get("run")
    remote = run.get("remote_workers") if isinstance(run, dict) else None
    if remote is None:
        return None
    if (
        not isinstance(remote, dict)
        or remote.get("contract") != REMOTE_CONTRACT
        or remote.get("scheduler") != REMOTE_SCHEDULER
        or not isinstance(remote.get("endpoints"), list)
        or not remote["endpoints"]
        or not isinstance(remote.get("coordinator_slots"), list)
        or not isinstance(remote.get("assignment_store"), str)
        or not remote["assignment_store"]
        or not isinstance(remote.get("token_env"), str)
        or not remote["token_env"]
        or isinstance(remote.get("timeout_seconds", 7200), bool)
        or not isinstance(remote.get("timeout_seconds", 7200), (int, float))
        or remote.get("timeout_seconds", 7200) <= 0
        or isinstance(remote.get("max_attempts", 2), bool)
        or not isinstance(remote.get("max_attempts", 2), int)
        or not 1 <= remote.get("max_attempts", 2) <= 10
    ):
        raise ValueError("invalid EXL3 remote-worker run contract")
    token_value = os.environ.get(remote["token_env"])
    if not token_value:
        raise ValueError(f"EXL3 remote auth token env `{remote['token_env']}` is unset")
    endpoints = []
    names = set()
    for value in remote["endpoints"]:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("name"), str)
            or not value["name"]
            or value["name"] in names
            or not isinstance(value.get("url"), str)
            or not value["url"].startswith("http://")
            or urllib.parse.urlsplit(value["url"]).scheme != "http"
            or not urllib.parse.urlsplit(value["url"]).hostname
            or not isinstance(value.get("preflight_sha256"), str)
            or len(value["preflight_sha256"]) != 64
            or not isinstance(value.get("image_digest"), str)
            or not value["image_digest"].startswith("sha256:")
        ):
            raise ValueError("invalid EXL3 remote-worker endpoint contract")
        names.add(value["name"])
        endpoints.append(
            RemoteEndpoint(
                name=value["name"],
                url=value["url"],
                preflight_sha256=value["preflight_sha256"],
                image_digest=value["image_digest"],
            )
        )
    coordinator_slots = []
    devices = set()
    gpu_uuids = set()
    for value in remote["coordinator_slots"]:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("device"), str)
            or re.fullmatch(r"cuda:\d+", value["device"]) is None
            or value["device"] in devices
            or not isinstance(value.get("gpu_uuid"), str)
            or not value["gpu_uuid"]
            or value["gpu_uuid"] in gpu_uuids
            or not isinstance(value.get("preflight_sha256"), str)
            or len(value["preflight_sha256"]) != 64
            or not isinstance(value.get("image_digest"), str)
            or not value["image_digest"].startswith("sha256:")
        ):
            raise ValueError("invalid EXL3 coordinator-slot contract")
        devices.add(value["device"])
        gpu_uuids.add(value["gpu_uuid"])
        coordinator_slots.append(
            CoordinatorSlot(
                device=value["device"],
                gpu_uuid=value["gpu_uuid"],
                preflight_sha256=value["preflight_sha256"],
                image_digest=value["image_digest"],
            )
        )
    expected_orchestration_workers = len(coordinator_slots) + (
        REMOTE_PIPELINE_DEPTH + REMOTE_POSTPROCESS_ALLOWANCE
    ) * len(endpoints)
    if remote.get("orchestration_workers") != expected_orchestration_workers:
        raise ValueError("invalid EXL3 remote-worker orchestration width")
    return EXL3RemoteClient(
        endpoints=endpoints,
        token=token_value.encode(),
        coordinator_slots=coordinator_slots,
        timeout_seconds=float(remote.get("timeout_seconds", 7200)),
        max_attempts=int(remote.get("max_attempts", 2)),
        assignment_store_path=remote["assignment_store"],
    )


__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "EXL3_HESSIAN_CAPTURE_CONTRACT",
    "REMOTE_CONTRACT",
    "REMOTE_PIPELINE_DEPTH",
    "REMOTE_POSTPROCESS_ALLOWANCE",
    "REMOTE_REQUEST_SCHEMA",
    "REMOTE_RESULT_SCHEMA",
    "REMOTE_SCHEDULER",
    "EXL3RemoteClient",
    "CoordinatorSlot",
    "ExecutionSlotLease",
    "ExecutionSlot",
    "RemoteEndpoint",
    "decode_tensor_envelope",
    "encode_tensor_envelope",
    "exl3_quantization_failure_message",
    "execute_remote_projection",
    "remote_client_from_provenance",
    "validate_projection_request",
    "validate_remote_inputs",
    "validate_remote_output_tensors",
]
