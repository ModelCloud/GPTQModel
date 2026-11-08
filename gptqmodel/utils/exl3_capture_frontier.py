# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Crash-consistent EXL3 Hessian and routing capture frontiers."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterable

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
import xxhash

from .exl3_error_ledger import routed_expert_identity, routed_expert_processor_layer


CAPTURE_FRONTIER_ENV = "GPTQMODEL_EXL3_CAPTURE_FRONTIER"
CAPTURE_FRONTIER_SCHEMA = "ds4rt.exl3-capture-frontier"
CAPTURE_FRONTIER_SCHEMA_VERSION = 1
CAPTURE_FRONTIER_CONTRACT = (
    f"{CAPTURE_FRONTIER_SCHEMA}-v{CAPTURE_FRONTIER_SCHEMA_VERSION}"
)
PAYLOAD_HASH_ALGORITHM = "xxh3-128"
HESSIAN_CONTRACT = "normalized-2-over-n-xtx-fp32-v1"
MANIFEST_FILENAME = "manifest.json"
_COMMITTED_DIRECTORY = re.compile(
    r"layer-(?P<layer>[0-9]{6})-subset-(?P<subset>[0-9]{4})-"
    r"of-(?P<total>[0-9]{4})-(?P<digest>[0-9a-f]{16})\Z"
)


class EXL3CaptureFrontierError(RuntimeError):
    """A capture frontier is incomplete, corrupt, or belongs to another run."""


@dataclass(frozen=True)
class EXL3CaptureState:
    module: str
    hessian: torch.Tensor
    sample_count: int
    route_evidence: dict[str, Any] | None
    zero_route_recovery: dict[str, Any] | None = None


@dataclass(frozen=True)
class EXL3CaptureDescriptor:
    """Hessian-independent metadata used by the bounded streaming writer."""

    module: str
    sample_count: int
    route_evidence: dict[str, Any] | None
    zero_route_recovery: dict[str, Any] | None = None


@dataclass(frozen=True)
class EXL3CaptureRecord:
    """Validated on-disk capture that can be hydrated on demand."""

    module: str
    path: Path
    payload: dict[str, Any]
    sample_count: int
    route_evidence: dict[str, Any] | None
    zero_route_recovery: dict[str, Any] | None = None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _xxh3_128_file(path: Path) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json_bytes(value) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def _projection_phase(identity: dict[str, Any]) -> str:
    projection = identity.get("projection")
    if projection in ("w1", "w3"):
        return "gate-up"
    if projection == "w2":
        return "down"
    raise EXL3CaptureFrontierError("capture frontier has an unknown projection")


class EXL3CaptureFrontierStore:
    """Retain exact subset captures until their output layer is durable."""

    def __init__(self, root: str | os.PathLike[str], *, family_join: dict[str, Any]):
        self.root = Path(root).expanduser().resolve()
        if not isinstance(family_join, dict):
            raise EXL3CaptureFrontierError("capture frontier requires family identity")
        self.family_join = deepcopy(family_join)
        self.family_join_sha256 = _sha256_bytes(canonical_json_bytes(family_join))

    @staticmethod
    def _module_names(subset: dict[str, Any]) -> list[str]:
        names = []
        for task_name, named_module in subset.items():
            full_name = getattr(named_module, "full_name", None)
            if not isinstance(full_name, str) or not full_name:
                raise EXL3CaptureFrontierError(
                    f"capture subset task `{task_name}` has no full module name"
                )
            if routed_expert_identity(full_name) is None:
                raise EXL3CaptureFrontierError(
                    f"capture frontier only supports routed experts: {full_name}"
                )
            names.append(full_name)
        if not names or len(names) != len(set(names)):
            raise EXL3CaptureFrontierError("capture subset module names are invalid")
        return sorted(names)

    def _key(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        module_names: list[str],
    ) -> tuple[dict[str, Any], str, Path]:
        if layer_index < 0 or not 0 <= subset_index < subset_total:
            raise EXL3CaptureFrontierError("capture subset coordinates are invalid")
        body = {
            "family_join_sha256": self.family_join_sha256,
            "layer_index": int(layer_index),
            "subset_index": int(subset_index),
            "subset_total": int(subset_total),
            "module_names": module_names,
        }
        digest = _sha256_bytes(canonical_json_bytes(body))
        directory = self.root / (
            f"layer-{layer_index:06d}-subset-{subset_index:04d}-"
            f"of-{subset_total:04d}-{digest[:16]}"
        )
        return body, digest, directory

    def _prune_incomplete(self) -> None:
        if not self.root.exists():
            return
        if not self.root.is_dir() or self.root.is_symlink():
            raise EXL3CaptureFrontierError("capture-frontier root is unsafe")
        changed = False
        for path in self.root.iterdir():
            if path.name.startswith(".capture-") and path.name.endswith(".tmp"):
                if not path.is_dir() or path.is_symlink():
                    raise EXL3CaptureFrontierError("incomplete capture entry is unsafe")
                shutil.rmtree(path)
                changed = True
            elif _COMMITTED_DIRECTORY.fullmatch(path.name) is None:
                raise EXL3CaptureFrontierError(
                    f"unexpected capture-frontier entry: {path.name}"
                )
        if changed:
            _fsync_directory(self.root)

    def _read_manifest(
        self,
        directory: Path,
        *,
        key_body: dict[str, Any],
        key_sha256: str,
    ) -> dict[str, Any]:
        path = directory / MANIFEST_FILENAME
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EXL3CaptureFrontierError("cannot read capture manifest") from error
        if not isinstance(manifest, dict):
            raise EXL3CaptureFrontierError("capture manifest is not an object")
        digest = manifest.get("manifest_sha256")
        body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        match = _COMMITTED_DIRECTORY.fullmatch(directory.name)
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or match is None
            or manifest.get("schema") != CAPTURE_FRONTIER_SCHEMA
            or manifest.get("schema_version") != CAPTURE_FRONTIER_SCHEMA_VERSION
            or manifest.get("payload_hash_algorithm") != PAYLOAD_HASH_ALGORITHM
            or manifest.get("hessian_contract") != HESSIAN_CONTRACT
            or manifest.get("family_join") != self.family_join
            or manifest.get("capture_key") != key_body
            or manifest.get("capture_key_sha256") != key_sha256
            or not isinstance(digest, str)
            or _sha256_bytes(canonical_json_bytes(body)) != digest
            or match.group("digest") != key_sha256[:16]
            or int(match.group("layer")) != key_body["layer_index"]
            or int(match.group("subset")) != key_body["subset_index"]
            or int(match.group("total")) != key_body["subset_total"]
        ):
            raise EXL3CaptureFrontierError("capture manifest failed validation")
        return manifest

    def restore_index(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: dict[str, Any],
    ) -> dict[str, EXL3CaptureRecord] | None:
        """Validate a frontier and return lightweight on-disk records.

        Payload hashes and the complete file set are checked here, but Hessian
        tensors are not materialized.  Callers can therefore recover a very
        large MoE layer with memory bounded by active projection concurrency.
        """

        module_names = self._module_names(subset)
        key_body, key_sha256, directory = self._key(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            module_names=module_names,
        )
        self._prune_incomplete()
        if not directory.exists():
            return None
        manifest = self._read_manifest(
            directory, key_body=key_body, key_sha256=key_sha256
        )
        records = manifest.get("captures")
        if not isinstance(records, list) or len(records) != len(module_names):
            raise EXL3CaptureFrontierError("capture manifest has incomplete modules")

        expected_files = {MANIFEST_FILENAME}
        records_by_module: dict[str, EXL3CaptureRecord] = {}
        validated_files: set[str] = set()
        for record in records:
            module_name = record.get("module") if isinstance(record, dict) else None
            identity = (
                routed_expert_identity(module_name, family_join=self.family_join)
                if isinstance(module_name, str)
                else None
            )
            payload = record.get("hessian") if isinstance(record, dict) else None
            sample_count = record.get("sample_count") if isinstance(record, dict) else None
            if (
                module_name not in module_names
                or module_name in records_by_module
                or identity is None
                or record.get("expert_identity") != identity
                or record.get("phase") != _projection_phase(identity)
                or isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or not isinstance(payload, dict)
            ):
                raise EXL3CaptureFrontierError("capture module record is invalid")
            relative = payload.get("file")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or Path(relative).parts[:1] != ("hessians",)
            ):
                raise EXL3CaptureFrontierError("capture Hessian path is unsafe")
            path = directory / relative
            expected_files.add(relative)
            tensor_spec = payload.get("tensor")
            if (
                not isinstance(tensor_spec, dict)
                or tensor_spec.get("dtype") != str(torch.float32)
                or not isinstance(tensor_spec.get("shape"), list)
                or len(tensor_spec["shape"]) != 2
                or tensor_spec["shape"][0] != tensor_spec["shape"][1]
                or tensor_spec.get("bytes")
                != tensor_spec["shape"][0] * tensor_spec["shape"][1] * 4
                or not isinstance(payload.get("bytes"), int)
                or payload["bytes"] < tensor_spec["bytes"]
            ):
                raise EXL3CaptureFrontierError(
                    f"capture Hessian has invalid geometry: {relative}"
                )
            if relative not in validated_files:
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != payload.get("bytes")
                    or _xxh3_128_file(path) != payload.get("xxh3_128")
                ):
                    raise EXL3CaptureFrontierError(
                        f"capture Hessian failed validation: {relative}"
                    )
                validated_files.add(relative)
            records_by_module[module_name] = EXL3CaptureRecord(
                module=module_name,
                path=path,
                payload=deepcopy(payload),
                sample_count=sample_count,
                route_evidence=deepcopy(record.get("route_evidence")),
                zero_route_recovery=deepcopy(
                    record.get("zero_route_recovery")
                ),
            )

        actual_files: set[str] = set()
        for path in directory.rglob("*"):
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise EXL3CaptureFrontierError("capture frontier contains a symlink")
            if path.is_dir():
                if relative != "hessians":
                    raise EXL3CaptureFrontierError(
                        f"capture frontier contains an unexpected directory: {relative}"
                    )
            elif path.is_file():
                actual_files.add(relative)
            else:
                raise EXL3CaptureFrontierError("capture frontier entry is unsupported")
        if actual_files != expected_files or set(records_by_module) != set(module_names):
            raise EXL3CaptureFrontierError("capture-frontier file set is inconsistent")
        return records_by_module

    @staticmethod
    def load_record_hessian(
        record: EXL3CaptureRecord,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Hydrate one already validated Hessian payload."""

        tensors = load_safetensors_file(record.path, device=str(device))
        hessian = tensors.get("H")
        if (
            set(tensors) != {"H"}
            or not isinstance(hessian, torch.Tensor)
            or _tensor_spec(hessian) != record.payload.get("tensor")
            or hessian.dtype != torch.float32
            or hessian.ndim != 2
            or hessian.shape[0] != hessian.shape[1]
        ):
            raise EXL3CaptureFrontierError(
                f"capture Hessian has invalid geometry: {record.path.name}"
            )
        return hessian

    def restore(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: dict[str, Any],
    ) -> dict[str, EXL3CaptureState] | None:
        """Compatibility API that eagerly materializes a complete frontier."""

        records = self.restore_index(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            subset=subset,
        )
        if records is None:
            return None
        loaded_hessians: dict[Path, torch.Tensor] = {}
        states: dict[str, EXL3CaptureState] = {}
        for module_name, record in records.items():
            hessian = loaded_hessians.get(record.path)
            if hessian is None:
                hessian = self.load_record_hessian(record)
                loaded_hessians[record.path] = hessian
            states[module_name] = EXL3CaptureState(
                module=module_name,
                hessian=hessian,
                sample_count=record.sample_count,
                route_evidence=deepcopy(record.route_evidence),
                zero_route_recovery=deepcopy(
                    record.zero_route_recovery
                ),
            )
        return states

    def commit(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: dict[str, Any],
        states: Iterable[EXL3CaptureState],
    ) -> dict[str, Any]:
        state_by_module = {state.module: state for state in states}
        descriptors = {
            module: EXL3CaptureDescriptor(
                module=module,
                sample_count=state.sample_count,
                route_evidence=deepcopy(state.route_evidence),
                zero_route_recovery=deepcopy(
                    state.zero_route_recovery
                ),
            )
            for module, state in state_by_module.items()
        }
        return self.commit_streaming(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            subset=subset,
            descriptors=descriptors.values(),
            hessian_loader=lambda module: state_by_module[module].hessian,
        )

    def commit_streaming(
        self,
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        subset: dict[str, Any],
        descriptors: Iterable[EXL3CaptureDescriptor],
        hessian_loader: Callable[[str], torch.Tensor],
    ) -> dict[str, Any]:
        """Commit a frontier while holding only one expert family in memory."""

        module_names = self._module_names(subset)
        state_by_module = {state.module: state for state in descriptors}
        if set(state_by_module) != set(module_names):
            raise EXL3CaptureFrontierError("capture commit has incomplete modules")
        key_body, key_sha256, destination = self._key(
            layer_index=layer_index,
            subset_index=subset_index,
            subset_total=subset_total,
            module_names=module_names,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise EXL3CaptureFrontierError("capture-frontier root cannot be a symlink")
        self._prune_incomplete()
        if destination.exists():
            return self._read_manifest(
                destination, key_body=key_body, key_sha256=key_sha256
            )

        temporary = Path(
            tempfile.mkdtemp(prefix=".capture-", suffix=".tmp", dir=self.root)
        )
        try:
            hessian_root = temporary / "hessians"
            hessian_root.mkdir()
            grouped: dict[
                tuple[str, int, int, str], list[EXL3CaptureDescriptor]
            ] = {}
            identities: dict[str, dict[str, Any]] = {}
            for module_name in module_names:
                state = state_by_module[module_name]
                identity = routed_expert_identity(module_name, family_join=self.family_join)
                if identity is None:
                    raise EXL3CaptureFrontierError("capture module is not routed")
                phase = _projection_phase(identity)
                if state.sample_count <= 0:
                    raise EXL3CaptureFrontierError(
                        f"capture sample count is invalid: {module_name}"
                    )
                identities[module_name] = identity
                grouped.setdefault(
                    (
                        identity["block_namespace"],
                        identity["logical_layer"],
                        identity["expert"],
                        phase,
                    ),
                    [],
                ).append(state)

            payload_by_module: dict[str, dict[str, Any]] = {}
            for payload_index, (_group, group_states) in enumerate(sorted(grouped.items())):
                reference = group_states[0]
                reference_hessian = hessian_loader(reference.module)
                if (
                    not isinstance(reference_hessian, torch.Tensor)
                    or reference_hessian.device.type != "cpu"
                    or reference_hessian.dtype != torch.float32
                    or reference_hessian.ndim != 2
                    or reference_hessian.shape[0] != reference_hessian.shape[1]
                ):
                    raise EXL3CaptureFrontierError(
                        f"capture Hessian is not normalized CPU FP32: {reference.module}"
                    )
                for candidate in group_states[1:]:
                    candidate_hessian = hessian_loader(candidate.module)
                    matches = (
                        isinstance(candidate_hessian, torch.Tensor)
                        and candidate_hessian.device.type == "cpu"
                        and candidate_hessian.dtype == torch.float32
                        and candidate_hessian.ndim == 2
                        and candidate_hessian.shape == reference_hessian.shape
                        and candidate.sample_count == reference.sample_count
                        and candidate.route_evidence == reference.route_evidence
                        and candidate.zero_route_recovery
                        == reference.zero_route_recovery
                        and torch.equal(candidate_hessian, reference_hessian)
                    )
                    del candidate_hessian
                    if not matches:
                        raise EXL3CaptureFrontierError(
                            "gate/up captures for one expert are not identical"
                        )
                relative = Path("hessians") / f"hessian-{payload_index:06d}.safetensors"
                path = temporary / relative
                save_safetensors_file({"H": reference_hessian.contiguous()}, path)
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                payload = {
                    "file": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "xxh3_128": _xxh3_128_file(path),
                    "tensor": _tensor_spec(reference_hessian),
                }
                for state in group_states:
                    payload_by_module[state.module] = payload
                del reference_hessian
            _fsync_directory(hessian_root)

            captures = []
            for module_name in module_names:
                state = state_by_module[module_name]
                identity = identities[module_name]
                captures.append(
                    {
                        "module": module_name,
                        "expert_identity": identity,
                        "phase": _projection_phase(identity),
                        "sample_count": int(state.sample_count),
                        "route_evidence": deepcopy(state.route_evidence),
                        "zero_route_recovery": deepcopy(
                            state.zero_route_recovery
                        ),
                        "hessian": payload_by_module[module_name],
                    }
                )
            body = {
                "schema": CAPTURE_FRONTIER_SCHEMA,
                "schema_version": CAPTURE_FRONTIER_SCHEMA_VERSION,
                "payload_hash_algorithm": PAYLOAD_HASH_ALGORITHM,
                "hessian_contract": HESSIAN_CONTRACT,
                "family_join": self.family_join,
                "capture_key": key_body,
                "capture_key_sha256": key_sha256,
                "captures": captures,
            }
            manifest = {
                **body,
                "manifest_sha256": _sha256_bytes(canonical_json_bytes(body)),
            }
            _atomic_json(temporary / MANIFEST_FILENAME, manifest)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.root)
            return manifest
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def discard_through(
        self,
        layer_index: int,
        *,
        block_namespace: str | None = None,
    ) -> None:
        """Remove frontiers made obsolete by one namespace's output boundary."""

        if block_namespace not in {None, "base", "mtp"}:
            raise EXL3CaptureFrontierError(
                "capture-frontier discard namespace is invalid"
            )

        self._prune_incomplete()
        if not self.root.exists():
            return
        changed = False
        for path in self.root.iterdir():
            match = _COMMITTED_DIRECTORY.fullmatch(path.name)
            if match is None or int(match.group("layer")) > layer_index:
                continue
            if block_namespace is not None:
                try:
                    manifest = json.loads(
                        (path / MANIFEST_FILENAME).read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise EXL3CaptureFrontierError(
                        "cannot read capture manifest during scoped discard"
                    ) from error
                key_body = (
                    manifest.get("capture_key")
                    if isinstance(manifest, dict)
                    else None
                )
                key_sha256 = (
                    manifest.get("capture_key_sha256")
                    if isinstance(manifest, dict)
                    else None
                )
                if (
                    not isinstance(key_body, dict)
                    or not isinstance(key_sha256, str)
                    or _sha256_bytes(canonical_json_bytes(key_body))
                    != key_sha256
                ):
                    raise EXL3CaptureFrontierError(
                        "capture key failed validation during scoped discard"
                    )
                manifest = self._read_manifest(
                    path,
                    key_body=key_body,
                    key_sha256=key_sha256,
                )
                module_names = key_body.get("module_names")
                identities = (
                    [routed_expert_identity(name, family_join=self.family_join) for name in module_names]
                    if isinstance(module_names, list)
                    and module_names
                    and all(isinstance(name, str) for name in module_names)
                    else []
                )
                captures = manifest.get("captures")
                capture_modules = (
                    [record.get("module") for record in captures]
                    if isinstance(captures, list)
                    and all(isinstance(record, dict) for record in captures)
                    else []
                )
                namespaces = {
                    identity["block_namespace"]
                    for identity in identities
                    if isinstance(identity, dict)
                }
                logical_layers = {
                    routed_expert_processor_layer(identity, family_join=self.family_join)
                    for identity in identities
                    if isinstance(identity, dict)
                }
                if (
                    len(identities) != len(module_names)
                    or any(identity is None for identity in identities)
                    or len(namespaces) != 1
                    or logical_layers != {key_body.get("layer_index")}
                    or len(capture_modules) != len(module_names)
                    or set(capture_modules) != set(module_names)
                    or len(set(capture_modules)) != len(capture_modules)
                ):
                    raise EXL3CaptureFrontierError(
                        "capture identity failed validation during scoped discard"
                    )
                if namespaces != {block_namespace}:
                    continue
            shutil.rmtree(path)
            changed = True
        if changed:
            _fsync_directory(self.root)


__all__ = [
    "CAPTURE_FRONTIER_CONTRACT",
    "CAPTURE_FRONTIER_ENV",
    "EXL3CaptureDescriptor",
    "EXL3CaptureFrontierError",
    "EXL3CaptureFrontierStore",
    "EXL3CaptureRecord",
    "EXL3CaptureState",
]
