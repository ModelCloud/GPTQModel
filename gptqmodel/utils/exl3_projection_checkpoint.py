# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Atomic, content-addressed checkpoints for completed EXL3 projections."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors


CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_CONTRACT = f"{CHECKPOINT_SCHEMA}-v{CHECKPOINT_SCHEMA_VERSION}"


def _finite_json_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite EXL3 checkpoint value at {path}: {value}")
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
    raise TypeError(
        f"unsupported EXL3 checkpoint value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    """Hash logical tensor bytes together with their exact shape and dtype."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("EXL3 checkpoint identities require torch tensors")
    host = tensor.detach().contiguous().to(device="cpu")
    byte_view = host.reshape(-1).view(torch.uint8).numpy()
    digest = hashlib.sha256(memoryview(byte_view)).hexdigest()
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "numel": host.numel(),
        "bytes": host.numel() * host.element_size(),
        "sha256": digest,
    }


def build_projection_request(
    *,
    module_full_name: str,
    layer_index: int | None,
    input_weight: torch.Tensor,
    hessian: torch.Tensor,
    sample_count: int,
    quantizer_contract: dict[str, Any],
    family_join: dict[str, Any] | None,
    route_evidence: dict[str, Any] | None,
    zero_route_recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every input capable of changing one packed projection result."""

    request = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "module": module_full_name,
        "processor_layer_index": layer_index,
        "sample_count": int(sample_count),
        "input_weight": tensor_identity(input_weight),
        "hessian": tensor_identity(hessian),
        "quantizer_contract": deepcopy(quantizer_contract),
        "family_join": deepcopy(family_join),
        "route_evidence": deepcopy(route_evidence),
    }
    # Preserve every existing natural-route request identity byte-for-byte.
    # Recovery is additive and enters the request hash only for a projection
    # whose natural route count was exactly zero.
    if zero_route_recovery is not None:
        request["zero_route_recovery"] = deepcopy(zero_route_recovery)
    clean = _finite_json_value(request, "request")
    clean["request_sha256"] = sha256_bytes(canonical_json_bytes(clean))
    return clean


def checkpoint_root_from_provenance(
    provenance: dict[str, Any] | None,
) -> Path | None:
    """Resolve the opt-in production checkpoint root from run provenance."""

    if not isinstance(provenance, dict):
        return None
    run = provenance.get("run")
    if not isinstance(run, dict):
        return None
    checkpoint = run.get("projection_checkpoint")
    if checkpoint is None:
        return None
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("contract") != CHECKPOINT_CONTRACT
        or not isinstance(checkpoint.get("root"), str)
        or not checkpoint["root"]
    ):
        raise ValueError("invalid EXL3 projection-checkpoint run contract")
    return Path(checkpoint["root"]).expanduser().resolve()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            # Projection checkpoints are model artifacts, not secrets. Make
            # an atomically published checkpoint readable by an unprivileged
            # host validator even when quantization runs as root in Docker.
            os.fchmod(target.fileno(), 0o644)
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EXL3ProjectionCheckpointStore:
    """Publish and validate immutable packed projection results."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._module_request_lock = threading.Lock()
        self._module_requests: dict[tuple[str, str], str] | None = None

    @staticmethod
    def _module_request_key(request: dict[str, Any]) -> tuple[str, str]:
        """Return the immutable reservation key for uniform or inline tiers."""

        module = request.get("module")
        contract = request.get("quantizer_contract")
        inline = contract.get("inline_mixed") if isinstance(contract, dict) else None
        if inline is None:
            return str(module), "uniform"
        role = inline.get("role") if isinstance(inline, dict) else None
        policy_sha256 = inline.get("policy_sha256") if isinstance(inline, dict) else None
        bits = contract.get("bits")
        base_bits = inline.get("base_bits")
        upgrade_bits = inline.get("upgrade_bits")
        candidate_role = f"candidate_k{base_bits}"
        selected_role = f"selected_k{upgrade_bits}"
        if (
            not isinstance(base_bits, int)
            or isinstance(base_bits, bool)
            or not isinstance(upgrade_bits, int)
            or isinstance(upgrade_bits, bool)
            or upgrade_bits != base_bits + 1
            or role not in {candidate_role, selected_role}
            or not isinstance(policy_sha256, str)
            or len(policy_sha256) != 64
            or bits != (base_bits if role == candidate_role else upgrade_bits)
        ):
            raise ValueError("EXL3 inline-mixed checkpoint tier is inconsistent")
        if role == selected_role and any(
            not isinstance(inline.get(field), str) or len(inline[field]) != 64
            for field in ("candidate_request_sha256", "tier_plan_sha256")
        ):
            raise ValueError(
                "EXL3 selected-tier checkpoint lacks its candidate/tier-plan binding"
            )
        return str(module), role

    def reserve_module_request(self, request: dict[str, Any]) -> None:
        """Fail before work when one immutable module acquires a new identity.

        Request hashes bind the weight, Hessian, route evidence, quantizer
        contract, and run family.  A second hash for the same logical module is
        therefore recovery drift, not another valid checkpoint.  Reservations
        also close the race between concurrent projection workers before either
        result is committed.

        This check is explicit because remote-worker checkpoint stores are
        bounded scratch caches and may legitimately recycle one module name
        across independent coordinator runs.  The coordinator calls it on its
        run-scoped, shared store.
        """

        module = request.get("module")
        request_sha256 = request.get("request_sha256")
        request_body = {
            key: value for key, value in request.items() if key != "request_sha256"
        }
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(request_sha256, str)
            or sha256_bytes(canonical_json_bytes(request_body)) != request_sha256
        ):
            raise ValueError("invalid EXL3 module checkpoint request")
        self._paths(request_sha256)

        with self._module_request_lock:
            if self._module_requests is None:
                module_requests: dict[tuple[str, str], str] = {}
                module_modes: dict[str, str] = {}
                for committed_request, _result in self.inspect_committed_manifests():
                    committed_module = committed_request.get("module")
                    committed_sha256 = committed_request.get("request_sha256")
                    if (
                        not isinstance(committed_module, str)
                        or not committed_module
                        or not isinstance(committed_sha256, str)
                    ):
                        raise ValueError(
                            "EXL3 projection checkpoint has no module identity"
                        )
                    key = self._module_request_key(committed_request)
                    mode = "uniform" if key[1] == "uniform" else "inline_mixed"
                    previous_mode = module_modes.setdefault(committed_module, mode)
                    if previous_mode != mode:
                        raise ValueError(
                            "EXL3 projection checkpoint mixes uniform and inline "
                            f"requests for `{committed_module}`"
                        )
                    previous = module_requests.setdefault(key, committed_sha256)
                    if previous != committed_sha256:
                        raise ValueError(
                            "EXL3 projection checkpoint contains immutable module "
                            f"request drift for `{committed_module}`: "
                            f"{previous} != {committed_sha256}"
                        )
                self._module_requests = module_requests

            key = self._module_request_key(request)
            inline = request.get("quantizer_contract", {}).get("inline_mixed")
            base_bits = inline.get("base_bits") if isinstance(inline, dict) else None
            candidate_role = f"candidate_k{base_bits}"
            selected_role = (
                f"selected_k{inline.get('upgrade_bits')}"
                if isinstance(inline, dict)
                else None
            )
            if key[1] == selected_role:
                candidate_digest = inline["candidate_request_sha256"]
                if (
                    self._module_requests.get((module, candidate_role))
                    != candidate_digest
                ):
                    raise ValueError(
                        "EXL3 selected tier is not bound to the reserved "
                        f"candidate for `{module}`"
                    )
            conflicting = [
                existing
                for existing in self._module_requests
                if existing[0] == module
                and ((existing[1] == "uniform") != (key[1] == "uniform"))
            ]
            if conflicting:
                raise ValueError(
                    "EXL3 projection checkpoint cannot mix uniform and inline "
                    f"requests for `{module}`"
                )
            previous = self._module_requests.setdefault(key, request_sha256)
            if previous != request_sha256:
                raise ValueError(
                    "EXL3 immutable module request drift for "
                    f"`{module}`: {previous} != {request_sha256}"
                )

    def _paths(self, request_sha256: str) -> tuple[Path, Path]:
        if len(request_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in request_sha256
        ):
            raise ValueError("invalid EXL3 checkpoint request digest")
        prefix = self.root / request_sha256[:2] / request_sha256[2:4]
        return (
            prefix / f"{request_sha256}.json",
            prefix / f"{request_sha256}.safetensors",
        )

    def committed_tensor_path(self, request_sha256: str) -> Path:
        """Return the regular tensor path for an already-validated checkpoint."""

        _manifest_path, tensor_path = self._paths(request_sha256)
        if not tensor_path.is_file() or tensor_path.is_symlink():
            raise ValueError("EXL3 checkpoint tensor path is not a regular file")
        return tensor_path.resolve(strict=True)

    def inspect_committed_manifests(
        self,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Inspect every committed request/result without reading tensor payloads.

        This is the inexpensive discovery half of layer-boundary catch-up.  It
        authenticates the directory layout, manifest, request, and declared
        tensor identity, but deliberately does not hash or decode the packed
        tensor file.  A caller must use :meth:`load_committed` before trusting
        or installing any discovered result.
        """

        if not self.root.exists():
            return
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("EXL3 checkpoint root is not a regular directory")

        hex_chars = frozenset("0123456789abcdef")
        seen_manifests: dict[str, Path] = {}
        seen_tensors: dict[str, Path] = {}
        for first in self.root.iterdir():
            if (
                not first.is_dir()
                or first.is_symlink()
                or len(first.name) != 2
                or any(char not in hex_chars for char in first.name)
            ):
                raise ValueError("EXL3 checkpoint root contains an unsafe entry")
            for second in first.iterdir():
                if (
                    not second.is_dir()
                    or second.is_symlink()
                    or len(second.name) != 2
                    or any(char not in hex_chars for char in second.name)
                ):
                    raise ValueError("EXL3 checkpoint root contains an unsafe prefix")
                for path in second.iterdir():
                    if not path.is_file() or path.is_symlink():
                        raise ValueError("EXL3 checkpoint contains a non-regular file")
                    if path.suffix not in {".json", ".safetensors"}:
                        raise ValueError("EXL3 checkpoint contains an unexpected file")
                    request_sha256 = path.name.removesuffix(path.suffix)
                    self._paths(request_sha256)
                    if (
                        request_sha256[:2] != first.name
                        or request_sha256[2:4] != second.name
                    ):
                        raise ValueError("EXL3 checkpoint file is under the wrong prefix")
                    target = (
                        seen_manifests
                        if path.suffix == ".json"
                        else seen_tensors
                    )
                    if request_sha256 in target:
                        raise ValueError("EXL3 checkpoint contains a duplicate file")
                    target[request_sha256] = path

        if set(seen_manifests) != set(seen_tensors):
            raise ValueError("EXL3 checkpoint contains an incomplete committed pair")

        for request_sha256 in sorted(seen_manifests):
            manifest_path = seen_manifests[request_sha256]
            tensor_path = seen_tensors[request_sha256]
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("cannot inspect EXL3 projection checkpoint") from error
            if not isinstance(manifest, dict):
                raise ValueError("EXL3 projection checkpoint manifest is not an object")
            manifest_digest = manifest.get("manifest_sha256")
            manifest_body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            request = manifest.get("request")
            request_body = (
                {
                    key: value
                    for key, value in request.items()
                    if key != "request_sha256"
                }
                if isinstance(request, dict)
                else None
            )
            tensor_sha256 = manifest.get("tensor_sha256")
            tensor_specs = manifest.get("tensors")
            result = manifest.get("result")
            if (
                manifest.get("schema") != CHECKPOINT_SCHEMA
                or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or manifest.get("request_sha256") != request_sha256
                or not isinstance(request, dict)
                or request.get("request_sha256") != request_sha256
                or sha256_bytes(canonical_json_bytes(request_body))
                != request_sha256
                or manifest.get("tensor_file") != tensor_path.name
                or not isinstance(tensor_sha256, str)
                or len(tensor_sha256) != 64
                or any(char not in hex_chars for char in tensor_sha256)
                or not isinstance(tensor_specs, dict)
                or not tensor_specs
                or not isinstance(result, dict)
                or manifest_digest
                != sha256_bytes(canonical_json_bytes(manifest_body))
            ):
                raise ValueError("EXL3 projection checkpoint failed manifest validation")
            yield deepcopy(request), deepcopy(result)

    def load_committed(
        self,
        request_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]] | None:
        """Load one self-authenticating checkpoint without rebuilding its Hessian.

        A completed layer-boundary checkpoint records the request digest for
        every projection in that layer.  Resuming from that boundary must be
        able to restore the packed modules without replaying the calibration
        corpus merely to recreate the request object.  The stored request is
        still fully validated here, including its own digest, before any tensor
        is returned.
        """

        manifest_path, tensor_path = self._paths(str(request_sha256))
        if not manifest_path.exists():
            return None
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or not tensor_path.is_file()
            or tensor_path.is_symlink()
        ):
            raise ValueError("EXL3 checkpoint contains a non-regular committed path")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read EXL3 projection checkpoint") from error
        if not isinstance(manifest, dict):
            raise ValueError("EXL3 projection checkpoint manifest is not an object")
        manifest_digest = manifest.get("manifest_sha256")
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        stored_request = manifest.get("request")
        stored_request_body = (
            {
                key: value
                for key, value in stored_request.items()
                if key != "request_sha256"
            }
            if isinstance(stored_request, dict)
            else None
        )
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("request_sha256") != request_sha256
            or not isinstance(stored_request, dict)
            or stored_request.get("request_sha256") != request_sha256
            or sha256_bytes(canonical_json_bytes(stored_request_body))
            != request_sha256
            or manifest.get("tensor_file") != tensor_path.name
            or manifest_digest != sha256_bytes(canonical_json_bytes(manifest_body))
            or manifest.get("tensor_sha256") != sha256_file(tensor_path)
        ):
            raise ValueError("EXL3 projection checkpoint failed content validation")
        payload = tensor_path.read_bytes()
        try:
            tensors = load_safetensors(payload)
        except Exception as error:
            raise ValueError(
                "cannot decode EXL3 projection checkpoint tensors"
            ) from error
        tensor_specs = manifest.get("tensors")
        if not isinstance(tensor_specs, dict) or set(tensor_specs) != set(tensors):
            raise ValueError("EXL3 projection checkpoint tensor set is inconsistent")
        for name, tensor in tensors.items():
            if tensor_specs[name] != tensor_identity(tensor):
                raise ValueError(
                    f"EXL3 projection checkpoint tensor `{name}` is inconsistent"
                )
        result = manifest.get("result")
        if not isinstance(result, dict):
            raise ValueError("EXL3 projection checkpoint result is malformed")
        return stored_request, tensors, result

    def load(
        self,
        request: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
        request_sha256 = request.get("request_sha256")
        loaded = self.load_committed(str(request_sha256))
        if loaded is None:
            return None
        stored_request, tensors, result = loaded
        if stored_request != request:
            raise ValueError("EXL3 projection checkpoint request is inconsistent")
        return tensors, result

    def commit(
        self,
        request: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        request_sha256 = request.get("request_sha256")
        manifest_path, tensor_path = self._paths(str(request_sha256))
        existing = self.load(request)
        if existing is not None:
            existing_tensors, existing_result = existing
            if existing_result != result or any(
                not torch.equal(existing_tensors[name], tensor.detach().cpu())
                for name, tensor in tensors.items()
            ):
                raise ValueError("EXL3 projection checkpoint collision")
            return existing_result

        host_tensors = {
            str(name): tensor.detach().contiguous().to(device="cpu")
            for name, tensor in tensors.items()
        }
        if not host_tensors:
            raise ValueError("EXL3 projection checkpoint cannot commit no tensors")
        tensor_payload = save_safetensors(host_tensors)
        clean_result = _finite_json_value(result, "result")
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "request": deepcopy(request),
            "request_sha256": request_sha256,
            "tensor_file": tensor_path.name,
            "tensor_sha256": sha256_bytes(tensor_payload),
            "tensors": {
                name: tensor_identity(tensor) for name, tensor in host_tensors.items()
            },
            "result": clean_result,
        }
        manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
        _atomic_write(tensor_path, tensor_payload)
        _atomic_write(manifest_path, canonical_json_bytes(manifest) + b"\n")
        _fsync_directory(manifest_path.parent)
        loaded = self.load(request)
        if loaded is None or loaded[1] != clean_result:
            raise RuntimeError("EXL3 projection checkpoint did not commit durably")
        return clean_result

    def prune(
        self,
        *,
        max_entries: int,
        preserve_request_sha256: Iterable[str] = (),
    ) -> dict[str, int]:
        """Bound packed-result scratch without touching live request payloads.

        Remote workers serialize projection execution, so retaining the current
        request plus the newest preceding request preserves the coordinator's
        commit/retry handoff while preventing a full model run from accumulating
        one checkpoint per projection. Coordinator stores never call this method.
        """

        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("EXL3 checkpoint retention must be positive")
        preserve = {str(value) for value in preserve_request_sha256}
        for request_sha256 in preserve:
            self._paths(request_sha256)
        if len(preserve) > max_entries:
            raise ValueError(
                "EXL3 checkpoint retention cannot preserve too many entries"
            )
        if not self.root.exists():
            if preserve:
                raise ValueError("preserved EXL3 checkpoint does not exist")
            return {
                "retained_entries": 0,
                "removed_entries": 0,
                "removed_bytes": 0,
                "removed_orphans": 0,
            }
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("EXL3 checkpoint root is not a regular directory")

        entries: list[tuple[int, str, Path, Path]] = []
        orphan_tensors: list[Path] = []
        seen_manifests: set[str] = set()
        seen_tensors: set[str] = set()
        hex_chars = frozenset("0123456789abcdef")
        for first in self.root.iterdir():
            if (
                not first.is_dir()
                or first.is_symlink()
                or len(first.name) != 2
                or any(char not in hex_chars for char in first.name)
            ):
                raise ValueError("EXL3 checkpoint root contains an unsafe entry")
            for second in first.iterdir():
                if (
                    not second.is_dir()
                    or second.is_symlink()
                    or len(second.name) != 2
                    or any(char not in hex_chars for char in second.name)
                ):
                    raise ValueError("EXL3 checkpoint root contains an unsafe prefix")
                for path in second.iterdir():
                    if not path.is_file() or path.is_symlink():
                        raise ValueError("EXL3 checkpoint contains a non-regular file")
                    if path.suffix not in {".json", ".safetensors"}:
                        raise ValueError("EXL3 checkpoint contains an unexpected file")
                    request_sha256 = path.name.removesuffix(path.suffix)
                    self._paths(request_sha256)
                    if (
                        request_sha256[:2] != first.name
                        or request_sha256[2:4] != second.name
                    ):
                        raise ValueError(
                            "EXL3 checkpoint file is under the wrong prefix"
                        )
                    target = (
                        seen_manifests if path.suffix == ".json" else seen_tensors
                    )
                    if request_sha256 in target:
                        raise ValueError("EXL3 checkpoint contains a duplicate file")
                    target.add(request_sha256)

        for request_sha256 in sorted(seen_manifests):
            manifest_path, tensor_path = self._paths(request_sha256)
            if request_sha256 not in seen_tensors:
                raise ValueError("EXL3 checkpoint manifest has no tensor payload")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "cannot inspect EXL3 checkpoint retention entry"
                ) from error
            request = manifest.get("request") if isinstance(manifest, dict) else None
            if not isinstance(request, dict) or self.load(request) is None:
                raise ValueError("EXL3 checkpoint retention entry is invalid")
            entries.append(
                (
                    manifest_path.stat().st_mtime_ns,
                    request_sha256,
                    manifest_path,
                    tensor_path,
                )
            )
        for request_sha256 in sorted(seen_tensors - seen_manifests):
            orphan_tensors.append(self._paths(request_sha256)[1])

        available = {entry[1] for entry in entries}
        missing_preserved = preserve - available
        if missing_preserved:
            raise ValueError("preserved EXL3 checkpoint is not committed")
        keep = set(preserve)
        for _mtime_ns, request_sha256, _manifest_path, _tensor_path in sorted(
            entries, reverse=True
        ):
            if len(keep) >= max_entries:
                break
            keep.add(request_sha256)

        removed_entries = 0
        removed_bytes = 0
        touched_directories: set[Path] = set()
        for _mtime_ns, request_sha256, manifest_path, tensor_path in entries:
            if request_sha256 in keep:
                continue
            removed_bytes += manifest_path.stat().st_size + tensor_path.stat().st_size
            # Removing the manifest first makes an interrupted prune look like
            # an uncommitted tensor orphan, never a committed corrupt result.
            manifest_path.unlink()
            _fsync_directory(manifest_path.parent)
            tensor_path.unlink()
            touched_directories.add(manifest_path.parent)
            removed_entries += 1
        for tensor_path in orphan_tensors:
            removed_bytes += tensor_path.stat().st_size
            tensor_path.unlink()
            touched_directories.add(tensor_path.parent)
        for directory in sorted(
            touched_directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
            if not any(directory.iterdir()):
                directory.rmdir()
                parent = directory.parent
                if parent != self.root and not any(parent.iterdir()):
                    parent.rmdir()
        _fsync_directory(self.root)
        return {
            "retained_entries": len(keep),
            "removed_entries": removed_entries,
            "removed_bytes": removed_bytes,
            "removed_orphans": len(orphan_tensors),
        }


__all__ = [
    "CHECKPOINT_CONTRACT",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXL3ProjectionCheckpointStore",
    "build_projection_request",
    "canonical_json_bytes",
    "checkpoint_root_from_provenance",
    "tensor_identity",
]
