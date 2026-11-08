# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Atomic calibration-batch records used to rebuild EXL3 Hessians."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save as serialize_safetensors
import xxhash


CAPTURE_BATCH_SPOOL_ENV = "GPTQMODEL_EXL3_CAPTURE_BATCH_SPOOL"
CAPTURE_BATCH_CHECKPOINT_INTERVAL_ENV = (
    "GPTQMODEL_EXL3_CAPTURE_BATCH_CHECKPOINT_INTERVAL"
)
CAPTURE_BATCH_SPOOL_SCHEMA = "ds4rt.exl3-capture-batch-spool"
CAPTURE_BATCH_SPOOL_SCHEMA_VERSION = 1
CAPTURE_BATCH_SPOOL_CONTRACT = (
    f"{CAPTURE_BATCH_SPOOL_SCHEMA}-v{CAPTURE_BATCH_SPOOL_SCHEMA_VERSION}"
)
_DIRECTORY = re.compile(
    r"layer-(?P<layer>[0-9]{6})-subset-(?P<subset>[0-9]{4})-"
    r"of-(?P<total>[0-9]{4})-(?P<digest>[0-9a-f]{16})\Z"
)
_BATCH_FILE = re.compile(r"batch-(?P<batch>[0-9]{9})\.safetensors\Z")


class EXL3CaptureBatchSpoolError(RuntimeError):
    """A batch record is incomplete, corrupt, or from another run."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _xxh3_128_file(path: Path) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, body: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json_bytes(body) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
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


class EXL3CaptureBatchSpool:
    """Content-bound rolling records for one layer/subset capture phase."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        layer_index: int,
        subset_index: int,
        subset_total: int,
        expected_batches: int,
        payload_contract: str,
        phase: str,
        module_names: list[str],
        provenance: dict[str, Any],
        ownership: dict[str, str],
        checkpoint_interval: int = 1,
    ) -> None:
        if (
            layer_index < 0
            or not 0 <= subset_index < subset_total
            or expected_batches <= 0
            or not isinstance(payload_contract, str)
            or not payload_contract
            or phase not in {"gate-up", "down"}
            or not module_names
            or len(module_names) != len(set(module_names))
            or not isinstance(provenance, dict)
            or not provenance
            or not isinstance(ownership, dict)
            or isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval <= 0
        ):
            raise EXL3CaptureBatchSpoolError("capture batch identity is invalid")
        self.root = Path(root).expanduser().resolve()
        self.key = {
            "layer_index": int(layer_index),
            "subset_index": int(subset_index),
            "subset_total": int(subset_total),
            "expected_batches": int(expected_batches),
            "payload_contract": payload_contract,
            "phase": phase,
            "module_names": sorted(module_names),
            "provenance": json.loads(json.dumps(provenance, sort_keys=True)),
            "ownership": dict(sorted(ownership.items())),
        }
        self.key_sha256 = _sha256(self.key)
        # Durability cadence is an execution property rather than capture
        # identity. A different cadence may safely resume the same committed
        # batch set without changing any Hessian or route evidence.
        self.checkpoint_interval = int(checkpoint_interval)
        self.directory = self.root / (
            f"layer-{layer_index:06d}-subset-{subset_index:04d}-"
            f"of-{subset_total:04d}-{self.key_sha256[:16]}"
        )
        self._records: dict[int, dict[str, Any]] = {}
        self._pending_records: dict[int, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise EXL3CaptureBatchSpoolError("capture spool root is a symlink")
        self.directory.mkdir(exist_ok=True)
        if self.directory.is_symlink():
            raise EXL3CaptureBatchSpoolError("capture spool entry is a symlink")
        self._restore()
        self._prune_obsolete_identities()

    @property
    def phase(self) -> str:
        return self.key["phase"]

    @property
    def committed_indices(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._records)

    @property
    def pending_indices(self) -> frozenset[int]:
        """Return batches written since the latest durable checkpoint."""

        with self._lock:
            return frozenset(self._pending_records)

    def _progress_body(self) -> dict[str, Any]:
        return {
            "schema": CAPTURE_BATCH_SPOOL_SCHEMA,
            "schema_version": CAPTURE_BATCH_SPOOL_SCHEMA_VERSION,
            "status": "partial",
            "payload_hash_algorithm": "xxh3-128",
            "capture_key": self.key,
            "capture_key_sha256": self.key_sha256,
            "records": [self._records[index] for index in sorted(self._records)],
        }

    def _restore(self) -> None:
        changed = False
        for path in self.directory.iterdir():
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                if not path.is_file() or path.is_symlink():
                    raise EXL3CaptureBatchSpoolError("unsafe incomplete batch record")
                path.unlink()
                changed = True
        progress_path = self.directory / "progress.json"
        if not progress_path.exists():
            unexpected = list(self.directory.iterdir())
            if unexpected:
                raise EXL3CaptureBatchSpoolError(
                    "unmanifested capture records have no progress identity"
                )
            _atomic_json(progress_path, self._progress_body())
            return
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EXL3CaptureBatchSpoolError("cannot read capture progress") from error
        expected = self._progress_body()
        for key in (
            "schema",
            "schema_version",
            "status",
            "payload_hash_algorithm",
            "capture_key",
            "capture_key_sha256",
        ):
            if progress.get(key) != expected[key]:
                raise EXL3CaptureBatchSpoolError("capture progress identity differs")
        records = progress.get("records")
        if not isinstance(records, list):
            raise EXL3CaptureBatchSpoolError("capture progress lacks records")
        expected_files = {"progress.json"}
        for record in records:
            index = record.get("batch_index") if isinstance(record, dict) else None
            relative = record.get("file") if isinstance(record, dict) else None
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.key["expected_batches"]
                or index in self._records
                or relative != f"batch-{index:09d}.safetensors"
                or not isinstance(record.get("tensors"), dict)
                or not isinstance(record.get("metadata"), dict)
            ):
                raise EXL3CaptureBatchSpoolError("capture batch record is malformed")
            path = self.directory / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record.get("bytes")
                or _xxh3_128_file(path) != record.get("xxh3_128")
            ):
                raise EXL3CaptureBatchSpoolError("capture batch payload differs")
            self._records[index] = record
            expected_files.add(relative)
        actual_files = {
            path.name
            for path in self.directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        missing = expected_files - actual_files
        if missing:
            raise EXL3CaptureBatchSpoolError("capture spool file set differs")
        for filename in sorted(actual_files - expected_files):
            match = _BATCH_FILE.fullmatch(filename)
            if (
                match is None
                or int(match.group("batch")) >= self.key["expected_batches"]
            ):
                raise EXL3CaptureBatchSpoolError("capture spool file set differs")
            (self.directory / filename).unlink()
            changed = True
        if changed:
            _fsync_directory(self.directory)

    def _prune_obsolete_identities(self) -> None:
        """Drop scratch records that cannot satisfy the active capture key."""

        changed = False
        for path in self.root.iterdir():
            match = _DIRECTORY.fullmatch(path.name)
            if match is None or path == self.directory:
                continue
            if (
                int(match.group("layer")) != self.key["layer_index"]
                or int(match.group("subset")) != self.key["subset_index"]
                or int(match.group("total")) != self.key["subset_total"]
            ):
                continue
            if not path.is_dir() or path.is_symlink():
                raise EXL3CaptureBatchSpoolError(
                    "obsolete capture spool identity is unsafe"
                )
            shutil.rmtree(path)
            changed = True
        if changed:
            _fsync_directory(self.root)

    def commit(
        self,
        batch_index: int,
        *,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if batch_index in self._records:
                return self._records[batch_index]
            if batch_index in self._pending_records:
                return self._pending_records[batch_index]
            if (
                isinstance(batch_index, bool)
                or not isinstance(batch_index, int)
                or not 0 <= batch_index < self.key["expected_batches"]
                or not tensors
                or not all(
                    isinstance(name, str)
                    and name
                    and isinstance(tensor, torch.Tensor)
                    for name, tensor in tensors.items()
                )
                or not isinstance(metadata, dict)
            ):
                raise EXL3CaptureBatchSpoolError("capture batch payload is invalid")
            host = {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in tensors.items()
            }
            payload = serialize_safetensors(host)
            filename = f"batch-{batch_index:09d}.safetensors"
            destination = self.directory / filename
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=self.directory
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as target:
                    target.write(payload)
                    target.flush()
                os.replace(temporary, destination)
                record = {
                    "batch_index": batch_index,
                    "file": filename,
                    "bytes": len(payload),
                    "xxh3_128": xxhash.xxh3_128_hexdigest(payload),
                    "tensors": {
                        name: _tensor_spec(tensor)
                        for name, tensor in sorted(host.items())
                    },
                    "metadata": json.loads(json.dumps(metadata, sort_keys=True)),
                }
                self._pending_records[batch_index] = record
                if (
                    len(self._pending_records) >= self.checkpoint_interval
                    or len(self._records) + len(self._pending_records)
                    == self.key["expected_batches"]
                ):
                    self.checkpoint()
                return record
            except BaseException:
                if temporary.exists():
                    temporary.unlink()
                if destination.exists() and batch_index not in self._records:
                    destination.unlink()
                raise

    def checkpoint(self) -> None:
        """Durably publish the bounded group of newly written batch files."""

        with self._lock:
            if not self._pending_records:
                return
            pending = [
                self._pending_records[index]
                for index in sorted(self._pending_records)
            ]
            # The payloads were written and renamed before this point. Flush
            # the bounded group before publishing any of it in progress.json;
            # an interruption before the manifest commit simply causes the
            # unmanifested files to be discarded and those batches replayed.
            for record in pending:
                descriptor = os.open(
                    self.directory / record["file"],
                    os.O_RDONLY | os.O_NOFOLLOW,
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _fsync_directory(self.directory)

            previous = dict(self._records)
            self._records.update(self._pending_records)
            try:
                _atomic_json(
                    self.directory / "progress.json", self._progress_body()
                )
            except BaseException:
                self._records = previous
                raise
            self._pending_records.clear()

    def load(self, batch_index: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        with self._lock:
            record = self._records.get(batch_index) or self._pending_records.get(
                batch_index
            )
            if record is None:
                raise KeyError(batch_index)
            path = self.directory / record["file"]
            tensors: dict[str, torch.Tensor] = {}
            with safe_open(path, framework="pt", device="cpu") as source:
                if set(source.keys()) != set(record["tensors"]):
                    raise EXL3CaptureBatchSpoolError("capture tensor set differs")
                for name in source.keys():
                    tensor = source.get_tensor(name)
                    if _tensor_spec(tensor) != record["tensors"][name]:
                        raise EXL3CaptureBatchSpoolError(
                            "capture tensor geometry differs"
                        )
                    tensors[name] = tensor
            return tensors, json.loads(json.dumps(record["metadata"]))

    def discard(self) -> None:
        with self._lock:
            if self.directory.exists():
                shutil.rmtree(self.directory)
                _fsync_directory(self.root)
            self._records.clear()
            self._pending_records.clear()


__all__ = [
    "CAPTURE_BATCH_SPOOL_CONTRACT",
    "CAPTURE_BATCH_SPOOL_ENV",
    "CAPTURE_BATCH_CHECKPOINT_INTERVAL_ENV",
    "EXL3CaptureBatchSpool",
    "EXL3CaptureBatchSpoolError",
]
