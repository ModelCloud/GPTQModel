# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file


@dataclass
class InputCache:
    """Stores captured layer inputs and per-batch kwargs for replayed forwards."""

    layer_inputs: List[List[torch.Tensor]]
    layer_input_kwargs: List[Dict[str, torch.Tensor]]
    position_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    src_inputs: Optional[List[List[torch.Tensor]]] = None

    def __post_init__(self):
        if self.src_inputs is None:
            self.src_inputs = list(self.layer_inputs) if self.layer_inputs is not None else []

    def module_kwargs(self):
        """Returns the replay kwargs that are shared across cached module calls."""

        result = {}
        result["position_ids"] = self.position_ids
        result["attention_masks"] = self.attention_masks
        return result

    def cleanup(self):
        """Release captured tensors so a per-layer caller can reclaim them.

        Staged GSQ keeps one captured input cache alive per decoder block.
        ``quantize_llama_gsq_model`` releases each cache in a ``finally`` block
        once its block is written, but only for caches that expose
        ``cleanup``. Without this method the guard silently matched nothing and
        GPU-resident captures accumulated for the whole model: at paper scale
        one block's capture is ~69 GB, so the run exhausted a 287 GiB device
        part-way through the blocks even though each block in isolation fits.
        """

        self.layer_inputs = []
        self.layer_input_kwargs = []
        self.position_ids = []
        self.attention_masks = []
        self.src_inputs = []


class TensorLifetimeDiagnostic:
    """Track tensor lifetime without retaining the tensors being observed."""

    def __init__(self, value: Any):
        self._issued_tensor_refs: list[
            tuple[weakref.ReferenceType[torch.Tensor], int]
        ] = []
        seen: set[int] = set()

        def visit(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                identity = id(item)
                if identity not in seen:
                    seen.add(identity)
                    self.observe(item)
                return
            if isinstance(item, dict):
                for nested in item.values():
                    visit(nested)
                return
            if isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)

    def observe(self, tensor: torch.Tensor) -> None:
        """Add one tensor without taking a strong ownership reference."""

        self._issued_tensor_refs.append(
            (weakref.ref(tensor), tensor.untyped_storage().nbytes())
        )

    @staticmethod
    def _referrer_summary(value: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {"type": type(value).__name__}
        if isinstance(value, (list, tuple, set, dict)):
            summary["length"] = len(value)
        if isinstance(value, dict):
            summary["string_keys"] = sorted(
                key for key in value if isinstance(key, str)
            )[:12]
        return summary

    def lifetime_diagnostic(self) -> dict[str, Any]:
        """Describe tracked tensors that survived until the current boundary."""

        import gc

        alive_ids: set[int] = set()
        alive_storage_bytes = 0
        sample_ref: weakref.ReferenceType[torch.Tensor] | None = None
        sample_bytes = -1
        for tensor_ref, storage_bytes in self._issued_tensor_refs:
            tensor = tensor_ref()
            if tensor is None or id(tensor) in alive_ids:
                continue
            alive_ids.add(id(tensor))
            alive_storage_bytes += storage_bytes
            if storage_bytes > sample_bytes:
                sample_ref = tensor_ref
                sample_bytes = storage_bytes
        # Do not let the loop local become another apparent owner.
        tensor = None
        result: dict[str, Any] = {
            "issued": len(self._issued_tensor_refs),
            "alive_tensors": len(alive_ids),
            "alive_storage_bytes": alive_storage_bytes,
        }
        if sample_ref is None:
            return result

        sample = sample_ref()
        if sample is None:
            return result
        ignored = {id(result), id(alive_ids)}
        direct = [
            value for value in gc.get_referrers(sample) if id(value) not in ignored
        ]
        result["sample_shape"] = list(sample.shape)
        result["sample_direct_referrers"] = [
            self._referrer_summary(value) for value in direct[:12]
        ]
        parents: list[dict[str, Any]] = []
        for value in direct[:4]:
            for parent in gc.get_referrers(value):
                if id(parent) in ignored or parent is direct:
                    continue
                parents.append(self._referrer_summary(parent))
                if len(parents) == 12:
                    break
            if len(parents) == 12:
                break
        result["sample_parent_referrers"] = parents
        return result


_DISK_BATCH_SCHEMA = "gptqmodel.disk-backed-layer-outputs"
_DISK_BATCH_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _xxh3_128_file(path: Path) -> str:
    # Keep the optional crash-recovery hash dependency out of GPTQModel's
    # ordinary import path.  DeepSeek V4 quantization images provide xxhash.
    import xxhash

    digest = xxhash.xxh3_128()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary = Path(output.name)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DiskBackedLayerOutputSequence(Sequence[List[torch.Tensor]]):
    """Random-access tensor batches stored in content-checked shards."""

    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.manifest = manifest
        self._shards = list(manifest["shards"])
        self._batch_count = int(manifest["batch_count"])
        self._shard_batches = int(manifest["shard_batches"])
        self._row_counts = [0] * self._batch_count
        for shard in self._shards:
            start = int(shard["start"])
            for offset, shape in enumerate(shard["shapes"]):
                self._row_counts[start + offset] = int(shape[0])
        if any(count <= 0 for count in self._row_counts):
            raise ValueError("disk-backed layer output manifest has invalid rows")
        self._issued_tensor_refs: list[
            tuple[weakref.ReferenceType[torch.Tensor], int]
        ] = []

    @classmethod
    def open(cls, root: str | os.PathLike, *, verify_hashes: bool = True):
        root = Path(root).resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read disk-backed layer output manifest") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != _DISK_BATCH_SCHEMA
            or manifest.get("schema_version") != _DISK_BATCH_SCHEMA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("hash_algorithm") != "xxh3-128"
            or not isinstance(manifest.get("shards"), list)
        ):
            raise ValueError("disk-backed layer output manifest is malformed")
        digest = manifest.get("manifest_sha256")
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        import hashlib

        if digest != hashlib.sha256(_canonical_json(body)).hexdigest():
            raise ValueError("disk-backed layer output manifest digest differs")
        expected_start = 0
        for shard in manifest["shards"]:
            path = root / shard.get("path", "")
            shapes = shard.get("shapes")
            if (
                not path.is_file()
                or path.is_symlink()
                or shard.get("start") != expected_start
                or not isinstance(shapes, list)
                or not shapes
            ):
                raise ValueError("disk-backed layer output shard set is malformed")
            expected_start += len(shapes)
            if verify_hashes and _xxh3_128_file(path) != shard.get("xxh3_128"):
                raise ValueError("disk-backed layer output shard hash differs")
        if expected_start != manifest.get("batch_count"):
            raise ValueError("disk-backed layer output coverage is incomplete")
        return cls(root, manifest)

    def __len__(self) -> int:
        return self._batch_count

    @property
    def row_counts(self) -> list[int]:
        return list(self._row_counts)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = index // self._shard_batches
        shard = self._shards[shard_index]
        path = self.root / shard["path"]
        with safe_open(path, framework="pt", device="cpu") as source:
            tensor = source.get_tensor(
                "hidden"
                if self._shard_batches == 1
                else f"batch_{index:09d}"
            )
        expected_shape = tuple(shard["shapes"][index - int(shard["start"])])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError("disk-backed layer output tensor shape differs")
        self._issued_tensor_refs.append(
            (weakref.ref(tensor), tensor.untyped_storage().nbytes())
        )
        return [tensor]

    def lifetime_diagnostic(self) -> dict[str, Any]:
        """Describe tensors issued from disk that survived the layer handoff."""

        diagnostic = TensorLifetimeDiagnostic.__new__(TensorLifetimeDiagnostic)
        diagnostic._issued_tensor_refs = self._issued_tensor_refs
        return diagnostic.lifetime_diagnostic()


class DiskBackedLayerOutputWriter:
    """Bounded, crash-consistent collector for a layer's replay outputs."""

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        layer_index: int,
        expected_batches: int,
        provenance: dict[str, Any],
        shard_batches: int = 128,
        on_finalize: Callable[["DiskBackedLayerOutputSequence"], None] | None = None,
    ) -> None:
        if layer_index < 0 or expected_batches <= 0 or shard_batches <= 0:
            raise ValueError("disk-backed output geometry must be positive")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("disk-backed outputs require provenance")
        if on_finalize is not None and not callable(on_finalize):
            raise TypeError("disk-backed output finalizer must be callable")
        self.parent = Path(root).expanduser().resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.destination = self.parent / f"layer-{layer_index:06d}"
        self.temporary = self.parent / f".layer-{layer_index:06d}.partial"
        self.layer_index = int(layer_index)
        self.expected_batches = int(expected_batches)
        self.provenance = json.loads(json.dumps(provenance, sort_keys=True))
        self.shard_batches = int(shard_batches)
        self._pending: dict[int, dict[int, torch.Tensor]] = {}
        self._shards: dict[int, dict[str, Any]] = {}
        self._committed_indices: set[int] = set()
        self._lock = threading.RLock()
        self._closed = False
        self._complete: DiskBackedLayerOutputSequence | None = None
        self._on_finalize = on_finalize
        self._on_finalize_called = False

        if self.destination.exists():
            complete = DiskBackedLayerOutputSequence.open(
                self.destination, verify_hashes=True
            )
            manifest = complete.manifest
            if (
                manifest.get("layer_index") != self.layer_index
                or manifest.get("batch_count") != self.expected_batches
                or manifest.get("shard_batches") != self.shard_batches
                or manifest.get("provenance") != self.provenance
            ):
                raise ValueError("completed disk-backed layer output identity differs")
            self._complete = complete
            self._committed_indices.update(range(self.expected_batches))
            return

        self.temporary.mkdir(exist_ok=True)
        progress_path = self.temporary / "progress.json"
        if progress_path.exists():
            self._restore_progress(progress_path)
        else:
            self._write_progress()

    def __len__(self) -> int:
        return self.expected_batches

    @property
    def row_counts(self) -> list[int]:
        if self._complete is None:
            raise RuntimeError("disk-backed outputs are not finalized")
        return self._complete.row_counts

    @property
    def committed_indices(self) -> frozenset[int]:
        """Batches already durable and safe for the executor to skip."""

        with self._lock:
            return frozenset(self._committed_indices)

    def _progress_body(self) -> dict[str, Any]:
        return {
            "schema": _DISK_BATCH_SCHEMA,
            "schema_version": _DISK_BATCH_SCHEMA_VERSION,
            "status": "partial",
            "hash_algorithm": "xxh3-128",
            "layer_index": self.layer_index,
            "batch_count": self.expected_batches,
            "shard_batches": self.shard_batches,
            "provenance": self.provenance,
            "shards": [self._shards[index] for index in sorted(self._shards)],
        }

    def _write_progress(self) -> None:
        _atomic_json(self.temporary / "progress.json", self._progress_body())
        _fsync_directory(self.temporary)

    def _restore_progress(self, path: Path) -> None:
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read disk-backed output progress") from error
        expected = self._progress_body()
        for key in (
            "schema",
            "schema_version",
            "status",
            "hash_algorithm",
            "layer_index",
            "batch_count",
            "shard_batches",
            "provenance",
        ):
            if progress.get(key) != expected[key]:
                raise ValueError("disk-backed output progress identity differs")
        shards = progress.get("shards")
        if not isinstance(shards, list):
            raise ValueError("disk-backed output progress has no shard list")
        for shard in shards:
            start = shard.get("start")
            shapes = shard.get("shapes")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or start % self.shard_batches
                or not isinstance(shapes, list)
                or not shapes
            ):
                raise ValueError("disk-backed output progress shard is malformed")
            shard_index = start // self.shard_batches
            expected_count = min(
                self.shard_batches, self.expected_batches - start
            )
            file_path = self.temporary / shard.get("path", "")
            if (
                len(shapes) != expected_count
                or not file_path.is_file()
                or _xxh3_128_file(file_path) != shard.get("xxh3_128")
                or shard_index in self._shards
            ):
                raise ValueError("disk-backed output progress shard differs")
            self._shards[shard_index] = shard
            self._committed_indices.update(range(start, start + expected_count))

    def _flush_shard(self, shard_index: int) -> None:
        pending = self._pending.get(shard_index)
        if not pending:
            return
        start = shard_index * self.shard_batches
        count = min(self.shard_batches, self.expected_batches - start)
        expected_indices = set(range(start, start + count))
        if set(pending) != expected_indices:
            return
        filename = f"outputs-{shard_index:06d}.safetensors"
        final_path = self.temporary / filename
        with tempfile.NamedTemporaryFile(
            dir=self.temporary,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
        try:
            ordered = (
                {"hidden": pending[start]}
                if self.shard_batches == 1
                else {
                    f"batch_{index:09d}": pending[index]
                    for index in range(start, start + count)
                }
            )
            save_safetensors_file(ordered, temporary)
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
            os.replace(temporary, final_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        record = {
            "path": filename,
            "start": start,
            "shapes": [
                list(pending[index].shape)
                for index in range(start, start + count)
            ],
            "dtype": str(pending[start].dtype),
            "bytes": final_path.stat().st_size,
            "xxh3_128": _xxh3_128_file(final_path),
        }
        self._shards[shard_index] = record
        self._committed_indices.update(expected_indices)
        del self._pending[shard_index]
        self._write_progress()

    def put(self, index: int, value: List[torch.Tensor]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("disk-backed output writer is closed")
            if not 0 <= index < self.expected_batches:
                raise IndexError(index)
            if index in self._committed_indices:
                return
            if (
                not isinstance(value, list)
                or len(value) != 1
                or not isinstance(value[0], torch.Tensor)
            ):
                raise TypeError("disk-backed output writer requires one tensor batch")
            tensor = value[0].detach().to(device="cpu").contiguous()
            if tensor.ndim == 0 or int(tensor.shape[0]) <= 0:
                raise ValueError("disk-backed output batch has no rows")
            shard_index = index // self.shard_batches
            pending = self._pending.setdefault(shard_index, {})
            if index in pending:
                raise RuntimeError("disk-backed output batch was written twice")
            pending[index] = tensor
            self._flush_shard(shard_index)

    def append(self, value: List[torch.Tensor]) -> None:
        # Sequential ForwardExecutor compatibility.
        with self._lock:
            index = len(self._committed_indices) + sum(
                len(values) for values in self._pending.values()
            )
        self.put(index, value)

    def finalize(self) -> DiskBackedLayerOutputSequence:
        with self._lock:
            if self._complete is not None:
                self._closed = True
                return self._notify_finalized()
            for shard_index in list(self._pending):
                self._flush_shard(shard_index)
            if len(self._committed_indices) != self.expected_batches or self._pending:
                raise RuntimeError(
                    "disk-backed layer output coverage is incomplete: "
                    f"actual={len(self._committed_indices)} "
                    f"expected={self.expected_batches}"
                )
            body = self._progress_body()
            body["status"] = "complete"
            import hashlib

            body["manifest_sha256"] = hashlib.sha256(
                _canonical_json(body)
            ).hexdigest()
            _atomic_json(self.temporary / "manifest.json", body)
            progress = self.temporary / "progress.json"
            progress.unlink()
            _fsync_directory(self.temporary)
            os.replace(self.temporary, self.destination)
            _fsync_directory(self.parent)
            self._complete = DiskBackedLayerOutputSequence.open(
                self.destination, verify_hashes=False
            )
            self._closed = True
            return self._notify_finalized()

    def _notify_finalized(self) -> DiskBackedLayerOutputSequence:
        if self._complete is None:
            raise RuntimeError("disk-backed outputs are not finalized")
        if self._on_finalize is not None and not self._on_finalize_called:
            self._on_finalize(self._complete)
            self._on_finalize_called = True
        return self._complete

    def abort(self) -> None:
        # Complete shards and their progress manifest deliberately survive.
        with self._lock:
            self._pending.clear()
            self._closed = True
