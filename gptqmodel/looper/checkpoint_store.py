# SPDX-License-Identifier: Apache-2.0
"""Transactional, content-addressed checkpoint storage, independent of the looper.

Only CURRENT publishes a generation. Directory listings are never recovery
candidates: they may contain writes from an interrupted transaction. A manifest
names *all* artifacts needed by its continuation, including earlier layers.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import get_ident
from uuid import uuid4

from filelock import FileLock, Timeout


class CheckpointError(RuntimeError):
    pass


class CheckpointCorrupt(CheckpointError):
    pass


@dataclass(frozen=True)
class CheckpointConfig:
    path: str | Path = "auto"
    resume: str = "auto"
    interval: str = "layer:1"
    keep_last: int = 2
    strict_device_check: bool = True

    def __post_init__(self):
        if type(self.strict_device_check) is not bool:
            raise ValueError("strict_device_check must be a boolean")
        if self.resume not in {"auto", "required", "never"}:
            raise ValueError("resume must be auto, required, or never")
        if not isinstance(self.interval, str):
            raise ValueError("interval must be a string such as 'layer:1'")
        match = re.fullmatch(r"layer:(\d+)", self.interval.strip())
        if match is None or int(match.group(1)) < 1:
            raise ValueError(
                "interval must use the form 'layer:<positive integer>'"
            )
        if type(self.keep_last) is not int or self.keep_last < 2:
            raise ValueError("keep_last must be at least two")
        if not str(self.path):
            raise ValueError("checkpoint path cannot be empty")

    @property
    def interval_layers(self) -> int:
        """Return the layer interval encoded by ``interval``."""
        return int(self.interval.split(":", 1)[1])


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sync_directory(path):
    # Do not silently promise host-crash durability on unsupported filesystems.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointStore:
    VERSION = 1

    def __init__(self, config: CheckpointConfig, *, identity_projection=None):
        self.config = config
        self._identity_projection = identity_projection or (lambda identity: identity)
        self.root = Path(config.path).absolute()
        self._lease = None
        self._owner = None
        self._identity = None
        self._cursor = -1

    def __enter__(self):
        if self._lease is not None:
            raise CheckpointError("checkpoint lease is already held")
        missing = []
        parent = self.root
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing):
            _sync_directory(directory.parent)
        lease = FileLock(str(self.root / "writer.lock"), timeout=0)
        try:
            lease.acquire()
        except Timeout as exc:
            raise CheckpointError(
                "another writer owns this checkpoint directory"
            ) from exc
        self._lease, self._owner = lease, get_ident()
        self._identity = None
        self._cursor = -1
        try:
            (self.root / "objects").mkdir(exist_ok=True)
            _sync_directory(self.root)
            _sync_directory(self.root.parent)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        self._check_owner()
        self._lease.release()
        self._lease = self._owner = None

    def _check_owner(self):
        if self._lease is None or self._owner != get_ident():
            raise CheckpointError(
                "checkpoint operations require the lease-owning thread"
            )

    def _atomic_write(self, destination, data):
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def put(self, data: bytes) -> str:
        """Persist an immutable object; publishing its reference is separate."""
        self._check_owner()
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "objects" / digest
        if path.exists():
            try:
                self.object_path(digest)
            except CheckpointCorrupt:
                # Recomputed content may repair a damaged object, but can never
                # change the bytes promised by this content-addressed name.
                self._atomic_write(path, data)
        else:
            self._atomic_write(path, data)
        return digest

    def get(self, digest: str) -> bytes:
        self._check_owner()
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise CheckpointCorrupt("invalid object reference")
        try:
            data = (self.root / "objects" / digest).read_bytes()
        except FileNotFoundError as exc:
            raise CheckpointCorrupt(f"missing checkpoint object {digest}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise CheckpointCorrupt(f"checksum mismatch for checkpoint object {digest}")
        return data

    def object_path(self, digest: str) -> Path:
        """Verify an object without allocating its tensor payload in memory."""
        self._check_owner()
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise CheckpointCorrupt("invalid object reference")
        path = self.root / "objects" / digest
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise CheckpointCorrupt(f"missing checkpoint object {digest}") from exc
        if actual != digest:
            raise CheckpointCorrupt(f"checksum mismatch for checkpoint object {digest}")
        return path

    def put_file(self, source: Path) -> str:
        """Stream a quiesced artifact into the store using bounded host memory."""
        self._check_owner()
        source = source.absolute()
        if source.parent == self.root / "objects":
            self.object_path(source.name)
            return source.name
        temporary = self.root / "objects" / f".{uuid4().hex}.tmp"
        digest = hashlib.sha256()
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while chunk := reader.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            ref = digest.hexdigest()
            destination = self.root / "objects" / ref
            if destination.exists():
                try:
                    self.object_path(ref)
                except CheckpointCorrupt:
                    os.replace(temporary, destination)
                    _sync_directory(destination.parent)
            else:
                os.replace(temporary, destination)
                _sync_directory(destination.parent)
            return ref
        finally:
            temporary.unlink(missing_ok=True)

    def _current(self):
        path = self.root / "CURRENT"
        if not path.exists():
            return []
        try:
            current = json.loads(path.read_bytes())
            refs = current["generations"]
            if (
                current["version"] != self.VERSION
                or not isinstance(refs, list)
                or not refs
                or len(refs) != len(set(refs))
                or any(
                    not isinstance(ref, str)
                    or re.fullmatch(r"[0-9a-f]{64}", ref) is None
                    for ref in refs
                )
            ):
                raise ValueError("invalid CURRENT")
            return refs
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointCorrupt(
                "invalid CURRENT; refusing to scan unpublished generations"
            ) from exc

    def _manifest(self, ref):
        try:
            manifest = json.loads(self.get(ref))
            if (
                manifest["version"] != self.VERSION
                or type(manifest["cursor"]) is not int
                or manifest["cursor"] < 0
                or not isinstance(manifest["artifacts"], dict)
                or not isinstance(manifest["identity"], dict)
            ):
                raise ValueError("invalid manifest")
            self.get(manifest["continuation"])
            for digest in manifest["artifacts"].values():
                self.object_path(digest)
            return manifest
        except (ValueError, KeyError, TypeError) as exc:
            raise CheckpointCorrupt(f"invalid checkpoint manifest {ref}") from exc

    def load(self, identity: dict):
        """Load the newest complete *published* generation, or fail closed."""
        self._check_owner()
        refs = self._current()
        if self.config.resume == "never" and refs:
            raise CheckpointError(
                "checkpoint exists; resume='never' requires a new directory"
            )
        failures = []
        for ref in refs:
            try:
                manifest = self._manifest(ref)
            except CheckpointCorrupt as exc:
                failures.append(str(exc))
                continue
            expected_identity = self._identity_projection(manifest["identity"])
            actual_identity = self._identity_projection(identity)
            if _json(expected_identity) != _json(actual_identity):
                fields = sorted(
                    key
                    for key in actual_identity.keys() | expected_identity.keys()
                    if _json(actual_identity.get(key))
                    != _json(expected_identity.get(key))
                    or (key in actual_identity) != (key in expected_identity)
                )
                raise CheckpointError(
                    f"incompatible checkpoint identity fields: {', '.join(fields)}"
                )
            self._identity = _json(identity)
            self._cursor = manifest["cursor"]
            return manifest
        if refs:
            raise CheckpointCorrupt(
                "no complete published checkpoint: " + "; ".join(failures)
            )
        if self.config.resume == "required":
            raise CheckpointError("resume='required' but no checkpoint is published")
        self._identity = _json(identity)
        self._cursor = -1
        return None

    def commit(
        self,
        *,
        identity: dict,
        cursor: int,
        continuation: str,
        artifacts: dict[str, str],
    ):
        self._check_owner()
        # Enforce policy and identity even when callers forget to call load().
        if self._identity is None:
            self.load(identity)
        if self._identity != _json(identity):
            raise CheckpointError("cannot change checkpoint identity within a run")
        if type(cursor) is not int or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer")
        if cursor <= self._cursor:
            raise CheckpointError("checkpoint cursor must advance")
        manifest = {
            "version": self.VERSION,
            "identity": identity,
            "cursor": cursor,
            "continuation": continuation,
            "artifacts": artifacts,
        }
        self.get(continuation)
        for digest in artifacts.values():
            self.object_path(digest)
        ref = self.put(_json(manifest))
        # Retain only verified generations. Broken generations must not evict
        # the last good fallback when recovery publishes its next checkpoint.
        retained = []
        for old in self._current():
            if old == ref:
                continue
            try:
                self._manifest(old)
            except CheckpointCorrupt:
                continue
            retained.append(old)
        self._atomic_write(
            self.root / "CURRENT",
            _json(
                {
                    "version": self.VERSION,
                    "generations": ([ref] + retained)[: self.config.keep_last],
                }
            ),
        )
        self._cursor = cursor
        return manifest

    def collect(self):
        """Remove unreachable objects under the run lease, after publication.

        Kept separate from commit so a cleanup failure cannot report a durable
        commit as failed. Corrupt retained metadata aborts cleanup conservatively.
        """
        self._check_owner()
        reachable = set(self._current())
        for ref in tuple(reachable):
            manifest = self._manifest(ref)
            reachable.add(manifest["continuation"])
            reachable.update(manifest["artifacts"].values())
        for path in (self.root / "objects").iterdir():
            if re.fullmatch(r"\.(?:[0-9a-f]{64}\.)?[0-9a-f]{32}\.tmp", path.name) or (
                re.fullmatch(r"[0-9a-f]{64}", path.name) and path.name not in reachable
            ):
                path.unlink()
        _sync_directory(self.root / "objects")
        for path in self.root.iterdir():
            if re.fullmatch(r"\.CURRENT\.[0-9a-f]{32}\.tmp", path.name):
                path.unlink()
        _sync_directory(self.root)
