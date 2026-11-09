"""Atomic, non-pickle V4.1 replay frontiers with explicit resume identity.

The caller commits the returned digest in its run journal only after save
returns. A frontier alone is not a completed quantization-block commit.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .v41_replay import V41ReplayBatch


def _digest(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _encode_state(value, tensors):
    # Module-level recursion: an inner recursive function forms a reference
    # cycle through its closure and retains all copied tensor payloads until GC.
    if isinstance(value, torch.Tensor):
        name = f"tensor_{len(tensors)}"
        tensors[name] = value.detach().to(device="cpu", copy=True).contiguous()
        return ["tensor", name]
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("frontier dictionary keys must be strings or integers")
        return ["dict", [[_encode_state(key, tensors), _encode_state(item, tensors)] for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return ["tuple" if isinstance(value, tuple) else "list", [_encode_state(item, tensors) for item in value]]
    if value is None or type(value) in (str, int, float, bool):
        return ["scalar", value]
    raise TypeError(f"unsupported frontier value: {type(value).__name__}")


def _decode_state(node, reader, used):
    kind, payload = node
    if kind == "tensor":
        if payload in used:
            raise ValueError("aliased frontier tensor reference")
        used.add(payload)
        return reader.get_tensor(payload).clone()
    if kind == "dict":
        return {_decode_state(key, reader, used): _decode_state(value, reader, used) for key, value in payload}
    if kind == "tuple":
        return tuple(_decode_state(value, reader, used) for value in payload)
    if kind == "list":
        return [_decode_state(value, reader, used) for value in payload]
    if kind == "scalar":
        return payload
    raise ValueError("invalid frontier node")


def _save_state(state, path, *, provenance, kind):
    """Publish an owned CPU snapshot by atomic rename; return its SHA-256.

    Provenance must bind source, corpus, recipe and implementation in the
    production caller. It is mandatory and matched exactly on load; a reviewed
    code upgrade must explicitly migrate the run identity, never silently skip
    this check. Saving is synchronous: do not mutate the batch concurrently.
    """
    if not isinstance(state, dict) or not isinstance(provenance, dict) or not provenance:
        raise ValueError("V4.1 state and nonempty provenance are required")
    # Reject values JSON would silently normalize (e.g. integer mapping keys).
    if json.loads(json.dumps(provenance, allow_nan=False)) != provenance:
        raise ValueError("provenance must round-trip through JSON unchanged")
    tensors = {}

    manifest = json.dumps({"version": 1, "kind": kind, "provenance": provenance, "state": _encode_state(state, tensors)},
                          allow_nan=False, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        save_file(tensors, temporary, metadata={"v41_frontier": manifest})
        with open(temporary, "rb") as stream:
            digest = _digest(stream)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_state(path, *, expected_sha256, expected_provenance, kind):
    """Verify journal digest and identity before allocating owned CPU state.

    Keep one open inode through verification and safetensors mapping so an
    atomic path replacement cannot switch the file between these operations.
    Linux /proc is required, consistent with the quantization worker platform.
    """
    with open(path, "rb") as stream:
        if _digest(stream) != expected_sha256:
            raise ValueError("V4.1 frontier checksum mismatch")
        with safe_open(f"/proc/self/fd/{stream.fileno()}", framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            manifest = json.loads(metadata["v41_frontier"])
            if manifest.get("version") != 1:
                raise ValueError("unsupported V4.1 frontier version")
            # Existing version-1 snapshots predate the explicit kind field and
            # contain replay states only. They cannot be read as routed batches.
            if manifest.get("kind", "replay") != kind:
                raise ValueError("V4.1 frontier kind mismatch")
            if not expected_provenance or manifest.get("provenance") != expected_provenance:
                raise ValueError("V4.1 frontier provenance mismatch")
            used = set()

            state = _decode_state(manifest["state"], reader, used)
            if used != set(reader.keys()):
                raise ValueError("unreferenced frontier tensors")
    return state


def save_frontier(batch, path, *, provenance):
    """Atomically save replay state; journal the returned digest after success."""
    if not isinstance(batch, V41ReplayBatch):
        raise ValueError("expected a V4.1 replay batch")
    return _save_state(vars(batch), path, provenance=provenance, kind="replay")


def load_frontier(path, *, expected_sha256, expected_provenance):
    """Load owned replay state after verifying digest, kind and provenance."""
    state = _load_state(path, expected_sha256=expected_sha256,
                        expected_provenance=expected_provenance, kind="replay")
    batch = V41ReplayBatch(**state)
    if type(batch.next_layer) is not int or batch.next_layer < 0:
        raise ValueError("invalid frontier layer")
    if not isinstance(batch.hidden, torch.Tensor) or not isinstance(batch.pre_mix, torch.Tensor):
        raise ValueError("invalid frontier hidden state")
    if any(not isinstance(value, dict) for value in (batch.shared, batch.kwargs, batch.engram_rows)):
        raise ValueError("invalid frontier mappings")
    return batch


def save_routed_batch(batch, path, *, provenance):
    """Atomically persist the reusable FFN frontier, never a completed block."""
    from .v41_routed_batch import V41RoutedBatch
    if not isinstance(batch, V41RoutedBatch):
        raise ValueError("expected a V4.1 routed batch")
    batch.validate()
    return _save_state(vars(batch), path, provenance=provenance, kind="routed")


def load_routed_batch(path, *, expected_sha256, expected_provenance):
    from .v41_routed_batch import V41RoutedBatch
    state = _load_state(path, expected_sha256=expected_sha256,
                        expected_provenance=expected_provenance, kind="routed")
    batch = V41RoutedBatch(**state)
    batch.validate()
    return batch
