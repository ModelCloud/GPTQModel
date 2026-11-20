# SPDX-License-Identifier: Apache-2.0
"""Explicit tensor/value continuation format; never deserialize Python code."""

import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load, save, save_file

from ..utils.device_telemetry import emit_device_telemetry
from .input_cache import InputCache


def _tensor_version(tensor):
    """Return the shared version counter, or ``None`` for inference tensors."""
    try:
        return tensor._version
    except RuntimeError:
        return None


class ContinuationCodec:
    VERSION = 2

    @staticmethod
    def _encode(state):
        tensors = {}
        names = {}
        storage_ranges = {}
        active = set()

        def register(value):
            name = names.get(id(value))
            if name is not None:
                return name
            name = str(len(tensors))
            tensor = value.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.to(device="cpu")
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            storage = tensor.untyped_storage().data_ptr()
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            spans = storage_ranges.setdefault(storage, [])
            if any(start < stop and begin < end for begin, stop in spans):
                # safetensors rejects entries whose storage overlaps another
                # entry. Copy only the conflicting view: ordinary CPU tensors
                # stay owned by the live state instead of being cloned once per
                # reference.
                tensor = tensor.clone()
                storage = tensor.untyped_storage().data_ptr()
                start = tensor.data_ptr()
                end = start + tensor.numel() * tensor.element_size()
                storage_ranges.setdefault(storage, []).append((start, end))
            else:
                spans.append((start, end))
            tensors[name] = tensor
            names[id(value)] = name
            emit_device_telemetry(
                "checkpoint_tensor_captured",
                tensor_id=name,
                source_device=str(value.device),
                storage_device="cpu",
                shape=list(value.shape),
                dtype=str(value.dtype),
            )
            return name

        def encode(value):
            if value is None or type(value) in (bool, int, str):
                return ["scalar", value]
            if type(value) is float and math.isfinite(value):
                return ["scalar", value]
            if isinstance(value, torch.Tensor):
                if (
                    value.device.type not in {"cpu", "cuda"}
                    or value.layout != torch.strided
                ):
                    raise TypeError("continuation requires materialized dense tensors")
                name = register(value)
                return ["tensor", {"name": name, "device": str(value.device)}]
            if type(value) is torch.device:
                device = value
                if device.type not in {"cpu", "cuda"}:
                    raise TypeError("unsupported continuation metadata device")
                if device.type == "cuda" and device.index is None:
                    device = torch.device("cuda", torch.cuda.current_device())
                emit_device_telemetry(
                    "checkpoint_device_metadata_captured", device=str(device)
                )
                return ["device", str(device)]
            if id(value) in active:
                raise TypeError("cyclic continuation state is unsupported")
            active.add(id(value))
            try:
                if type(value) is InputCache:
                    return ["input_cache", encode(vars(value))]
                if type(value) is dict:
                    if any(type(key) not in (str, int, bool) for key in value):
                        raise TypeError(
                            "continuation keys must be strings, integers, or booleans"
                        )
                    return [
                        "dict",
                        [[encode(key), encode(item)] for key, item in value.items()],
                    ]
                if type(value) in (list, tuple):
                    return [
                        "tuple" if type(value) is tuple else "list",
                        [encode(item) for item in value],
                    ]
            finally:
                active.remove(id(value))
            raise TypeError(f"unsupported continuation value: {type(value).__name__}")

        tree = encode(state)
        # Metadata and tensor payload are one immutable object, not independently
        # published files.
        metadata = {
            "continuation": json.dumps(
                {"version": ContinuationCodec.VERSION, "tree": tree},
                allow_nan=False,
            )
        }
        return tensors, metadata

    @staticmethod
    def dumps(state) -> bytes:
        tensors, metadata = ContinuationCodec._encode(state)
        return save(tensors, metadata=metadata)

    @staticmethod
    def dump_file(state, path):
        """Write the continuation straight to ``path`` without a second buffer."""
        tensors, metadata = ContinuationCodec._encode(state)
        save_file(tensors, path, metadata=metadata)

    @staticmethod
    def sha256(state, directory) -> str:
        """Return the digest of the exact bytes :meth:`dumps` would produce."""
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".continuation-", suffix=".tmp", delete=False
        ) as staged:
            path = Path(staged.name)
        try:
            ContinuationCodec.dump_file(state, path)
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _document(metadata):
        document = json.loads(metadata["continuation"])
        if document["version"] != ContinuationCodec.VERSION:
            raise ValueError("unsupported continuation version")
        return document

    @staticmethod
    def loads(data: bytes):
        # safetensors exposes metadata through safe_open for files, but this
        # codec operates on verified bytes, so parse its length-prefixed header.
        header_size = int.from_bytes(data[:8], "little")
        if header_size > len(data) - 8 or header_size < 2:
            raise ValueError("invalid continuation header")
        header = json.loads(data[8 : 8 + header_size])
        document = ContinuationCodec._document(header["__metadata__"])
        tensors = load(data)
        return ContinuationCodec._decode(document["tree"], tensors.__getitem__)

    @staticmethod
    def load_file(path):
        """Decode a verified continuation file, materializing each tensor once."""
        with safe_open(path, framework="pt", device="cpu") as tensors:
            metadata = tensors.metadata()
            if not metadata:
                raise ValueError("invalid continuation metadata")
            document = ContinuationCodec._document(metadata)
            return ContinuationCodec._decode(document["tree"], tensors.get_tensor)

    @staticmethod
    def _decode(tree, get_tensor):
        restored = {}

        def decode(node):
            kind, value = node
            if kind == "scalar" and (
                value is None
                or type(value) in (str, bool, int)
                or type(value) is float
                and math.isfinite(value)
            ):
                return value
            if kind == "tensor":
                device = torch.device(value["device"])
                if (
                    device.type not in {"cpu", "cuda"}
                    or device.type == "cuda"
                    and device.index is None
                ):
                    raise ValueError(
                        "unsupported or unindexed continuation tensor device"
                    )
                name = value["name"]
                tensor = restored.get(name)
                if tensor is None:
                    stored = get_tensor(name)
                    expected_shape, expected_dtype = stored.shape, stored.dtype
                    # A transfer that mutates its source in place bumps the
                    # shared version counter; comparing the live bytes then
                    # would otherwise compare the corruption to itself.
                    expected_version = _tensor_version(stored)
                    expected_bits = stored.contiguous().reshape(-1).view(torch.uint8)
                    if expected_version is None:
                        # Inference tensors do not track a version counter, so
                        # keep an independent reference for the bitwise check.
                        expected_bits = expected_bits.clone()
                    tensor = stored.to(device=device)
                    placement_matched = tensor.device == device
                    state_matched = (
                        tensor.dtype == expected_dtype
                        and tensor.shape == expected_shape
                        and _tensor_version(stored) == expected_version
                        and torch.equal(
                            tensor.detach()
                            .cpu()
                            .contiguous()
                            .reshape(-1)
                            .view(torch.uint8),
                            expected_bits,
                        )
                    )
                    matched = placement_matched and state_matched
                    emit_device_telemetry(
                        "checkpoint_tensor_restored",
                        tensor_id=name,
                        expected_device=str(device),
                        actual_device=str(tensor.device),
                        matched=matched,
                        placement_matched=placement_matched,
                        state_matched=state_matched,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                    )
                    if not matched:
                        raise ValueError(
                            "checkpoint tensor state or device placement restore mismatch"
                        )
                    restored[name] = tensor
                elif tensor.device != device:
                    raise ValueError(
                        "checkpoint tensor state or device placement restore mismatch"
                    )
                return tensor
            if kind == "device":
                device = torch.device(value)
                if (
                    device.type not in {"cpu", "cuda"}
                    or device.type == "cuda"
                    and device.index is None
                ):
                    raise ValueError(
                        "unsupported or unindexed continuation metadata device"
                    )
                emit_device_telemetry(
                    "checkpoint_device_metadata_restored",
                    expected_device=value,
                    actual_device=str(device),
                    matched=str(device) == value,
                )
                if str(device) != value:
                    raise ValueError("checkpoint device metadata restore mismatch")
                return device
            if kind == "input_cache":
                fields = decode(value)
                if set(fields) != set(InputCache.__dataclass_fields__):
                    raise ValueError("unsupported InputCache schema")
                return InputCache(**fields)
            if kind == "dict":
                result = {}
                for key, item in value:
                    key = decode(key)
                    if type(key) not in (str, int, bool) or key in result:
                        raise ValueError("invalid continuation dictionary key")
                    result[key] = decode(item)
                return result
            if kind in ("list", "tuple"):
                items = [decode(item) for item in value]
                return tuple(items) if kind == "tuple" else items
            raise ValueError(f"unsupported continuation tag: {kind}")

        return decode(tree)
