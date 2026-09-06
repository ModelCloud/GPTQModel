# SPDX-License-Identifier: Apache-2.0
"""Explicit tensor/value continuation format; never deserialize Python code."""

import json
import math

import torch
from safetensors.torch import load, save

from .input_cache import InputCache


class ContinuationCodec:
    VERSION = 1

    @staticmethod
    def dumps(state) -> bytes:
        tensors = {}
        active = set()

        def encode(value):
            if value is None or type(value) in (bool, int, str):
                return ["scalar", value]
            if type(value) is float and math.isfinite(value):
                return ["scalar", value]
            if isinstance(value, torch.Tensor):
                if value.device.type == "meta" or value.layout != torch.strided:
                    raise TypeError("continuation requires materialized dense tensors")
                name = str(len(tensors))
                tensors[name] = value.detach().to(device="cpu").contiguous().clone()
                return ["tensor", name]
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
        # published files. Cloning removes safetensors shared-storage ambiguity.
        return save(
            tensors,
            metadata={
                "continuation": json.dumps(
                    {"version": ContinuationCodec.VERSION, "tree": tree},
                    allow_nan=False,
                )
            },
        )

    @staticmethod
    def loads(data: bytes):
        # safetensors exposes metadata through safe_open for files, but this
        # codec operates on verified bytes, so parse its length-prefixed header.
        header_size = int.from_bytes(data[:8], "little")
        if header_size > len(data) - 8 or header_size < 2:
            raise ValueError("invalid continuation header")
        header = json.loads(data[8 : 8 + header_size])
        document = json.loads(header["__metadata__"]["continuation"])
        if document["version"] != ContinuationCodec.VERSION:
            raise ValueError("unsupported continuation version")
        tensors = load(data)

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
                return tensors[value]
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

        return decode(document["tree"])
