# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Safetensors IO only; no model/config/tokenizer framework is imported."""

import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path


def load_safetensors(model, directory):
    """Validate complete keys/aliases, then stream tensors into generated storage."""
    from safetensors import safe_open

    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in filenames):
            raise ValueError("checkpoint shard filenames must be local basenames")
    else:
        filenames, weight_map = ["model.safetensors"], None
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(safe_open(directory / name, framework="pt", device="cpu")) for name in filenames
        }
        actual = {}
        for name, reader in readers.items():
            for key in reader.keys():
                if key in actual:
                    raise ValueError(f"duplicate checkpoint key: {key}")
                actual[key] = name
        if weight_map is not None and actual != weight_map:
            raise ValueError("checkpoint index disagrees with shard headers")

        class Weights(Mapping):
            def __len__(self):
                return len(actual)

            def __iter__(self):
                return iter(actual)

            def __getitem__(self, key):
                return readers[actual[key]].get_tensor(key)

        model.load_weights(Weights())
    return model
