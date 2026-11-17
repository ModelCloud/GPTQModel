# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the pinned checkpoint inventory using safetensors headers only."""

import gzip
import hashlib
import json
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/nemotron_h_puzzle"
    manifest = json.loads((root / "source_manifest.json").read_text())
    base = manifest["base_url"]
    response = requests.get(base + "model.safetensors.index.json", timeout=90)
    response.raise_for_status()
    assert hashlib.sha256(response.content).hexdigest() == manifest["sha256"]["model.safetensors.index.json"]
    weight_map = response.json()["weight_map"]

    def read_range(name, end):
        with requests.get(base + name, headers={"Range": f"bytes=0-{end}"}, stream=True, timeout=90) as shard:
            shard.raise_for_status()
            return shard.raw.read(end + 1)

    shapes, checksums = {}, {}
    for name in sorted(set(weight_map.values())):
        size = int.from_bytes(read_range(name, 7), "little")
        assert 0 < size < 20_000_000, size
        data = read_range(name + "?header=1", size + 7)
        assert len(data) == size + 8
        header = data[8:]
        checksums[name] = hashlib.sha256(header).hexdigest()
        entries = {key: value["shape"] for key, value in json.loads(header).items() if key != "__metadata__"}
        assert not set(entries) & set(shapes)
        assert all(weight_map[key] == name for key in entries)
        shapes.update(entries)
    assert set(shapes) == set(weight_map)
    output = {"revision": manifest["revision"], "shapes": shapes, "header_sha256": checksums}
    with gzip.open(root / "checkpoint_shapes.json.gz", "wt") as handle:
        json.dump(output, handle, sort_keys=True, separators=(",", ":"))
    print(f"Read {len(shapes)} tensors from {len(checksums)} pinned safetensors headers")


if __name__ == "__main__":
    main()
