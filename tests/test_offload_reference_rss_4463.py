# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4463 item 1: proving a base module matches its source must not copy it.

`_module_source_reference_index` used to load every source tensor with
`safe_open(...).get_tensor()` (a full owned copy plus the mmap'd file pages)
before `torch.equal`, which added about twice the embedding size to the
quantize peak RSS. The comparison now streams the source payload through one
bounded buffer against a byte view of the live tensor.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

import gptqmodel.utils.offload as offload
from gptqmodel.utils.offload import offload_to_disk, undo_offload_to_disk
from gptqmodel.utils.tensor import tensor_buffer


class _Holder(torch.nn.Module):
    def __init__(self, rows: int = 128, cols: int = 64) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(rows, cols)

    def tie_weights(self) -> None:  # pragma: no cover - interface match
        return None


def _save_source(model: torch.nn.Module, path) -> None:
    save_file({name: value.detach().clone() for name, value in model.state_dict().items()}, str(path))


def _index(path) -> dict:
    with open(os.path.join(path, "index.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_chunked_compare_references_source_and_hashes_live_bytes(tmp_path, monkeypatch):
    # An odd chunk size puts chunk boundaries inside elements and leaves a
    # short final chunk.
    monkeypatch.setattr(offload, "_SOURCE_COMPARE_CHUNK_BYTES", 7)
    torch.manual_seed(4463)
    model = _Holder().eval()
    source = tmp_path / "model.safetensors"
    _save_source(model, source)
    expected = model.model.embed_tokens.weight.detach().clone()
    _owner, payload = tensor_buffer(expected)
    expected_sha = hashlib.sha256(payload).hexdigest()

    offload_to_disk(
        module=["model.embed_tokens"],
        model=model,
        disk_path=str(tmp_path),
        source_root=str(tmp_path),
        source_checkpoint=str(source),
    )

    module_dir = tmp_path / "model.embed_tokens"
    assert sorted(os.listdir(module_dir)) == ["index.json"]
    entry = _index(module_dir)["weight"]
    assert entry["safetensors_file"] == str(source)
    assert entry["sha256"] == expected_sha
    undo_offload_to_disk(model.model.embed_tokens, device=torch.device("cpu"))
    assert torch.equal(model.model.embed_tokens.weight, expected)


@pytest.mark.parametrize("flat_index", (0, -1))
def test_chunked_compare_detects_a_single_byte_difference(tmp_path, monkeypatch, flat_index):
    monkeypatch.setattr(offload, "_SOURCE_COMPARE_CHUNK_BYTES", 7)
    torch.manual_seed(4463)
    model = _Holder().eval()
    source = tmp_path / "model.safetensors"
    _save_source(model, source)
    with torch.no_grad():
        # Flip one bit of the first or last element: only one chunk differs.
        bits = model.model.embed_tokens.weight.view(-1).view(torch.int32)
        bits[flat_index] ^= 1
    expected = model.model.embed_tokens.weight.detach().clone()

    offload_to_disk(
        module=["model.embed_tokens"],
        model=model,
        disk_path=str(tmp_path),
        source_root=str(tmp_path),
        source_checkpoint=str(source),
    )

    module_dir = tmp_path / "model.embed_tokens"
    assert "module.safetensors" in os.listdir(module_dir)
    undo_offload_to_disk(model.model.embed_tokens, device=torch.device("cpu"))
    assert torch.equal(model.model.embed_tokens.weight, expected)


def _status_kib(field: str) -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise KeyError(field)


def _reset_peak_rss() -> bool:
    # Writing 5 to clear_refs resets VmHWM to the current RSS (Linux >= 4.0).
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc/self peak-RSS reset")
def test_source_reference_check_does_not_materialize_the_tensor(tmp_path):
    # 128 MiB embedding. The old full-tensor check added ~2x this to the peak
    # RSS (an owned copy plus the mapped file pages); the chunked compare adds
    # at most one 16 MiB chunk.
    model = _Holder(rows=32768, cols=1024).eval()
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
    tensor_bytes = model.model.embed_tokens.weight.nbytes
    source = tmp_path / "model.safetensors"
    _save_source(model, source)
    gc.collect()

    if not _reset_peak_rss():
        pytest.skip("/proc/self/clear_refs is not writable")
    before = _status_kib("VmRSS")
    index = offload._module_source_reference_index(
        model.model.embed_tokens,
        module_name="model.embed_tokens",
        source_root=str(tmp_path),
        source_checkpoint=str(source),
        offload_dir=str(tmp_path / "offload" / "model.embed_tokens"),
    )
    extra = (_status_kib("VmHWM") - before) * 1024

    assert index is not None and index["weight"]["safetensors_file"] == str(source)
    # One 16 MiB compare chunk plus allocator slack; far below one tensor copy.
    assert extra < 16 * 2**20 + tensor_bytes // 8, (
        f"reference check raised peak RSS by {extra / 2**20:.1f} MiB "
        f"for a {tensor_bytes / 2**20:.0f} MiB tensor"
    )
