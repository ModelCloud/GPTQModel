# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5396: a resumed export must continue the same export.

``dequantize_model(..., resume=True)`` used to accept any readable output shard
whose filename matched a source shard. Pointing the same output directory at a
different checkpoint, or re-running with another target dtype, therefore
returned success with stale weights while the config described the new dtype.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from gptqmodel.utils.model_dequant import dequantize_model
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]


def _source(path, *, bits=4):
    path.mkdir()
    _tiny_llama_checkpoint(path, bits=bits, lm_head_bits=bits, legacy_config=False)
    return path


def _exported(out):
    """Read every exported tensor, detached from the shard's memory map.

    The tensors `safe_open` returns are views into an mmap; rewriting the file
    while one is alive faults (SIGBUS), so clone them out first.
    """

    tensors = {}
    for shard in sorted(out.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as reader:
            for key in reader.keys():
                tensors[key] = reader.get_tensor(key).clone()
    return tensors


def _dtypes(tensors):
    return {str(tensor.dtype) for tensor in tensors.values() if tensor.is_floating_point()}


def _assert_close(left, right):
    assert set(left) == set(right)
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], rtol=0.0, atol=0.0)


def test_resume_rebuilds_when_the_source_checkpoint_changes(tmp_path):
    first = _source(tmp_path / "first", bits=4)
    second = _source(tmp_path / "second", bits=8)

    out = tmp_path / "out"
    dequantize_model(first, out, target_dtype=torch.float32)
    stale = _exported(out)

    dequantize_model(second, out, target_dtype=torch.float32, resume=True)
    resumed = _exported(out)

    control = tmp_path / "control"
    dequantize_model(second, control, target_dtype=torch.float32)
    expected = _exported(control)

    _assert_close(resumed, expected)
    assert any(
        not torch.equal(resumed[name], stale[name]) for name in resumed
    ), "the resumed export kept the previous checkpoint's weights"


def test_resume_rebuilds_when_the_target_dtype_changes(tmp_path):
    source = _source(tmp_path / "source")

    out = tmp_path / "out"
    dequantize_model(source, out, target_dtype=torch.float32)
    assert _dtypes(_exported(out)) == {"torch.float32"}

    dequantize_model(source, out, target_dtype=torch.float16, resume=True)

    assert _dtypes(_exported(out)) == {"torch.float16"}
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["dtype"] == "float16"


def test_resume_reuses_a_compatible_export(tmp_path):
    source = _source(tmp_path / "source")
    out = tmp_path / "out"

    dequantize_model(source, out, target_dtype=torch.bfloat16)
    shard = out / "model.safetensors"
    before = (shard.stat().st_mtime_ns, shard.stat().st_size)
    first = _exported(out)

    manifest = json.loads(
        (out / "gptqmodel_dequant_resume.json").read_text(encoding="utf-8")
    )
    assert manifest["completed"] == ["model.safetensors"]
    assert manifest["target_dtype"] == "bfloat16"

    dequantize_model(source, out, target_dtype=torch.bfloat16, resume=True)

    # Reused, not rewritten.
    assert (shard.stat().st_mtime_ns, shard.stat().st_size) == before
    _assert_close(_exported(out), first)


def test_resume_rebuilds_an_unreadable_shard(tmp_path):
    source = _source(tmp_path / "source")
    out = tmp_path / "out"

    dequantize_model(source, out, target_dtype=torch.float32)
    expected = _exported(out)
    (out / "model.safetensors").write_bytes(b"not a safetensors file")

    dequantize_model(source, out, target_dtype=torch.float32, resume=True)

    _assert_close(_exported(out), expected)


def test_resume_without_a_manifest_rebuilds(tmp_path):
    """An export made before this fix has no manifest; do not trust it."""

    source = _source(tmp_path / "source")
    out = tmp_path / "out"

    dequantize_model(source, out, target_dtype=torch.float32)
    expected = _exported(out)
    (out / "gptqmodel_dequant_resume.json").unlink()
    (out / "model.safetensors").write_bytes(b"stale")

    dequantize_model(source, out, target_dtype=torch.float32, resume=True)

    _assert_close(_exported(out), expected)
