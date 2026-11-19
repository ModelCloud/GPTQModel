# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5424: ``dequantize_model`` must not read and write the same tree.

``resume=True`` treats every file already present in the output directory as a
completed dense shard. If the output resolves to the source checkpoint, those
"completed" files are the quantized source, so the export rewrites the source
``config.json``/index as dense metadata while leaving the packed weight bytes
in place and reporting success. The preflight check must reject the overlap
before any write, including equivalent spellings and nested directories.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from gptqmodel.utils.model_dequant import dequantize_model
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]


def _source(path):
    source = path / "source"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=4, lm_head_bits=4, legacy_config=False)
    return source


def _snapshot(source):
    return (
        (source / "config.json").read_bytes(),
        hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest(),
    )


def _assert_rejected(source, output):
    before = _snapshot(source)
    with pytest.raises(ValueError, match="separate directory"):
        dequantize_model(source, output, target_dtype=torch.float32, resume=True)
    # No write may happen before the overlap is rejected.
    assert _snapshot(source) == before


def test_resume_into_the_source_directory_is_rejected(tmp_path):
    source = _source(tmp_path)
    _assert_rejected(source, source)

    # The quantized payload is still declared and the packed bytes are untouched.
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["quant_method"] == "gptq"
    assert config["dtype"] == "float32"


@pytest.mark.parametrize("spelling", ["dot", "parent"])
def test_equivalent_path_spellings_are_rejected(tmp_path, spelling):
    source = _source(tmp_path)
    output = source / "." if spelling == "dot" else source / "nested" / ".."
    _assert_rejected(source, output)


def test_symlinked_output_alias_is_rejected(tmp_path):
    source = _source(tmp_path)
    link = tmp_path / "alias"
    try:
        link.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform guard
        pytest.skip("symlinks are not available on this platform")
    _assert_rejected(source, link)


def test_output_nested_in_the_source_is_rejected(tmp_path):
    source = _source(tmp_path)
    _assert_rejected(source, source / "dense")


def test_source_nested_in_the_output_is_rejected(tmp_path):
    container = tmp_path / "container"
    container.mkdir()
    source = _source(container)
    _assert_rejected(source, container)


def test_resume_into_a_separate_directory_still_reuses_completed_shards(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "source-dense"

    dequantize_model(source, output, target_dtype=torch.float32, resume=True)
    assert "quantization_config" not in json.loads(
        (output / "config.json").read_text(encoding="utf-8")
    )
    assert (output / "model.safetensors").exists()

    # A second resume run reuses the completed shard instead of re-converting it.
    dequantize_model(source, output, target_dtype=torch.float32, resume=True)
    assert "quantization_config" not in json.loads(
        (output / "config.json").read_text(encoding="utf-8")
    )
