# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5398: sharded GPTQ/AWQ checkpoints dequantize to the same weights.

GPT-QModel's writer can place one module's ``qweight``, ``qzeros``, ``scales``
and ``g_idx`` in different indexed files. The converters used to look only at
the current file, so a valid sharded checkpoint failed with "Incomplete
GPTQ/AWQ buffers". Each module's dense weight must also be emitted exactly
once, by the shard that owns its packed payload.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel.utils.model_dequant import dequantize_model
from tests.test_awq_dequant_layouts import _PREFIX as AWQ_PREFIX
from tests.test_awq_dequant_layouts import _gemm_layout_writer, _packed_tensors
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]

_AUX_FILES = (
    "config.json",
    "quantize_config.json",
    "quant_config.json",
    "tokenizer_config.json",
)


def _split_source(source: Path, target: Path, assign) -> Path:
    """Rewrite ``source/model.safetensors`` into shards chosen by ``assign``."""

    tensors = load_file(str(source / "model.safetensors"))
    groups: dict[str, list[str]] = {}
    for key in sorted(tensors):
        groups.setdefault(assign(key), []).append(key)

    target.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    count = len(groups)
    for index, name in enumerate(sorted(groups), start=1):
        filename = f"model-{index:05d}-of-{count:05d}.safetensors"
        save_file(
            {key: tensors[key].contiguous() for key in groups[name]},
            str(target / filename),
        )
        total_size += (target / filename).stat().st_size
        for key in groups[name]:
            weight_map[key] = filename

    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    for name in _AUX_FILES:
        aux = source / name
        if aux.exists():
            shutil.copy2(aux, target / name)
    return target


def _gptq_assign(key: str) -> str:
    """Split every module's packed payload from its companion buffers."""

    if key.endswith(".qweight"):
        return "packed"
    if key.endswith((".qzeros", ".scales", ".g_idx")):
        return "companions"
    return "dense"


def _awq_assign(key: str) -> str:
    if key.endswith(".qweight"):
        return "packed"
    if key.endswith((".qzeros", ".scales", ".scaled_zeros")):
        return "companions"
    return "dense"


def _export_weights(out: Path) -> dict[str, torch.Tensor]:
    """Every dense tensor the export wrote, with its owning file name."""

    files = sorted(path.name for path in out.glob("*.safetensors"))
    assert files, f"no shards written to {out}"
    weights: dict[str, torch.Tensor] = {}
    owners: dict[str, str] = {}
    for filename in files:
        with safe_open(str(out / filename), framework="pt") as reader:
            for key in reader.keys():
                assert key not in weights, f"{key} was written by both {owners[key]} and {filename}"
                weights[key] = reader.get_tensor(key)
                owners[key] = filename
    return weights


def _gptq_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=4, lm_head_bits=4, legacy_config=False)
    return source


def test_sharded_gptq_export_matches_the_single_file_export(tmp_path):
    source = _gptq_source(tmp_path)
    single_out = tmp_path / "single"
    dequantize_model(source, single_out, target_dtype=torch.float32)
    expected = _export_weights(single_out)

    sharded = _split_source(source, tmp_path / "sharded", _gptq_assign)
    sharded_out = tmp_path / "sharded_out"
    dequantize_model(sharded, sharded_out, target_dtype=torch.float32)
    actual = _export_weights(sharded_out)

    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor, rtol=0.0, atol=0.0)

    # Each module's dense weight is written exactly once (_export_weights asserts
    # uniqueness); check the projections actually made it into the export.
    for prefix in ("model.layers.0.self_attn.q_proj", "lm_head", "model.embed_tokens"):
        assert f"{prefix}.weight" in actual


def test_sharded_awq_export_matches_the_single_file_export(tmp_path):
    tensors, expected_weight = _packed_tensors(AWQ_PREFIX, _gemm_layout_writer(4), 4, 128, 128, 128)
    source = tmp_path / "awq_source"
    source.mkdir()
    save_file(tensors, str(source / "model.safetensors"))
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "method": "awq",
            "format": "gemm",
            "bits": 4,
            "group_size": 128,
        },
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")

    single_out = tmp_path / "awq_single"
    dequantize_model(source, single_out, target_dtype=torch.float32)
    expected = _export_weights(single_out)
    torch.testing.assert_close(
        expected[f"{AWQ_PREFIX}.weight"], expected_weight, rtol=0.0, atol=0.0
    )

    sharded = _split_source(source, tmp_path / "awq_sharded", _awq_assign)
    sharded_out = tmp_path / "awq_sharded_out"
    dequantize_model(sharded, sharded_out, target_dtype=torch.float32)
    actual = _export_weights(sharded_out)

    assert set(actual) == set(expected)
    torch.testing.assert_close(
        actual[f"{AWQ_PREFIX}.weight"], expected_weight, rtol=0.0, atol=0.0
    )


def test_a_genuinely_incomplete_module_still_fails(tmp_path):
    """Only a buffer the index can resolve is borrowed from another shard."""

    source = _gptq_source(tmp_path)
    sharded = _split_source(source, tmp_path / "sharded", _gptq_assign)

    # Drop the scales shard from the index: the module is now incomplete.
    index_path = sharded / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {
        key: shard
        for key, shard in index["weight_map"].items()
        if not key.endswith(".scales")
    }
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(KeyError, match="Incomplete GPTQ buffers"):
        dequantize_model(sharded, tmp_path / "out", target_dtype=torch.float32)
