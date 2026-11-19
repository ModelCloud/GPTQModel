# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5416: reshard keeps a referenced local adapter's weights.

An indexed checkpoint can ship a LoRA payload beside its weights and point its
quantization config at it. Those tensors are not in the base weight index, and
the auxiliary copy skipped every weight-suffixed file, so the resharded
checkpoint kept the adapter reference and ``adapter_config.json`` while the
adapter's safetensors was missing.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel import ShardConfig, ShardLayout, reshard
from tests.test_reshard import _build_source


pytestmark = [pytest.mark.cpu]


def _write_config(source, adapter):
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    if adapter is not None:
        config["quantization_config"] = {
            "bits": 4,
            "group_size": 128,
            "quant_method": "gptq",
            "adapter": adapter,
        }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_adapter_dir(source):
    adapter_dir = source / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(
            8, dtype=torch.float16
        ).reshape(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.arange(
            8, 16, dtype=torch.float16
        ).reshape(4, 2),
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    return adapter_dir, tensors


def test_reshard_copies_the_referenced_adapter_payload(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    adapter_dir, tensors = _write_adapter_dir(source)
    _write_config(source, {"rank": 8, "path": "adapter"})

    out = tmp_path / "out"
    reshard(source, out, config=ShardConfig(layout=ShardLayout.SINGLE))

    copied_dir = out / "adapter"
    assert (copied_dir / "adapter_config.json").is_file()
    copied = copied_dir / "adapter_model.safetensors"
    assert copied.is_file(), "the referenced adapter weights were dropped"

    loaded = load_file(str(copied))
    assert set(loaded) == set(tensors)
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, tensors[name])

    # The reference still points at the copied payload.
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["adapter"]["path"] == "adapter"


def test_reshard_copies_a_single_file_adapter_reference(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
            2, 4, dtype=torch.float16
        ),
    }
    save_file(tensors, str(source / "adapter_model.safetensors"))
    _write_config(source, {"rank": 8, "path": "adapter_model.safetensors"})

    out = tmp_path / "out"
    reshard(source, out, config=ShardConfig(layout=ShardLayout.SINGLE))

    copied = out / "adapter_model.safetensors"
    assert copied.is_file()
    loaded = load_file(str(copied))
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, tensors[name])


def test_reshard_still_skips_unreferenced_weight_files(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    _write_adapter_dir(source)
    _write_config(source, {"rank": 8, "path": "adapter"})
    extras = source / "extras"
    extras.mkdir()
    save_file({"other.weight": torch.ones(2, 2)}, str(extras / "other.safetensors"))

    out = tmp_path / "out"
    reshard(source, out, config=ShardConfig(layout=ShardLayout.SINGLE))

    assert (out / "adapter" / "adapter_model.safetensors").is_file()
    assert not (out / "extras" / "other.safetensors").exists()


def test_reshard_ignores_an_adapter_reference_outside_the_checkpoint(tmp_path):
    source = tmp_path / "source"
    _build_source(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    save_file({"x.weight": torch.ones(2, 2)}, str(outside / "adapter_model.safetensors"))
    _write_config(source, {"rank": 8, "path": "../outside"})

    out = tmp_path / "out"
    reshard(source, out, config=ShardConfig(layout=ShardLayout.SINGLE))

    # The escaping reference is not copied, and the outside file is untouched.
    assert not (out / "adapter_model.safetensors").exists()
    assert sorted(path.name for path in outside.iterdir()) == ["adapter_model.safetensors"]
