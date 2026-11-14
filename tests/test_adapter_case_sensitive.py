# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""PEFT adapter module names are case-sensitive.

MPT (`Wqkv`) and Baichuan (`W_pack`) projections use mixed-case module names.
Loading an adapter must match the canonical name, and EoRA export must write the
canonical name into both the safetensors keys and ``target_modules`` so standard
PEFT consumers can match it. Legacy GPT-QModel exports that lowercased module
names remain loadable through an explicit fallback.
"""

import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from gptqmodel.adapter.adapter import Lora
from gptqmodel.models.writer import ModelWriter


def _write_adapter(directory, module_key, *, target_module, rank=2):
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [target_module]})
    )
    save_file(
        {
            f"base_model.model.{module_key}.lora_A.weight": torch.arange(
                rank * 4, dtype=torch.float16
            ).reshape(rank, 4),
            f"base_model.model.{module_key}.lora_B.weight": torch.arange(
                3 * rank, dtype=torch.float16
            ).reshape(3, rank),
        },
        str(directory / "adapter_model.safetensors"),
    )


def test_post_init_loads_canonical_case_module_name(tmp_path):
    _write_adapter(tmp_path, "transformer.blocks.0.atten.Wqkv", target_module="Wqkv")
    adapter = Lora(rank=2, path=str(tmp_path))

    adapter.post_init("transformer.blocks.0.atten.Wqkv", torch.device("cpu"))

    assert adapter.lora_A.shape == (4, 2)
    assert adapter.lora_B.shape == (2, 3)


def test_post_init_falls_back_to_legacy_lowercase_keys(tmp_path):
    _write_adapter(tmp_path, "transformer.blocks.0.atten.wqkv", target_module="wqkv")
    adapter = Lora(rank=2, path=str(tmp_path))

    adapter.post_init("transformer.blocks.0.atten.Wqkv", torch.device("cpu"))

    assert adapter.lora_A.shape == (4, 2)
    assert adapter.lora_B.shape == (2, 3)


def _export(tmp_path, module_name, adapter):
    class _Writer:
        pass

    ModelWriter(_Writer)
    writer = _Writer()
    writer.quantize_config = SimpleNamespace(adapter=Lora(rank=adapter.rank), dynamic=None)
    writer.lora_results = {module_name: adapter}
    writer.eora_save(save_dir=str(tmp_path), model_save_dir=str(tmp_path / "base"))


def test_eora_export_preserves_canonical_module_case(tmp_path):
    adapter = Lora(
        rank=2,
        lora_A=torch.ones(2, 4, dtype=torch.float16),
        lora_B=torch.ones(3, 2, dtype=torch.float16),
    )

    _export(tmp_path, "transformer.blocks.0.atten.Wqkv", adapter)

    weights = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert "base_model.model.transformer.blocks.0.atten.Wqkv.lora_A.weight" in weights
    assert "base_model.model.transformer.blocks.0.atten.Wqkv.lora_B.weight" in weights
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["target_modules"] == ["Wqkv"]


def test_eora_export_round_trips_canonical_module_case(tmp_path):
    adapter = Lora(
        rank=2,
        lora_A=torch.arange(8, dtype=torch.float16).reshape(2, 4),
        lora_B=torch.arange(6, dtype=torch.float16).reshape(3, 2),
    )

    _export(tmp_path, "transformer.blocks.0.atten.Wqkv", adapter)

    loaded = Lora(rank=2, path=str(tmp_path))
    loaded.post_init("transformer.blocks.0.atten.Wqkv", torch.device("cpu"))

    torch.testing.assert_close(loaded.lora_A, adapter.lora_A.T)
    torch.testing.assert_close(loaded.lora_B, adapter.lora_B.T)
