# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Loaded LoRA adapters and EoRA exports must use PEFT scaling semantics."""

import importlib.util
import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.models.writer import ModelWriter


CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


def _write_adapter(directory, *, config, weights):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    save_file(weights, str(directory / "adapter_model.safetensors"))


@pytest.mark.parametrize("use_rslora", [False, True])
def test_loaded_lora_uses_global_alpha_scale(tmp_path, use_rslora):
    module_key = "model.layers.0.self_attn.q_proj"
    rank = 8
    alpha = 32
    source_a = torch.eye(4, dtype=torch.float32).repeat(2, 1)
    source_b = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 5
    _write_adapter(
        tmp_path,
        config={
            "r": rank,
            "lora_alpha": alpha,
            "use_rslora": use_rslora,
            "target_modules": ["q_proj"],
        },
        weights={
            f"base_model.model.{module_key}.lora_A.weight": source_a,
            f"base_model.model.{module_key}.lora_B.weight": source_b,
        },
    )

    adapter = Lora(rank=rank, path=str(tmp_path))
    adapter.post_init(module_key, CPU)
    x = torch.tensor([[0.5, -1.0, 2.0, -3.0]], dtype=torch.float32)
    base = torch.zeros((1, 2), dtype=torch.float32)
    result = adapter.apply(x, base)

    scale = alpha / rank**0.5 if use_rslora else alpha / rank
    oracle = scale * (x.double() @ source_a.T.double()) @ source_b.T.double()
    torch.testing.assert_close(result.double(), oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("use_rslora", [False, True])
def test_loaded_lora_uses_per_module_peft_scale(tmp_path, use_rslora):
    module_key = "model.layers.1.self_attn.q_proj"
    base_rank = 8
    module_rank = 4
    base_alpha = 8
    module_alpha = 3
    in_features = 3
    out_features = 2

    source_a = torch.arange(
        module_rank * in_features,
        dtype=torch.float32,
    ).reshape(module_rank, in_features) / 10
    source_b = torch.arange(
        out_features * module_rank,
        dtype=torch.float32,
    ).reshape(out_features, module_rank) / 7
    _write_adapter(
        tmp_path,
        config={
            "r": base_rank,
            "lora_alpha": base_alpha,
            "use_rslora": use_rslora,
            "target_modules": ["q_proj"],
            "rank_pattern": {module_key: module_rank},
            "alpha_pattern": {module_key: module_alpha},
        },
        weights={
            f"base_model.model.{module_key}.lora_A.weight": source_a,
            f"base_model.model.{module_key}.lora_B.weight": source_b,
        },
    )

    adapter = Lora(rank=base_rank, path=str(tmp_path))
    adapter.post_init(module_key, CPU)

    x = torch.tensor([[0.25, -0.5, 1.5]], dtype=torch.float32)
    base = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    result = adapter.apply(x, base)

    scale = (
        module_alpha / module_rank**0.5
        if use_rslora
        else module_alpha / module_rank
    )
    source_delta = (x.double() @ source_a.T.double()) @ source_b.T.double()
    oracle = base.double() + scale * source_delta

    assert adapter.rank == module_rank
    torch.testing.assert_close(result.double(), oracle, rtol=1e-6, atol=1e-6)


def test_eora_dynamic_rank_export_round_trips_at_unit_scale(tmp_path):
    global_rank = 16
    module_rank = 4
    module_key = "model.layers.1.self_attn.q_proj"
    pattern = {"model.layers.1.*": module_rank}

    class _QuantizeConfig:
        adapter = Lora(rank=global_rank)
        dynamic = {"+:model.layers.1.*": {"adapter": {"rank": module_rank}}}

        @staticmethod
        def extract_adapter_rank_patterns():
            return pattern

    class _Writer:
        pass

    ModelWriter(_Writer)
    writer = _Writer()
    writer.quantize_config = _QuantizeConfig()
    source_a = torch.arange(module_rank * 3, dtype=torch.float32).reshape(
        module_rank, 3
    ) / 9
    source_b = torch.arange(2 * module_rank, dtype=torch.float32).reshape(
        2, module_rank
    ) / 11
    writer.lora_results = {
        module_key: Lora(rank=module_rank, lora_A=source_a, lora_B=source_b)
    }
    writer.eora_save(save_dir=str(tmp_path), model_save_dir=str(tmp_path / "base"))

    config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["rank_pattern"] == pattern
    assert config["alpha_pattern"] == pattern

    loaded = Lora(rank=global_rank, path=str(tmp_path))
    loaded.post_init(module_key, CPU)
    x = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    base = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    result = loaded.apply(x, base)
    oracle = base.double() + (
        x.double() @ source_a.T.double()
    ) @ source_b.T.double()
    torch.testing.assert_close(result.double(), oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("peft") is None,
    reason="PEFT is not installed",
)
def test_eora_export_alpha_pattern_matches_peft_module_scale(tmp_path):
    global_rank = 16
    module_rank = 4
    pattern = {"model.layers.1.*": module_rank}

    class _QuantizeConfig:
        adapter = Lora(rank=global_rank)
        dynamic = {"+:model.layers.1.*": {"adapter": {"rank": module_rank}}}

        @staticmethod
        def extract_adapter_rank_patterns():
            return pattern

    class _Writer:
        pass

    ModelWriter(_Writer)
    writer = _Writer()
    writer.quantize_config = _QuantizeConfig()
    writer.lora_results = {
        "model.layers.1.self_attn.q_proj": Lora(
            rank=module_rank,
            lora_A=torch.ones(module_rank, 3),
            lora_B=torch.ones(2, module_rank),
        )
    }
    writer.eora_save(save_dir=str(tmp_path), model_save_dir=str(tmp_path / "base"))

    from peft import LoraConfig as PeftLoraConfig

    peft_config = PeftLoraConfig.from_pretrained(str(tmp_path))
    assert (
        peft_config.alpha_pattern["model.layers.1.*"]
        / peft_config.rank_pattern["model.layers.1.*"]
        == 1.0
    )
