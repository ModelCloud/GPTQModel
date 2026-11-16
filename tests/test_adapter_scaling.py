# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Loaded LoRA adapters and EoRA exports must use PEFT scaling semantics."""

import importlib.util
import json
import re

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


def _peft_pattern_key(pattern_keys, name):
    """PEFT 0.21.2 `get_pattern_key`: `re.match(r"(.*\\.)?(key)$", name)`."""

    for key in pattern_keys:
        if re.match(rf"(.*\.)?({key})$", name):
            return key
    return name


def _peft_scale(*, name, rank_pattern, alpha_pattern, base_rank, base_alpha):
    rank = rank_pattern.get(
        _peft_pattern_key(rank_pattern.keys(), name),
        base_rank,
    )
    alpha = alpha_pattern.get(
        _peft_pattern_key(alpha_pattern.keys(), name),
        base_alpha,
    )
    return rank, alpha / rank


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
        def extract_adapter_rank_patterns(module_ranks=None):
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
        def extract_adapter_rank_patterns(module_ranks=None):
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


@pytest.mark.parametrize(
    "pattern,names,matched_name",
    [
        (
            "1.self_attn.q_proj",
            [
                "model.layers.1.self_attn.q_proj",
                "model.layers.10.self_attn.q_proj",
                "model.layers.11.self_attn.q_proj",
            ],
            "model.layers.1.self_attn.q_proj",
        ),
        (
            "model.layers.1",
            [
                "model.layers.1.self_attn.q_proj",
                "model.layers.10.self_attn.q_proj",
                "model.layers.11.self_attn.q_proj",
            ],
            None,
        ),
        (
            "up_proj",
            [
                "model.layers.0.mlp.gate_up_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.0.mlp.down_proj",
            ],
            "model.layers.0.mlp.up_proj",
        ),
    ],
)
def test_alpha_pattern_matches_peft_dot_boundary(tmp_path, pattern, names, matched_name):
    """A pattern may only select a module whose name ends at a dot boundary."""

    base_rank = 8
    base_alpha = 8
    pattern_alpha = 64
    alpha_pattern = {pattern: pattern_alpha}
    _write_adapter(
        tmp_path,
        config={
            "r": base_rank,
            "lora_alpha": base_alpha,
            "target_modules": ["q_proj", "up_proj"],
            "alpha_pattern": alpha_pattern,
        },
        weights={
            f"base_model.model.{name}.{suffix}": tensor
            for name in names
            for suffix, tensor in (
                ("lora_A.weight", torch.ones(base_rank, 4) * 0.1),
                ("lora_B.weight", torch.ones(3, base_rank) * 0.2),
            )
        },
    )

    for name in names:
        AdapterCache.reset()
        adapter = Lora(rank=base_rank, path=str(tmp_path))
        adapter.post_init(name, CPU)

        peft_rank, peft_scale = _peft_scale(
            name=name,
            rank_pattern={},
            alpha_pattern=alpha_pattern,
            base_rank=base_rank,
            base_alpha=base_alpha,
        )
        assert adapter.rank == peft_rank
        assert abs(adapter.lora_scale - peft_scale) <= 1e-7, (pattern, name)

    # Encode the expected selection, not only parity with the oracle.
    selected = [
        name
        for name in names
        if _peft_pattern_key(alpha_pattern.keys(), name) == pattern
    ]
    if matched_name is None:
        assert selected == []
    else:
        assert selected == [matched_name]


def test_float_alpha_pattern_loads_with_peft_scale(tmp_path):
    """`adapter_config.json` has no integer type; PEFT accepts `4.0`."""

    module_key = "model.layers.0.self_attn.q_projection"
    base_rank = 8
    base_alpha = 8
    float_alpha = 4.0
    _write_adapter(
        tmp_path,
        config={
            "r": base_rank,
            "lora_alpha": base_alpha,
            "target_modules": ["q_projection"],
            "alpha_pattern": {module_key: float_alpha},
        },
        weights={
            f"base_model.model.{module_key}.lora_A.weight": torch.ones(base_rank, 4) * 0.1,
            f"base_model.model.{module_key}.lora_B.weight": torch.ones(3, base_rank) * 0.2,
        },
    )

    adapter = Lora(rank=base_rank, path=str(tmp_path))
    adapter.post_init(module_key, CPU)

    _, peft_scale = _peft_scale(
        name=module_key,
        rank_pattern={},
        alpha_pattern={module_key: float_alpha},
        base_rank=base_rank,
        base_alpha=base_alpha,
    )
    assert adapter.lora_scale == pytest.approx(float_alpha / base_rank)
    assert abs(adapter.lora_scale - peft_scale) <= 1e-7


def test_eora_export_rank_pattern_reproduces_first_match_dynamic(tmp_path):
    """Issue #4199: shadowed and prefix `dynamic` rules must not leak to PEFT."""

    from gptqmodel.adapter.peft import LoraConfig as GptqLoraConfig
    from gptqmodel.quantization.config import QuantizeConfig

    base_rank = 64
    qconfig = QuantizeConfig(
        bits=4,
        group_size=128,
        adapter=Lora(rank=base_rank),
        dynamic={
            r".*\.layers\.0\..*": {"bits": 8},
            r".*\.q_proj": {"adapter": {"rank": 16}},
            r"model\.layers\.1\.self_attn": {"adapter": {"rank": 8}},
        },
    )
    trained = {
        # Shadowed by the first rule, which carries no adapter override.
        "model.layers.0.self_attn.q_proj": base_rank,
        # Selected by a prefix rule that PEFT cannot express directly.
        "model.layers.1.self_attn.k_proj": 8,
        # Unmatched: stays at the base rank, so no pattern entry is needed.
        "model.layers.2.mlp.down_proj": base_rank,
    }

    class _Writer:
        pass

    ModelWriter(_Writer)
    writer = _Writer()
    writer.quantize_config = qconfig
    writer.lora_results = {
        name: Lora(
            rank=rank,
            lora_A=torch.ones(rank, 3),
            lora_B=torch.ones(2, rank),
        )
        for name, rank in trained.items()
    }
    writer.eora_save(save_dir=str(tmp_path), model_save_dir=str(tmp_path / "base"))

    gptq_config = GptqLoraConfig.from_pretrained(
        str(tmp_path), "adapter_config.json",
    )
    assert set(gptq_config.rank_pattern) == {
        r"model\.layers\.1\.self_attn\.k_proj",
    }
    assert gptq_config.alpha_pattern == gptq_config.rank_pattern

    for name, rank in trained.items():
        AdapterCache.reset()
        adapter = Lora(rank=base_rank, path=str(tmp_path))
        adapter.post_init(name, CPU)
        assert adapter.rank == rank, name
