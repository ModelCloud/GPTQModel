# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Brumby checkpoint-owned weights and the existing plan consumer contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.brumby import BrumbyQModel
from gptqmodel.models.ports.brumby.configuration_brumby import BrumbyConfig
from gptqmodel.models.ports.brumby.modeling_brumby import BrumbyForCausalLM
from gptqmodel.utils.model import get_layers_with_prefixes


QUANTIZED = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
DENSE = {"input_layernorm", "post_attention_layernorm", "self_attn.g_proj", "self_attn.q_norm", "self_attn.k_norm"}
BLOCKS = [
    ["input_layernorm:!"],
    [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.g_proj:!",
        "self_attn.q_norm:!",
        "self_attn.k_norm:!",
    ],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def _config(layout, **kwargs):
    # Preserve each publisher layout's Hq/Hkv and projected/hidden width ratio.
    values = {
        "vocab_size": 32,
        "hidden_size": 80 if layout == "14b" else 40,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 10 if layout == "14b" else 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "chunk_size": 3,
    }
    values.update(kwargs)
    return BrumbyConfig(**values)


@pytest.mark.parametrize("layout", ["14b", "4b"])
@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("bias", [False, True])
def test_real_checkpoint_weight_owners_and_plan_consumers(layout, method, bias):
    config = _config(layout, attention_bias=bias)
    with torch.device("meta"):
        model = BrumbyForCausalLM(config)
    wrapper = BrumbyQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert MODEL_MAP["brumby"] is BrumbyQModel
    assert BrumbyQModel.extract_layers_node() == ["model.layers"]
    assert len(get_layers_with_prefixes(model, "model.layers")) == 2
    qc = SimpleNamespace(dynamic=None, method=method)
    assert BrumbyQModel.full_layer_modules(model_config=config, include_capture_only=True) == BLOCKS
    simple = {
        name
        for block in BrumbyQModel.simple_layer_modules(config, qc, is_awq_quantize=method == "awq")
        for name in block
    }
    assert simple == QUANTIZED
    owners = QUANTIZED | DENSE
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for index, layer in enumerate(model.model.layers):
        actual = layer.state_dict()
        assert {key.rsplit(".", 1)[0] for key in actual} == owners
        expected_keys = {name + ".weight" for name in owners}
        if bias:
            expected_keys |= {
                "self_attn." + name + ".bias" for name in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            }
        assert set(actual) == expected_keys
        expected |= {f"model.layers.{index}." + key for key in expected_keys}
        for owner in owners:
            spec = plan.find(owner)
            assert spec is not None
            assert spec.quantize == (owner in QUANTIZED)
            assert plan.find(owner + ".undeclared") is None
            assert plan.find("prefix." + owner) is None
            if owner in QUANTIZED:
                assert owner in simple
            else:
                assert not spec.quantize and not spec.capture_only
    assert set(model.state_dict()) == expected
    for path, role in [
        ("self_attn.q_proj", "q"),
        ("self_attn.k_proj", "k"),
        ("self_attn.v_proj", "v"),
        ("self_attn.o_proj", "o"),
        ("mlp.gate_proj", "gate"),
        ("mlp.up_proj", "up"),
        ("mlp.down_proj", "down"),
    ]:
        assert role in plan.find(path).roles
    assert BrumbyQModel.fusable is None


def test_shared_registry_case_stays_explicit_and_unchanged():
    fixture = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    assert fixture["models"]["brumby"] == "BrumbyQModel"
    assert fixture["cases"]["BrumbyQModel/gptq"] == {"layers": ["model.layers"], "blocks": BLOCKS}


@pytest.mark.parametrize(
    "layout,layers,heads,hidden,intermediate", [("14b", 40, 40, 5120, 17408), ("4b", 36, 32, 2560, 9728)]
)
def test_publisher_full_size_meta_checkpoint_inventory(layout, layers, heads, hidden, intermediate):
    config = BrumbyConfig(
        vocab_size=151936,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1000000,
    )
    with torch.device("meta"):
        model = BrumbyForCausalLM(config)
    # Pinned publisher safetensors indexes have these exact keys; no runtime
    # rotary buffers or retention-cache tensors belong to a checkpoint.
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    expected |= {f"model.layers.{i}.{name}.weight" for i in range(layers) for name in QUANTIZED | DENSE}
    assert set(model.state_dict()) == expected
    assert len(expected) == 3 + layers * 12
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (heads * 128, hidden)
    assert model.model.layers[0].self_attn.k_proj.weight.shape == (8 * 128, hidden)
