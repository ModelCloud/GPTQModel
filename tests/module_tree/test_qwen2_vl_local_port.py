# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.native.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig
from tests.port_assets import PORTS, read_port_json


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/ports/qwen2_vl"
BLOCKS = [
    ["input_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


@pytest.mark.parametrize("text", [False, True])
def test_pinned_checkpoint_shapes_and_live_plan_ownership(text):
    # Sources are headers, not a tree-generated approximation of a checkpoint.
    shapes = read_port_json(PORTS / "qwen2_vl" / "checkpoint_shapes.json")[
        "text" if text else "composite"
    ]
    config_class = Qwen2VLTextConfig if text else Qwen2VLConfig
    config = config_class.from_pretrained(FIXTURES / ("released_text_config.json" if text else "released_config.json"))
    with torch.device("meta"):
        model = Qwen2VLQModel.loader.from_config(config)
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    if not text:
        expected = {}
        for key, value in shapes.items():
            if key.startswith("model."):
                key = "model.language_model." + key.removeprefix("model.")
            elif key.startswith("visual."):
                key = "model." + key
            expected[key] = value
        # The official 2B source shares this parameter and omits its head alias.
        expected["lm_head.weight"] = shapes["model.embed_tokens.weight"]
    else:
        expected = shapes
        assert not any("lm_head" in key for key in actual)
    assert actual == expected
    tree = Qwen2VLQModel._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    prefix = "layers." if text else "model.language_model.layers."
    with Qwen2VLQModel._module_tree_context(tree):
        assert Qwen2VLQModel.extract_layers_node() == (
            ["layers"] if text else ["model.language_model.layers", "language_model.layers"]
        )
    for name in actual:
        owner = name.rsplit(".", 1)[0]
        if owner.startswith(prefix):
            relative = owner.removeprefix(prefix).split(".", 1)[1]
            spec = plan.find(relative)
            assert spec is not None, owner
            assert spec.quantize == ("layernorm" not in owner)
            if spec.quantize:
                assert isinstance(model.get_submodule(owner), torch.nn.Linear)
            assert plan.find(relative + ".undeclared") is None
        else:
            assert plan.find(owner) is None
            assert owner in (
                {"embed_tokens", "norm"}
                if text
                else {"lm_head", "model.language_model.embed_tokens", "model.language_model.norm"}
            ) or owner.startswith("model.visual.")


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_text_layout_and_tagged_bias_owners(kv_heads, device):
    config = Qwen2VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={"mrope_section": [1, 1, 2]},
    )
    with torch.device(device):
        model = Qwen2VLQModel.loader.from_config(config)
    tree = Qwen2VLQModel._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    assert Qwen2VLQModel.build_layer_modules(tree, include_capture_only=True) == BLOCKS
    assert Qwen2VLQModel.get_base_modules(model) == ["embed_tokens", "norm", "rotary_emb"]
    for key, value in model.state_dict().items():
        if key.startswith("layers."):
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner.split(".", 2)[2])
            assert spec is not None
            if key.endswith("bias"):
                assert spec.quantize and any(owner.endswith(f"{part}_proj") for part in "qkv")
        assert value.device.type == device


def test_default_registry_case_stays_composite():
    tree = Qwen2VLQModel._resolve_effective_module_tree(SimpleNamespace(), None)
    assert tree == Qwen2VLQModel.module_tree
    assert "qwen2_vl_text" not in Qwen2VLQModel.fusable.model_types
