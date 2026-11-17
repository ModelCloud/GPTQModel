# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Hymba checkpoint and tiny-layout module-plan contracts (no downloads)."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.hymba.configuration_hymba import HymbaConfig
from gptqmodel.models.ports.hymba.modeling_hymba import HymbaForCausalLM


PORT = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/hymba"
QC = SimpleNamespace(dynamic=None)


def _config(**kwargs):
    values = {"vocab_size": 37, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 3,
              "num_attention_heads": 2, "num_key_value_heads": 1, "mamba_d_state": 3,
              "mamba_dt_rank": 2, "num_experts": 1, "num_experts_per_tok": 1,
              "num_memory_tokens": 2, "kv_reuse_group": [[0, 1]], "rope": True}
    values.update(kwargs)
    return HymbaConfig(**values)


def _quantized(config):
    return {entry.split(":", 1)[0] for block in HymbaQModel.simple_layer_modules(config, QC) for entry in block}


def _audit_layer(layer, config):
    plan = ModulePlan.from_tree(HymbaQModel.module_tree)
    quantized = _quantized(config)
    modules = dict(layer.named_modules())
    owners = set()
    for key in layer.state_dict():
        owner = key.rsplit(".", 1)[0]
        assert owner in modules, key
        owners.add(owner)
        spec = plan.find(owner)
        assert spec is not None, key
        if spec.quantize:
            assert owner in quantized, key
            assert isinstance(modules[owner], torch.nn.Linear), owner
        else:
            assert "!" in spec.flags or "?" in spec.flags, key
            assert owner not in quantized, key
    return owners


def test_published_1_5b_state_dict_matches_every_checkpoint_key_and_shape():
    config = HymbaConfig(**json.loads((PORT / "upstream_config.json").read_text()))
    with torch.device("meta"):
        model = HymbaQModel.loader.from_config(config)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    expected = json.loads((PORT / "checkpoint_shapes.json").read_text())
    # The official safetensors header stores the tied embedding once. HF's
    # state_dict exposes its lm_head alias too; it is not a missing weight.
    assert model.lm_head.weight is model.model.embed_tokens.weight
    lm_head_shape = actual.pop("lm_head.weight")
    assert lm_head_shape == expected["model.embed_tokens.weight"]
    assert actual == expected
    assert len(model.model.layers) == config.num_hidden_layers == 32
    for layer in model.model.layers:
        _audit_layer(layer, config)


@pytest.mark.parametrize("options", [
    {}, {"kq_norm": "rms"}, {"kq_norm": "perhead-rms"},
    {"mamba_inner_layernorms": False, "mamba_proj_bias": True, "mamba_conv_bias": False},
    {"mlp_hidden_act": "relu2"}, {"num_experts": 3, "num_experts_per_tok": 2},
    {"intermediate_size": 0, "num_memory_tokens": 0},
    {"kv_reuse_group": None, "kv_reuse_every_i_layer": 3},
    {"kv_reuse_group": None, "kv_reuse_every_i_layer": -1, "kv_weight_reuse": True},
])
def test_all_layout_weight_owners_survive_actual_defuser_conversion(options):
    config = _config(**options)
    with torch.device("meta"):
        model = HymbaForCausalLM(config)
    before = set(model.state_dict())
    _convert_model_with_defuser(HymbaQModel, model, cleanup_original=False)
    assert set(model.state_dict()) == before
    owners = set.union(*(_audit_layer(layer, config) for layer in model.model.layers))
    assert {"mamba.A_log", "mamba.D", "mamba.conv1d", "mamba.x_proj.0", "mamba.dt_proj.0"} <= owners
    assert HymbaQModel.extract_layers_node() == ["model.layers"]
    assert {"mamba.in_proj", "mamba.out_proj"} <= _quantized(config)
    if config.intermediate_size:
        for i in range(config.num_experts):
            assert {f"moe.experts.{i}.up_proj", f"moe.experts.{i}.down_proj"} <= _quantized(config)


def test_registry_aliases_moe_hooks_and_looper_consumers():
    assert {key for key, cls in MODEL_MAP.items() if cls is HymbaQModel} == {"hymba"}
    plan = ModulePlan.from_tree(HymbaQModel.module_tree)
    assert [spec.template for spec in plan.moe_roots()] == ["moe"]
    assert plan.expert_container_leaf("moe") == "experts"
    assert plan.roles("moe.experts.2.gate_proj") == ("gate",)
    assert plan.roles("moe.experts.2.up_proj") == ("up",)
    assert plan.roles("moe.experts.2.down_proj") == ("down",)
    assert "nosrc" in plan.find("mamba.in_proj").flags
    assert "nosrc" in plan.find("mamba.out_proj").flags
    assert plan.find("moe.router").quantize is False
    config = _config(num_experts=3, num_experts_per_tok=2)
    with torch.device("meta"):
        model = HymbaForCausalLM(config)
    hooks = copy.copy(HymbaQModel.moe_lifecycle_hooks)
    hooks.module_plan = plan
    block = hooks.get_moe_block(model.model.layers[0], HymbaQModel)
    assert block is model.model.layers[0].moe
    assert hooks.get_experts_module(block, HymbaQModel) is block.experts
    for cls in (ModuleLooper, WeightOnlyLooper):
        consumer = cls.__new__(cls)
        consumer.gptq_model = SimpleNamespace(module_plan=lambda: plan)
        assert consumer._extract_moe_group_key("moe.experts.2.gate_proj") == "moe.experts.2"
        assert consumer._extract_moe_group_key("mamba.in_proj") is None
        assert consumer._extract_moe_group_key("moe.experts.2.gate_proj.extra") is None


def test_explicit_integrator_registry_fixture():
    # Parent owns the central registry JSON; this independent local fixture
    # pins the exact replacement entry that it must install during integration.
    expected = json.loads((PORT / "module_plan_fixture.json").read_text())["HymbaQModel/gptq"]
    assert expected["layers"] == HymbaQModel.extract_layers_node()
    assert expected["blocks"] == HymbaQModel.build_layer_modules(HymbaQModel.module_tree, include_capture_only=True)
