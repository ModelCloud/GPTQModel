# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint inventory and live owner contracts for Puzzle."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.nemotron_h_puzzle import NemotronHPuzzleConfig, NemotronHPuzzleForCausalLM
from tests.port_assets import read_port_json


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/nemotron_h_puzzle"
EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [
        ["norm:!"],
        ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj", "mixer.in_proj",
         "mixer.A_log:!", "mixer.D:!", "mixer.dt_bias:!"],
        ["mixer.o_proj", "mixer.out_proj"],
        ["mixer.conv1d:!"],
        ["mixer.norm:!"],
        ["mixer.gate:!"],
        ["mixer.fc1_latent_proj"],
        ["mixer.experts.{expert_index}.up_proj"],
        ["mixer.experts.{expert_index}.down_proj"],
        ["mixer.fc2_latent_proj"],
        ["mixer.shared_experts.up_proj"],
        ["mixer.shared_experts.down_proj"],
    ],
}


def small_config(**kwargs):
    values = {"vocab_size": 32, "hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
                  "mamba_num_heads": 4, "mamba_head_dim": 4, "mamba_n_groups": 2, "ssm_state_size": 3, "mamba_chunk_size": 4,
                  "n_routed_experts": 4, "n_group": 1, "topk_group": 1, "moe_latent_size": 8,
                  "moe_shared_expert_intermediate_size": 12, "num_experts_per_tok": 1, "moe_intermediate_size": 12,
                  "block_configs": [{"block_type": "mamba"}, {"block_type": "attention"},
                                 {"block_type": "moe", "moe_intermediate_size": 12, "num_experts_per_tok": 1},
                                 {"block_type": "moe", "moe_intermediate_size": 20, "num_experts_per_tok": 2}]}
    values.update(kwargs)
    return NemotronHPuzzleConfig(**values)


def test_puzzle_is_not_the_native_homogeneous_architecture():
    assert "nemotron_h" in CONFIG_MAPPING_NAMES
    assert "nemotron_h" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "nemotron_h_puzzle" not in CONFIG_MAPPING_NAMES
    assert "nemotron_h_puzzle" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert MODEL_MAP["nemotron_h_puzzle"] is NemotronHPuzzleQModel
    assert NemotronHPuzzleQModel.require_trust_remote_code is False
    assert NemotronHPuzzleQModel.require_monkeypatch is False
    assert NemotronHPuzzleQModel.require_pkgs == []
    with torch.device("meta"):
        assert type(NemotronHPuzzleQModel.loader.from_config(small_config())) is NemotronHPuzzleForCausalLM


def test_pinned_full_released_decoder_inventory_and_shapes():
    source = read_port_json(PORT / "checkpoint_shapes.json")
    assert source["revision"] == "8fe5546888e9bd03fdbf52d808adebdfca901b52"
    assert len(source["header_sha256"]) == 32
    shapes = source["shapes"]
    assert len(shapes) == 42683
    mtp = {k: v for k, v in shapes.items() if k.startswith("mtp.")}
    assert len(mtp) == 1040
    assert NemotronHPuzzleQModel.out_of_model_tensors == {"prefixes": ["mtp"]}
    config = NemotronHPuzzleConfig.from_dict(read_port_json(PORT / "checkpoint_config.json"))
    with torch.device("meta"):
        model = NemotronHPuzzleForCausalLM(config)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert actual == {k: v for k, v in shapes.items() if not k.startswith("mtp.")}
    assert len(actual) == 41643
    assert len(model.model.layers) == 88
    assert [layer.block_type for layer in model.model.layers].count("mamba") == 40
    assert [layer.block_type for layer in model.model.layers].count("moe") == 40
    assert [layer.block_type for layer in model.model.layers].count("attention") == 8
    _assert_all_owners(model)


def _assert_all_owners(model):
    wrapped = NemotronHPuzzleQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapped.module_plan()
    for layer in model.model.layers:
        for key in layer.state_dict():
            direct = key in ("mixer.A_log", "mixer.D", "mixer.dt_bias")
            owner = key if direct else key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, owner
            if direct or owner in ("norm", "mixer.norm", "mixer.conv1d", "mixer.gate"):
                assert not spec.quantize, owner
            else:
                assert spec.quantize, owner
                assert isinstance(layer.get_submodule(owner), nn.Linear), owner
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                assert plan.find(name).quantize, name
    assert plan.find("mixer.not_a_projection") is None
    assert plan.find("mixer.experts.not_an_index.up_proj") is None
    assert plan.find("mixer.q_proj.unlisted") is None
    for path in ("mixer.experts.0.up_proj", "mixer.shared_experts.up_proj"):
        assert "nosrc" in plan.find(path).flags
    assert plan.is_expert("mixer.experts.3.up_proj")
    assert not plan.is_expert("mixer.shared_experts.up_proj")


@pytest.mark.parametrize(
    "device,kv_heads,latent,bias", [("meta", 1, None, False), ("cpu", 2, 8, True), ("meta", 4, 8, False)]
)
def test_real_heterogeneous_live_owners(device, kv_heads, latent, bias):
    config = small_config(num_key_value_heads=kv_heads, moe_latent_size=latent, attention_bias=bias, mlp_bias=bias,
                          use_bias=bias)
    with torch.device(device):
        model = NemotronHPuzzleForCausalLM(config)
    assert model.model.layers[2].mixer.experts[0].up_proj.out_features == 12
    assert model.model.layers[3].mixer.experts[0].up_proj.out_features == 20
    assert model.model.layers[2].mixer.gate.config.num_experts_per_tok == 1
    assert model.model.layers[3].mixer.gate.config.num_experts_per_tok == 2
    _assert_all_owners(model)


def test_explicit_plan_and_plain_awq_policy():
    tree = NemotronHPuzzleQModel._resolve_effective_module_tree(SimpleNamespace(), SimpleNamespace(method="gptq"))
    assert NemotronHPuzzleQModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert NemotronHPuzzleQModel.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    plan = ModulePlan.from_tree(tree)
    assert plan.find("mixer.gate") is not None
    with torch.device("meta"):
        native = NemotronHPuzzleForCausalLM(small_config())
    wrapped = NemotronHPuzzleQModel(model=native, quantized=False, quantize_config=None)
    assert all(wrapped.awq_get_modules_for_scaling(layer, {}, {}) == [] for layer in native.model.layers)
    assert NemotronHPuzzleQModel.fusable is None
    with pytest.raises(ValueError):
        require_fusable_config(NemotronHPuzzleQModel, "nemotron_h_puzzle")


@pytest.mark.parametrize("changes,reason", [
    ({"block_configs": []}, "at least one"),
    ({"block_configs": [{"block_type": "not-puzzle"}]}, "block_configs"),
    ({"block_configs": [{"block_type": "moe", "num_experts_per_tok": -1}]}, "positive integer"),
    ({"block_configs": [{"block_type": "moe", "unreviewed": 7}]}, "Unknown Puzzle"),
    ({"mtp_block_configs": [{"block_type": "mamba"}]}, "MTP"),
    ({"num_key_value_heads": 3}, "divisible"),
    ({"mamba_n_groups": 3}, "divisible"),
    ({"n_routed_experts": 3, "n_group": 2}, "routing"),
    ({"topk_group": 2}, "topk_group"),
])
def test_invalid_config_rejected_and_valid_retry(changes, reason):
    with pytest.raises((ValueError, AttributeError), match=reason):
        small_config(**changes)
    assert NemotronHPuzzleForCausalLM(small_config()).model.layers[2].block_type == "moe"
