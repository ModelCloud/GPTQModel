# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint header inventories and live native mixer owners."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.nemotron_h import NemotronHConfig, NemotronHForCausalLM
from tests.port_assets import read_port_asset, read_port_json
from tests.test_nemotron_h_local_port import config, wrapper


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/nemotron_h"
INVENTORIES = read_port_json(PORT / "checkpoint_shapes.json")


def test_inspected_header_inventory_integrity():
    digest = hashlib.sha256(read_port_asset(PORT / "checkpoint_shapes.json")).hexdigest()
    assert digest == "e81db0dcb0fae2ecdbf705deca85836a05bd54d20caa6af71813b818fe459a0e"
    assert {key: len(value["shapes"]) for key, value in INVENTORIES.items()} == {"8b": 311, "nano": 6243, "super": 42683}
    assert [key for key, owner in MODEL_MAP.items() if owner is NemotronHQModel] == ["nemotron_h"]


@pytest.mark.parametrize("variant", ["8b", "nano", "super"])
def test_full_released_checkpoint_shapes_equal_exact_native_meta_owners(variant):
    inspected = INVENTORIES[variant]
    cfg = NemotronHConfig.from_dict(inspected["config"])
    with torch.device("meta"):
        model = NemotronHForCausalLM(cfg)
    assert model.config.num_hidden_layers == {"8b": 52, "nano": 52, "super": 88}[variant]
    actual = {}
    for key, tensor in model.state_dict().items():
        if key.endswith((".experts.up_proj", ".experts.down_proj")):
            for index in range(tensor.shape[0]):
                name = key.replace(".experts.", f".experts.{index}.") + ".weight"
                actual[name] = list(tensor.shape[1:])
        else:
            actual[key] = list(tensor.shape)
    expected = {key.replace("backbone.", "model.", 1): shape for key, shape in inspected["shapes"].items()
                if not key.startswith("mtp.")}
    assert actual == expected
    mtp = {key: shape for key, shape in inspected["shapes"].items() if key.startswith("mtp.")}
    assert len(mtp) == (1040 if variant == "super" else 0)
    if mtp:
        assert cfg.num_nextn_predict_layers == 1
        assert set(model._keys_to_ignore_on_load_unexpected) == {r"mtp.*"}
        assert not hasattr(model, "mtp")


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("latent", [None, 16])
@pytest.mark.parametrize("bias", [False, True])
def test_every_live_checkpoint_tensor_has_a_declared_exact_owner_and_policy(device, heads, latent, bias):
    cfg = config(num_key_value_heads=heads, moe_latent_size=latent, attention_bias=bias,
                 mamba_proj_bias=bias, mlp_bias=bias)
    with torch.device(device):
        model = NemotronHForCausalLM(cfg)
    _convert_model_with_defuser(NemotronHQModel, model, cleanup_original=False)
    actual = wrapper(model)
    plan = actual.module_plan()
    declared = {name for block in actual.simple_layer_modules(cfg, SimpleNamespace(dynamic=None)) for name in block}
    expected_by_type = {
        "linear_attention": {"mixer.in_proj", "mixer.out_proj"},
        "full_attention": {f"mixer.{name}_proj" for name in ["q", "k", "v", "o"]},
        "mlp": {"mixer.up_proj", "mixer.down_proj"},
        "moe": {f"mixer.experts.{index}.{proj}_proj" for index in range(4) for proj in ["up", "down"]}
               | {"mixer.shared_experts.up_proj", "mixer.shared_experts.down_proj"}
               | ({"mixer.fc1_latent_proj", "mixer.fc2_latent_proj"} if latent else set()),
    }
    for layer in model.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0] if key.endswith((".weight", ".bias")) else key
            # Router buffers are owned by the explicitly excluded router.
            if key.endswith("e_score_correction_bias"):
                owner = "mixer.gate"
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                assert owner in declared
                quantized.add(owner)
            else:
                assert "!" in spec.flags
        assert quantized == expected_by_type[layer.block_type]
        assert actual.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert not plan.find("mixer.experts.0.down_proj.undeclared")
    assert "nosrc" in plan.find("mixer.up_proj").flags
    assert "nosrc" in plan.find("mixer.experts.0.up_proj").flags
    assert NemotronHQModel.fusable is None


def test_effective_native_exclusions_do_not_mutate_independently_owned_sibling_trees():
    from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    originals = {owner: json.dumps(owner.module_tree)
                 for owner in (NemotronHQModel, NemotronHPuzzleQModel, NemotronOmniQModel)}
    for owner in [NemotronHQModel, NemotronHPuzzleQModel, NemotronOmniQModel, NemotronHQModel]:
        model_type = {NemotronHQModel: "nemotron_h", NemotronHPuzzleQModel: "nemotron_h_puzzle",
                      NemotronOmniQModel: "nemotron_h_omni"}[owner]
        model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
        tree = owner._resolve_effective_module_tree(model, None)
        model.gptqmodel_effective_module_tree = tree
        assert owner._resolve_effective_module_tree(model, None) == tree
        plan = ModulePlan.from_tree(tree)
        # All three released architectures own this excluded Mamba state.
        assert "!" in plan.find("mixer.A_log").flags
        assert owner.require_trust_remote_code is False
        assert json.dumps(owner.module_tree) == originals[owner]
    for owner, original in originals.items():
        assert json.dumps(owner.module_tree) == original
