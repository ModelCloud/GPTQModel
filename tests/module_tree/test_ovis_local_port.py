# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ovis import OvisQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.ovis.configuration_ovis import OvisConfig
from gptqmodel.models.ports.ovis.modeling_ovis import Ovis
from tests.test_ovis_local_port import LAYOUTS, PORT, tiny_config


EXPECTED_PLAN = {
    "layers": ["llm.model.layers"],
    "blocks": [["input_layernorm:!"], ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
               ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]],
}
LINEARS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
INVENTORY = json.loads(gzip.decompress((PORT / "release_inventory.json.gz").read_bytes()))


def test_exact_native_ovis_absent_and_alias_registry_distinct_from_ovis2():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert "ovis" not in CONFIG_MAPPING_NAMES
    assert "ovis" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    assert CONFIG_MAPPING_NAMES["ovis2"] == "Ovis2Config"
    assert [name for name, definition in MODEL_MAP.items() if definition is OvisQModel] == ["ovis"]


@pytest.mark.parametrize("release", sorted(INVENTORY))
def test_complete_pinned_release_ownership_keys_and_available_header_shapes(release):
    entry = INVENTORY[release]
    config = OvisConfig(**entry["config"])
    with torch.device("meta"):
        model = Ovis(config)
    state = model.state_dict()
    if config.tie_word_embeddings and "llm.lm_head.weight" not in entry["keys"]:
        assert model.llm.lm_head.weight is model.llm.model.embed_tokens.weight
        state.pop("llm.lm_head.weight")
    assert set(state) == set(entry["keys"])
    if entry["shapes"]:
        assert {name: list(value.shape) for name, value in state.items()} == entry["shapes"]
    _convert_model_with_defuser(OvisQModel, model, cleanup_original=False)
    tree = OvisQModel._resolve_effective_module_tree(model, SimpleNamespace(method="gptq"))
    plan = ModulePlan.from_tree(tree)
    for layer in model.llm.model.layers:
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (release, name)
            assert spec.quantize == (owner in LINEARS)
            if spec.quantize:
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert "!" in spec.flags


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kv_heads,bias", [(1, False), (2, True), (4, False)])
def test_actual_cpu_meta_live_module_plans_exclusions_and_plain_awq_policy(layout, kv_heads, bias):
    config = tiny_config(layout, kv_heads, bias)
    with torch.device("meta"):
        model = Ovis(config)
    wrapper = OvisQModel.__new__(OvisQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.effective_module_tree = OvisQModel._resolve_effective_module_tree(model, SimpleNamespace(method="gptq"))
    assert wrapper.extract_layers_node() == EXPECTED_PLAN["layers"]
    groups = wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None))
    assert {name for group in groups for name in group} == LINEARS
    assert wrapper.awq_get_modules_for_scaling(model.llm.model.layers[0], {}, {}) == []
    assert wrapper.lm_head == "llm.lm_head"
    assert wrapper.pre_lm_head_norm_module == "llm.model.norm"
    assert {"visual_tokenizer", "vte", "llm.model.embed_tokens", "llm.model.norm", "llm.model.rotary_emb"} <= set(wrapper.get_base_modules(model))


def test_central_default_registered_plan_is_unchanged():
    fixture = Path(__file__).parent / "fixtures/registered_module_plans.json"
    assert json.loads(fixture.read_text())["cases"]["OvisQModel/gptq"] == EXPECTED_PLAN
