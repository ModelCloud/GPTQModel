# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Real Falcon decoder owners, including the legacy 40B dual-norm layout."""

from pathlib import Path

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from transformers import FalconConfig, FalconForCausalLM

from gptqmodel.models.auto import MODEL_MAP_CASEFOLD
from gptqmodel.models.definitions.rw import FalconQModel, RwgQModel
from gptqmodel.models.fusable import require_fusable_config
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.rw import RWConfig
from tests.port_assets import read_port_json


PORT = Path(__file__).parents[2] / "gptqmodel/models/ports/rw"
RELEASES = read_port_json(PORT / "checkpoint_inventory.json")
EXPECTED_PLAN = {
    "layers": ["transformer.h"],
    "blocks": [
        ["input_layernorm:!"], ["ln_attn:!"], ["ln_mlp:!"],
        ["self_attention.query_key_value"], ["self_attention.dense"],
        ["post_attention_layernorm:!"], ["mlp.dense_h_to_4h"], ["mlp.dense_4h_to_h"],
    ],
}


@pytest.mark.parametrize("alias", ["falcon", "refinedweb", "refinedwebmodel"])
def test_alias_binding_has_no_unqualified_fusion_admission(alias):
    definition = MODEL_MAP_CASEFOLD[alias]
    assert definition is (FalconQModel if alias == "falcon" else RwgQModel)
    assert definition.require_trust_remote_code is False
    with pytest.raises(ValueError):
        require_fusable_config(definition, alias)


@pytest.mark.parametrize("definition", [RwgQModel, FalconQModel])
def test_exact_parent_registry_proposal(definition):
    tree = definition._resolve_effective_module_tree(None, None)
    assert definition.build_layer_modules(tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    with definition._module_tree_context(tree):
        assert definition.extract_layers_node() == EXPECTED_PLAN["layers"]


@pytest.mark.parametrize("release", sorted(RELEASES))
def test_pinned_real_checkpoint_inventory_meta_and_source_layout(release):
    fixture = RELEASES[release]
    config = RWConfig(**fixture["config"])
    with torch.device("meta"):
        model = RwgQModel.loader.from_config(config)
    state = model.state_dict()
    # Independently retrieved current in-library release metadata must have
    # exactly the same physical geometry as the pinned historical config.
    with torch.device("meta"):
        canonical = FalconQModel.loader.from_config(FalconConfig(**fixture["native_config"]))
    assert {key: list(value.shape) for key, value in canonical.state_dict().items()} == {
        key: list(value.shape) for key, value in state.items()
    }
    if "keys" in fixture:
        assert sorted(state) == fixture["keys"]
    if "shapes" in fixture:
        assert {key: list(val.shape) for key, val in state.items()} == fixture["shapes"]
    assert (config.num_hidden_layers, config.num_attention_heads) == {
        "falcon-rw-1b": (24, 32), "falcon-7b-instruct": (32, 71), "falcon-40b-instruct": (60, 128)
    }[release]
    tree = RwgQModel._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    for key in state:
        if key.startswith("transformer.h."):
            relative = key.split(".", 3)[-1].rsplit(".", 1)[0]
            assert plan.find(relative) is not None, key
    heads, dim = config.num_attention_heads, config.head_dim
    expected_rows = (heads + 2 * config.num_kv_heads) * dim if config.new_decoder_architecture else (
        (heads + 2) * dim if config.multi_query else 3 * heads * dim)
    assert state["transformer.h.0.self_attention.query_key_value.weight"].shape == (expected_rows, config.hidden_size)


@pytest.mark.parametrize("grouped,parallel,norms", [(False, False, 1), (False, True, 1), (True, True, 1), (True, True, 2)])
def test_input_tags_follow_actual_parallel_branch_inputs(grouped, parallel, norms):
    config = FalconConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                          num_kv_heads=2, new_decoder_architecture=grouped, parallel_attn=parallel,
                          num_ln_in_parallel_attn=norms)
    model = FalconForCausalLM(config).eval()
    _convert_model_with_defuser(FalconQModel, model, cleanup_original=False)
    layer = model.transformer.h[0]
    captured = {}

    def capture(name):
        def hook(module, args):
            captured[name] = args[0].detach().clone()
        return hook

    handles = [layer.self_attention.query_key_value.register_forward_pre_hook(capture("qkv")),
               layer.mlp.dense_h_to_4h.register_forward_pre_hook(capture("mlp"))]
    # Make dual norms observably different; at initialization both are identity.
    if norms == 2:
        with torch.no_grad():
            layer.ln_mlp.weight.mul_(1.4)
    try:
        with torch.no_grad():
            model(torch.tensor([[1, 4, 6, 7]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    shared = parallel and norms == 1
    assert torch.equal(captured["qkv"], captured["mlp"]) is shared
    plan = ModulePlan.from_tree(FalconQModel._resolve_effective_module_tree(model, None))
    for path in ("self_attention.query_key_value", "mlp.dense_h_to_4h"):
        assert ("in=x" in plan.find(path).flags) is shared
    assert "nosrc" in plan.find("mlp.dense_h_to_4h").flags


@pytest.mark.parametrize("definition", [RwgQModel, FalconQModel])
@pytest.mark.parametrize("grouped,parallel", [(False, False), (False, True), (True, True)])
def test_every_native_checkpoint_owner_is_declared(definition, grouped, parallel):
    config = FalconConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_kv_heads=2, new_decoder_architecture=grouped, parallel_attn=parallel, bias=True,
    )
    model = FalconForCausalLM(config).eval()
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    tree = definition._resolve_effective_module_tree(model, None)
    plan = ModulePlan.from_tree(tree)
    declared = {x.split(":", 1)[0] for block in definition.build_layer_modules(tree) for x in block}
    for key in model.transformer.h[0].state_dict():
        owner = key.rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, (definition.__name__, key)
        if spec.quantize:
            assert owner in declared
        else:
            assert set(spec.flags) & {"!", "?"}
    assert plan.find("self_attention.query_key_value.undeclared") is None
    assert plan.find("other.query_key_value") is None


QKV, MLP_IN = "self_attention.query_key_value", "mlp.dense_h_to_4h"


@pytest.mark.parametrize("definition", [RwgQModel, FalconQModel])
@pytest.mark.parametrize("grouped,parallel,norms,expected", [
    # Sequential (falcon-rw-1b): two pre-norms, GELU blocks dense_h_to_4h as a source.
    (False, False, None, [("input_layernorm", [QKV], []), ("post_attention_layernorm", [MLP_IN], [])]),
    # Parallel single norm (falcon-7b): one norm feeds both branches.
    (False, True, None, [("input_layernorm", [QKV, MLP_IN], [])]),
    (True, True, 1, [("input_layernorm", [QKV, MLP_IN], [])]),
    # Dual norm (falcon-40b/180b): each branch has its own producer.
    (True, True, 2, [("ln_attn", [QKV], []), ("ln_mlp", [MLP_IN], [])]),
    (True, True, None, [("ln_attn", [QKV], []), ("ln_mlp", [MLP_IN], [])]),
])
@pytest.mark.parametrize("alibi", [False, True])
def test_awq_scale_folds_preserve_every_decoder_layer_output(definition, grouped, parallel, norms, expected, alibi):
    torch.manual_seed(13)
    config = FalconConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                          num_kv_heads=2 if grouped else 4, multi_query=parallel and not grouped,
                          new_decoder_architecture=grouped, parallel_attn=parallel,
                          num_ln_in_parallel_attn=norms, bias=True, alibi=alibi)
    config._attn_implementation = "eager"
    model = definition.loader.from_config(config).eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.startswith("transformer.h.") and ("ln_" in name or "layernorm" in name):
                value.uniform_(0.5, 1.5)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    tree = definition._resolve_effective_module_tree(model, None)
    ids = torch.tensor([[1, 4, 6, 9, 3, 8, 5, 7]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    # The fused query_key_value rows are never the dense input, so the only
    # sources are norms. ALiBi attention replay is not qualified, so its
    # query_key_value consumers are left unscaled.
    if alibi:
        expected = [group for group in expected if QKV not in group[1]]
    if not expected:
        definition_wrapper = definition.__new__(definition)
        torch.nn.Module.__init__(definition_wrapper)
        definition_wrapper.__dict__.update(model=model, effective_module_tree=tree, module_tree=tree)
        for layer in model.transformer.h:
            features = {name: torch.randn(4, 32) for name in (QKV, MLP_IN)}
            assert definition_wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
        return
    for layer in model.transformer.h:
        assert assert_awq_folds_neutral(definition, model, layer, forward, tree=tree) == expected
