# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native dense/routed/shared ownership and neutral AWQ scale folds."""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models.definitions.ernie4_5_moe import Ernie4_5_MoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.model import find_modules
from tests.test_ernie4_5_moe_local_port import tiny_config, tiny_model


@pytest.fixture(params=[(0, False, 1, 1), (1, True, 1, 0), (2, False, 2, 1)])
def case(request):
    shared, bias, interval, start = request.param
    model = tiny_model(moe_num_shared_experts=shared, use_bias=bias,
                       moe_layer_interval=interval, moe_layer_start_index=start)
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    return Ernie4_5_MoeQModel(model=model, quantized=False, quantize_config=None)


def test_every_checkpoint_owned_tensor_has_an_explicit_plan_owner(case):
    plan = case.module_plan()
    blocks = case.simple_layer_modules(model_config=case.model.config, quantize_config=SimpleNamespace(dynamic=None))
    declared = {name.split(":", 1)[0] for block in blocks for name in block}
    assert not any("{expert_index}" in name for name in declared)
    resolved = set()
    for layer in case.model.model.layers:
        linears = find_modules(layer)
        resolved.update(linears)
        assert layer.state_dict()
        for key in layer.state_dict():
            owner = key.rpartition(".")[0]
            spec = plan.find(owner)
            assert spec is not None, f"undeclared checkpoint owner: {owner}"
            if isinstance(layer.get_submodule(owner), torch.nn.Linear):
                assert spec.quantize and owner in declared, owner
            else:
                assert not spec.quantize and "!" in spec.flags, owner
    absent = set()
    if case.model.config.moe_num_shared_experts == 0:
        absent.update(f"mlp.shared_experts.{projection}" for projection in ("gate_proj", "up_proj", "down_proj"))
    if all(hasattr(layer.mlp, "experts") for layer in case.model.model.layers):
        absent.update(f"mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj"))
    assert declared - resolved == absent


def test_plan_roles_expert_families_and_looper_consumer(case):
    plan = case.module_plan()
    assert tuple(plan.expert_containers()) == ("mlp.experts",)
    assert tuple(spec.template for spec in plan.moe_roots()) == ("mlp",)
    assert tuple(plan.shared_expert_containers()) == ("mlp.shared_experts",)
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = case
    for index in range(case.model.config.moe_num_experts):
        for projection, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            path = f"mlp.experts.{index}.{projection}"
            assert plan.roles(path) == (role,)
            assert plan.find(path).quantize
            assert looper._extract_moe_group_key(path) is not None
    for role in "qkvo":
        assert plan.roles(f"self_attn.{role}_proj") == (role,)
    assert not plan.find("mlp.gate").quantize
    assert not plan.find("mlp.gate.moe_statics").quantize
    for unknown in ("mlp.experts.0.up_proj_extra", "mlp.experts.0.up_proj.child", "mlp.gate.moe_statics_extra"):
        assert plan.find(unknown) is None


@pytest.mark.parametrize("kv_heads", [2, 4], ids=["gqa", "mha"])
def test_awq_folds_are_neutral_on_dense_and_moe_layers(case, kv_heads):
    if kv_heads != case.model.config.num_key_value_heads:
        config = case.model.config
        model = tiny_model(moe_num_shared_experts=config.moe_num_shared_experts, use_bias=config.use_bias,
                           moe_layer_interval=config.moe_layer_interval,
                           moe_layer_start_index=config.moe_layer_start_index, num_key_value_heads=kv_heads)
        assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    else:
        model = case.model
    ids = torch.tensor([[1, 7, 8, 9, 5, 6, 11, 3]])

    def forward():
        return model(ids).logits

    shared = model.config.moe_num_shared_experts > 0
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(Ernie4_5_MoeQModel, model, layer, forward)
        groups = {(source, tuple(targets)): consumers for source, targets, consumers in checked}
        assert groups[("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))] == []
        # GQA narrows v_proj below o_proj's input, so only MHA can fold it.
        assert (("self_attn.v_proj", ("self_attn.o_proj",)) in groups) == (kv_heads == 4)
        if hasattr(layer.mlp, "experts"):
            experts = range(model.config.moe_num_experts)
            targets = [f"mlp.shared_experts.{p}" for p in ("gate_proj", "up_proj") if shared]
            targets += [f"mlp.experts.{i}.{p}" for i in experts for p in ("gate_proj", "up_proj")]
            # The fp32 router reads the same normed input and is compensated;
            # the score-correction bias is added after softmax and is scale-free.
            assert groups[("post_attention_layernorm", tuple(targets))] == ["mlp.gate"]
            for index in experts:
                assert (f"mlp.experts.{index}.up_proj", (f"mlp.experts.{index}.down_proj",)) in groups
            assert (("mlp.shared_experts.up_proj", ("mlp.shared_experts.down_proj",)) in groups) == shared
        else:
            assert groups[("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"))] == []
            assert ("mlp.up_proj", ("mlp.down_proj",)) in groups


def test_native_meta_model_defuses_without_materializing_large_weights():
    from transformers import Ernie4_5_MoeForCausalLM

    with torch.device("meta"):
        model = Ernie4_5_MoeForCausalLM(tiny_config())
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    assert all(value.device.type == "meta" for value in model.state_dict().values())
    assert model.model.layers[1].mlp.experts.get_submodule("2.up_proj").weight.shape == (12, 16)
