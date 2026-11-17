# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native dense/routed/shared ownership and conservative AWQ policy."""

from types import SimpleNamespace

import pytest
import torch

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


def test_awq_is_explicit_plain_quantization_even_with_complete_features(case):
    captured = {}
    handles = []
    for index, layer in enumerate(case.model.model.layers):
        captured[index] = {}
        for name, module in find_modules(layer).items():
            def capture(_module, inputs, *, name=name, index=index):
                captured[index][name] = inputs[0].detach().clone()
            handles.append(module.register_forward_pre_hook(capture))
    ids = torch.tensor([[1, 7, 8, 9]])
    try:
        with torch.inference_mode():
            reference = case.model(ids).logits
    finally:
        for handle in handles:
            handle.remove()
    before = {name: value.clone() for name, value in case.model.state_dict().items()}
    for index, layer in enumerate(case.model.model.layers):
        assert captured[index]
        assert case.awq_get_modules_for_scaling(layer, captured[index], {}) == []
    for name, value in case.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(case.model(ids).logits, reference, rtol=0, atol=0)


def test_native_meta_model_defuses_without_materializing_large_weights():
    from transformers import Ernie4_5_MoeForCausalLM

    with torch.device("meta"):
        model = Ernie4_5_MoeForCausalLM(tiny_config())
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    assert all(value.device.type == "meta" for value in model.state_dict().values())
    assert model.model.layers[1].mlp.experts.get_submodule("2.up_proj").weight.shape == (12, 16)
