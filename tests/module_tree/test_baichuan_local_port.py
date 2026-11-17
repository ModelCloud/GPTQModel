# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint ownership and conservative AWQ policy on real Baichuan layers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.baichuan import BaiChuanQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.baichuan.configuration_baichuan import BaichuanConfig
from gptqmodel.models.ports.baichuan.modeling_baichuan import BaichuanForCausalLM
from gptqmodel.quantization.config import AWQConfig


VARIANTS = [(p, n) for p in ("rope", "alibi") for n in (False, True)]
QUANTIZED = {"self_attn.W_pack", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
EXPECTED_BLOCKS = [["input_layernorm:!"], ["self_attn.W_pack", "self_attn.rotary_emb:!"], ["self_attn.o_proj"],
                   ["post_attention_layernorm:!"],
                   ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]


def build(position, normalized, device="cpu"):
    config = BaichuanConfig(vocab_size=37, hidden_size=24, intermediate_size=40,
                            num_hidden_layers=2, num_attention_heads=3,
                            position_embedding_type=position, normalize_lm_head=normalized)
    with torch.device(device):
        return BaichuanForCausalLM(config).eval()


@pytest.mark.parametrize("position,normalized", VARIANTS)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_real_state_dict_owners_match_definition_and_checkpoint_layout(position, normalized, device):
    model = build(position, normalized, device)
    # The public loaded wrapper compiles the plan used by looper consumers.
    wrapper = BaiChuanQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert isinstance(plan, ModulePlan)
    assert MODEL_MAP["baichuan"] is BaiChuanQModel
    assert wrapper.extract_layers_node() == ["model.layers"]
    assert wrapper.full_layer_modules() == EXPECTED_BLOCKS
    assert {path for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None))
            for path in block} == QUANTIZED
    expected_layer = {path + ".weight" for path in QUANTIZED}
    expected_layer |= {"input_layernorm.weight", "post_attention_layernorm.weight"}
    if position == "rope" and not normalized:
        expected_layer.add("self_attn.rotary_emb.inv_freq")
    expected_keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    expected_keys |= {f"model.layers.{i}.{key}" for i in range(2) for key in expected_layer}
    assert set(model.state_dict()) == expected_keys
    assert model.model.layers[0].self_attn.W_pack.weight.shape == (72, 24)
    for path, roles in {"self_attn.W_pack": ("q", "k", "v"), "self_attn.o_proj": ("o",),
                        "mlp.gate_proj": ("gate",), "mlp.up_proj": ("up",), "mlp.down_proj": ("down",)}.items():
        assert plan.roles(path) == roles
    for key in model.state_dict():
        if not key.startswith("model.layers."):
            continue
        relative = ".".join(key.split(".")[3:])
        owner = relative.rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, key
        if owner in QUANTIZED:
            assert spec.quantize
        else:
            assert not spec.quantize
            assert "!" in spec.flags
        assert plan.find(owner + ".undeclared") is None
    # No phantom rotary tensors are obligations for ALiBi or Baichuan 2.
    assert not any("cos_cached" in key or "sin_cached" in key for key in model.state_dict())
    assert BaiChuanQModel.fusable is None


@pytest.mark.parametrize("position,normalized", VARIANTS)
def test_awq_plain_quantization_retains_all_five_real_projections(position, normalized, monkeypatch):
    torch.manual_seed(44)
    model = build(position, normalized)
    wrapper = BaiChuanQModel(model=model, quantized=False, quantize_config=None)
    layer = model.model.layers[0]
    features, handles = {}, []
    for name in QUANTIZED:
        handles.append(layer.get_submodule(name).register_forward_pre_hook(
            lambda module, args, name=name: features.__setitem__(name, args[0].detach().clone())))
    try:
        with torch.no_grad():
            model(torch.tensor([[1, 3, 4, 5, 6], [1, 7, 8, 9, 10]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert set(features) == QUANTIZED
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=8, fallback=None),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=model,
                             calculate_w_wq_diff=False)
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    state = processor._get_layer_state(0)
    state.layer_module = layer
    named = {name: NamedModule(layer.get_submodule(name), name, "model.layers.0." + name, 0) for name in QUANTIZED}
    state.modules = dict(named)
    # Supply actual native-captured features without the full calibration
    # scheduler. The real no-group branch and apply_quant run unmocked.
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    with torch.no_grad():
        processor._quantize_layer(0, state)
    assert state.quantized
    for name, module in named.items():
        assert "q_scales" in module.state and "q_zeros" in module.state
        assert not torch.equal(module.module.weight, before[name + ".weight"])
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        torch.testing.assert_close(layer.state_dict()[name], before[name], rtol=0, atol=0)
    # Idempotent completion cannot double-quantize or fold a norm on retry.
    once = {name: value.clone() for name, value in layer.state_dict().items()}
    processor._quantize_layer(0, state)
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, once[name], rtol=0, atol=0)


def test_central_fixture_current_alias_and_proposed_ownership_contract():
    fixtures = Path(__file__).parent / "fixtures"
    registry = json.loads((fixtures / "registered_module_plans.json").read_text())
    assert registry["models"]["baichuan"] == "BaiChuanQModel"
    # Parent owns central fixtures. Pin the independent new expectation here
    # until its integration updates BaiChuanQModel/gptq with the same JSON.
    assert BaiChuanQModel.build_layer_modules(BaiChuanQModel.module_tree, include_capture_only=True) == EXPECTED_BLOCKS
