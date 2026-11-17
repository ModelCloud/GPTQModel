# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Both aliases use real native layers, Defuser and the quantizer consumers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.config import AWQConfig
from tests.test_ernie4_5_vl_moe_local_port import ALIASES, tiny_config


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("layout", [["dense", "sparse"], ["dense", "dense"], ["sparse", "sparse"]])
@pytest.mark.parametrize("bias", [False, True])
def test_real_weight_ownership_and_plan_tags(alias, layout, bias):
    definition = MODEL_MAP[alias]
    config = tiny_config()
    config.text_config.mlp_layer_types = layout
    config.text_config.use_bias = bias
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    plan = ModulePlan.from_tree(definition.module_tree)
    declared = {path.split(":")[0] for block in definition.simple_layer_modules(config, SimpleNamespace(dynamic=None))
                for path in block}
    resolved = set()
    for kind, layer in zip(layout, model.model.language_model.layers):
        expected = {f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")}
        if kind == "dense":
            expected |= {f"mlp.{name}_proj" for name in ("gate", "up", "down")}
        else:
            expected |= {f"mlp.shared_experts.{name}_proj" for name in ("gate", "up", "down")}
            expected |= {f"mlp.{family}.experts.{index}.{name}_proj" for family in ("text_moe", "vision_moe")
                         for index in range(config.text_config.moe_num_experts) for name in ("gate", "up", "down")}
        actual = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), nn.Linear)
                actual.add(owner)
            else:
                assert set(spec.flags) & {"!", "?"}
        assert actual == expected
        resolved |= actual
        for name in expected:
            assert plan.find(name + ".undeclared") is None
        assert definition.__new__(definition).awq_get_modules_for_scaling(layer, {}, {}) == []
    assert resolved <= declared
    assert plan.find("mlp.text_moe.gate").quantize is False
    assert plan.find("mlp.vision_moe.gate.moe_statics").quantize is False
    assert plan.find("mlp.text_moe.experts_similar.0.gate_proj") is None


@pytest.mark.parametrize("alias", ALIASES)
def test_plain_awq_quantizes_all_real_dense_and_modality_expert_linears(alias, monkeypatch):
    definition = MODEL_MAP[alias]
    torch.manual_seed(37)
    model = definition.loader.from_config(tiny_config()).eval()
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["qlinear_kernel"] = None
    wrapper.__dict__["rotary_embedding"] = None
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=-1),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=model,
                             calculate_w_wq_diff=False)
    processor.draw_progress = lambda *a, **k: None
    processor.log_new_row = lambda *a, **k: None
    processor.apply_clip = False
    declared = {name for block in definition.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block}
    counts = []
    for index, layer in enumerate(model.model.language_model.layers):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        linears = {name: module for name, module in layer.named_modules() if name in declared}
        state.modules = {f"{index}.{name}": NamedModule(module, name, f"{index}.{name}", index)
                         for name, module in linears.items()}
        features = {name: torch.randn(4, named.module.in_features) for name, named in state.modules.items()}
        monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
        processor._feature_task_names = set()
        processor._awq_feature_kwargs = {}
        original = {name: named.weight.detach().clone() for name, named in state.modules.items()}
        modules = dict(state.modules)
        processor._quantize_layer(index, state)
        assert state.quantized
        for name, named in modules.items():
            assert named.state["wq"].shape == original[name].shape
            assert named.state["q_scales"] is not None
            assert torch.isfinite(named.state["wq"]).all()
            assert not torch.equal(named.state["wq"], original[name])
        counts.append(len(modules))
    assert counts == [7, 19]
