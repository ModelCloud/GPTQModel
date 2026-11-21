# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint audit, live hybrid plans and AWQ scale-fold qualification."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.intern_s2_preview import InternS2PreviewQModel as Definition
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.intern_s2_preview.configuration_interns2_preview import InternS2PreviewConfig
from gptqmodel.models.ports.intern_s2_preview.modeling_interns2_preview import InternS2PreviewForConditionalGeneration
from gptqmodel.quantization.config import AWQConfig
from tests.test_intern_s2_preview_local_port import tiny_config


FIXTURES = Path(__file__).parents[1] / "fixtures/ports/intern_s2_preview"
RELEASED_NON_MTP_KEYS_SHA256 = "9a3ddb41f442b04e68c0aaa764975bf8e4dab758c1a6890a749f816c5a50e1f3"
LAYOUTS = [
    ["linear_attention", "full_attention"], ["full_attention", "linear_attention"],
    ["linear_attention"] * 2, ["full_attention"] * 2,
]


def wrapper_for(model):
    wrapper = Definition.__new__(Definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["_module_plan"] = ModulePlan.from_tree(Definition.module_tree)
    return wrapper


def get_submodule(layer, name):
    try:
        return layer.get_submodule(name)
    except AttributeError:
        return None


def test_exact_released_checkpoint_meta_ownership_and_aliases():
    # internlm/Intern-S2-Preview@4f57cab5: 2098 safetensors tensors, 785 under mtp.*
    config = InternS2PreviewConfig(**json.loads((FIXTURES / "upstream_config.json").read_text()))
    with torch.device("meta"):
        model = InternS2PreviewForConditionalGeneration(config)
    keys = set(model.state_dict())
    assert len(keys) == 2098 - 785
    assert hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest() == RELEASED_NON_MTP_KEYS_SHA256
    assert len(model.model.language_model.layers) == 40
    assert len(model.model.visual.blocks) == 27
    assert Definition.out_of_model_tensors == {"prefixes": ["mtp"]}
    assert {alias for alias, cls in MODEL_MAP.items() if cls is Definition} == {"intern_s2_preview"}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("bias", [False, True])
@torch.no_grad()
def test_live_decoder_owners_defuser_and_plan_consumers(layout, bias):
    torch.manual_seed(41)
    model = InternS2PreviewForConditionalGeneration(tiny_config(layout, attention_bias=bias)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids, use_cache=False).logits
    _convert_model_with_defuser(Definition, model, cleanup_original=True)
    torch.testing.assert_close(model(ids, use_cache=False).logits, expected)
    wrapper = wrapper_for(model)
    plan = wrapper.module_plan()
    assert isinstance(plan, ModulePlan)
    assert wrapper.extract_layers_node() == ["model.language_model.layers"]
    declared = {name for block in wrapper.simple_layer_modules(model.config, None) for name in block}
    experts = {f"mlp.{branch}.{projection}" for branch in ["shared_expert", "experts.0", "experts.1", "experts.2"]
               for projection in ["gate_proj", "up_proj", "down_proj"]}
    common = {name + ".weight" for name in experts} | {
        "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight", "mlp.shared_expert_gate.weight"}
    for kind, layer in zip(layout, model.model.language_model.layers):
        if kind == "linear_attention":
            linears = {"linear_attn." + name for name in ("in_proj_qkv", "in_proj_z", "out_proj")}
            extra = {"linear_attn." + name for name in ("norm.weight", "conv1d.weight", "dt_bias", "A_log",
                                                       "in_proj_a.weight", "in_proj_b.weight")}
        else:
            linears = {"self_attn." + name + "_proj" for name in ("q", "k", "v", "o")}
            extra = {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}
            if bias:
                extra |= {name + ".bias" for name in linears}
        assert set(layer.state_dict()) == common | {name + ".weight" for name in linears} | extra
        live = {name for name in declared if isinstance(get_submodule(layer, name), nn.Linear)}
        assert live == experts | linears
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
            else:
                assert set(spec.flags) & {"!", "?"}
    assert plan.find("linear_attn").flags == ("!",)
    for path, role in [("self_attn.q_proj", "q"), ("self_attn.o_proj", "o"),
                       ("mlp.experts.1.gate_proj", "gate"), ("mlp.shared_expert.down_proj", "down")]:
        assert role in plan.find(path).roles
    assert plan.find("mlp").is_moe
    for path in ("linear_attn.undeclared", "self_attn.q_proj_extra", "mlp.experts.1.undeclared"):
        assert plan.find(path) is None
    assert wrapper.moe_lifecycle_hooks is not None


def test_independent_registry_block_contract():
    # The integrator owns the shared JSON registry. Assert the exact addition
    # here using the existing independent fixture with the new state owner.
    registry = Path(__file__).parent / "fixtures" / "registered_module_plans.json"
    expected = json.loads(registry.read_text())["cases"]["InternS2PreviewQModel/gptq"]["blocks"]
    expected = [block for block in expected if block != ["linear_attn:!"]]
    expected.insert(3, ["linear_attn:!"])
    assert Definition.build_layer_modules(Definition.module_tree, include_capture_only=True) == expected


@pytest.mark.parametrize("layout", LAYOUTS)
@torch.no_grad()
def test_awq_without_captured_features_quantizes_all_declared_linears_plainly(layout, monkeypatch):
    torch.manual_seed(43)
    model = InternS2PreviewForConditionalGeneration(tiny_config(layout)).eval()
    _convert_model_with_defuser(Definition, model, cleanup_original=True)
    wrapper = wrapper_for(model)
    declared = {name for block in wrapper.simple_layer_modules(model.config, None) for name in block}
    processor = AWQProcessor(
        tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=4, fallback=None),
        calibration=CalibrationConfig(), gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    for index, (kind, layer) in enumerate(zip(layout, model.model.language_model.layers)):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        state.modules = {
            f"model.language_model.layers.{index}.{name}": NamedModule(
                layer.get_submodule(name), name, f"model.language_model.layers.{index}.{name}", index,
            ) for name in sorted(declared) if isinstance(get_submodule(layer, name), nn.Linear)
        }
        original = {name: value.clone() for name, value in layer.state_dict().items()}
        named_linears = list(state.modules.values())
        assert len(named_linears) == (15 if kind == "linear_attention" else 16)
        monkeypatch.setattr(processor, "_layer_input_features", lambda state: {})
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        processor._quantize_layer(index, state)
        assert state.quantized
        for named in named_linears:
            assert "q_scales" in named.state and "q_zeros" in named.state
            assert not torch.equal(named.module.weight, original[named.name + ".weight"])
        changed = {name for name, value in layer.state_dict().items() if not torch.equal(value, original[name])}
        assert changed == {named.name + ".weight" for named in named_linears}
    assert torch.isfinite(model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits).all()


EXPERT_GATE_UP = [f"mlp.{branch}.{projection}" for branch in ("shared_expert", "experts.0", "experts.1", "experts.2")
                  for projection in ("gate_proj", "up_proj")]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_awq_folds_preserve_hybrid_model_output(layout):
    torch.manual_seed(47)
    model = InternS2PreviewForConditionalGeneration(tiny_config(layout)).eval()
    _convert_model_with_defuser(Definition, model, cleanup_original=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def forward():
        return model(ids, use_cache=False).logits

    for kind, layer in zip(layout, model.model.language_model.layers):
        checked = assert_awq_folds_neutral(Definition, model, layer, forward, tree=Definition.module_tree)
        groups = {source: (targets, consumers) for source, targets, consumers in checked}
        if kind == "linear_attention":
            # One native input edge for all four DeltaNet projections; the
            # unquantized b/a gates are compensated, nothing folds inside DeltaNet.
            assert groups["input_layernorm"] == (
                ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
                ["linear_attn.in_proj_b", "linear_attn.in_proj_a"],
            )
            assert not any(source.startswith("linear_attn.") for source in groups)
        else:
            assert groups["input_layernorm"] == (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], [])
        # Router and shared-expert gate read the same norm output as every expert.
        assert groups["post_attention_layernorm"] == (EXPERT_GATE_UP, ["mlp.gate", "mlp.shared_expert_gate"])
        for branch in ("shared_expert", "experts.0", "experts.1", "experts.2"):
            assert groups[f"mlp.{branch}.up_proj"] == ([f"mlp.{branch}.down_proj"], [])
        assert len(groups) == len(checked) == 6
