# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Both aliases use real native layers, Defuser and the quantizer consumers."""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import _capture, _quantized_features, assert_awq_folds_neutral
from torch import nn

from gptqmodel.looper import awq_processor
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
    assert resolved <= declared
    assert plan.find("mlp.text_moe.gate").quantize is False
    assert plan.find("mlp.vision_moe.gate.moe_statics").quantize is False
    assert plan.find("mlp.text_moe.experts_similar.0.gate_proj") is None


EXPERT_GATE_UP = [f"mlp.{branch}.{projection}"
                  for branch in ("shared_experts", "text_moe.experts.0", "text_moe.experts.1",
                                 "vision_moe.experts.0", "vision_moe.experts.1")
                  for projection in ("gate_proj", "up_proj")]
ROUTERS = ["mlp.text_moe.gate", "mlp.vision_moe.gate"]


def _mixed_modality_model(definition, *, kv_heads=1, seed=37):
    config = tiny_config()
    config.text_config.num_key_value_heads = kv_heads
    torch.manual_seed(seed)
    model = definition.loader.from_config(config).eval()
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    text = model.model.language_model
    ids = torch.randint(1, 40, (2, 8))
    # 0 = text token, 1 = image token: both modality expert families run.
    token_types = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1]] * 2)
    return model, text, ids, token_types


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("capture", ["mixed", "text_only"])
def test_awq_folds_preserve_dense_and_modality_moe_layer_output(alias, kv_heads, capture):
    definition = MODEL_MAP[alias]
    model, text, ids, token_types = _mixed_modality_model(definition, kv_heads=kv_heads)

    def forward():
        return text(input_ids=ids, moe_mm_token_type_ids=token_types).last_hidden_state

    def text_only():
        return text(input_ids=ids).last_hidden_state

    for kind, layer in zip(("dense", "sparse"), text.layers):
        provider = None
        if capture == "text_only":
            # Calibration without images never reaches the vision experts, yet
            # the post-attention norm still feeds them (and their router) at
            # inference: the fold must rescale them anyway.
            _, features = _capture(layer, text_only)
            features = _quantized_features(layer, features, definition.module_tree)

            def provider(wrapper, layer, _features, kwargs, features=features):
                return definition.awq_get_modules_for_scaling(wrapper, layer, features, kwargs)

        checked = assert_awq_folds_neutral(definition, model, layer, forward,
                                           tree=definition.module_tree, scaling_provider=provider)
        groups = {source: (targets, consumers) for source, targets, consumers in checked}
        assert groups["input_layernorm"] == (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], [])
        # v_proj -> o_proj only folds when the widths match (MHA, not GQA).
        assert ("self_attn.v_proj" in groups) == (kv_heads == 2)
        if kind == "dense":
            assert groups["post_attention_layernorm"] == (["mlp.gate_proj", "mlp.up_proj"], [])
            assert groups["mlp.up_proj"] == (["mlp.down_proj"], [])
        else:
            assert groups["post_attention_layernorm"] == (EXPERT_GATE_UP, ROUTERS)
            assert groups["mlp.shared_experts.up_proj"] == (["mlp.shared_experts.down_proj"], [])
            experts = [source for source in groups if source.startswith(("mlp.text_moe.", "mlp.vision_moe."))]
            assert experts and all(groups[source] == ([source.replace("up_proj", "down_proj")], [])
                                   for source in experts)


@pytest.mark.parametrize("alias", ALIASES)
def test_awq_scale_search_quantizes_all_real_dense_and_modality_expert_linears(alias, monkeypatch):
    definition = MODEL_MAP[alias]
    model, text, ids, token_types = _mixed_modality_model(definition)
    captured, layer_kwargs, handles = {}, {}, []
    for index, layer in enumerate(text.layers):
        def layer_hook(module, args, kwargs, index=index):
            layer_kwargs[index] = {key: value for key, value in kwargs.items() if key != "hidden_states"}

        handles.append(layer.register_forward_pre_hook(layer_hook, with_kwargs=True))
        for name, child in layer.named_modules():
            def hook(module, args, index=index, name=name):
                if args and isinstance(args[0], torch.Tensor):
                    captured.setdefault(index, {})[name] = args[0].detach().clone()

            if name:
                handles.append(child.register_forward_pre_hook(hook))
    with torch.no_grad():
        text(input_ids=ids, moe_mm_token_type_ids=token_types)
    for handle in handles:
        handle.remove()

    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["qlinear_kernel"] = None
    wrapper.__dict__["rotary_embedding"] = None
    # fallback=None: any failing scale group raises instead of quantizing plainly.
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=-1, fallback=None),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=model,
                             calculate_w_wq_diff=False)
    processor.draw_progress = lambda *a, **k: None
    processor.log_new_row = lambda *a, **k: None
    processor.apply_clip = False
    applied = []
    real_apply_scale = awq_processor.apply_scale

    def spy_apply_scale(module, scales_list, *args, **kwargs):
        consumer_map = kwargs.get("consumer_map") or {}
        applied.append({prev: (list(layers), list(consumer_map.get(prev, ())))
                        for prev, layers, _scales, _loss in scales_list})
        return real_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", spy_apply_scale)
    declared = {name for block in definition.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block}
    counts = []
    for index, layer in enumerate(text.layers):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        linears = {name: module for name, module in layer.named_modules() if name in declared}
        # The looper keys a layer's modules by their layer-relative names.
        prefix = f"model.language_model.layers.{index}"
        state.modules = {name: NamedModule(module, name, f"{prefix}.{name}", index)
                         for name, module in linears.items()}
        monkeypatch.setattr(processor, "_layer_input_features", lambda state, index=index: captured[index])
        processor._module_forward_kwargs = dict(layer_kwargs[index])
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
    dense, sparse = applied
    assert dense["post_attention_layernorm"] == (["mlp.gate_proj", "mlp.up_proj"], [])
    assert sparse["input_layernorm"][0] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    assert sparse["post_attention_layernorm"] == (EXPERT_GATE_UP, ROUTERS)
    assert sparse["mlp.shared_experts.up_proj"] == (["mlp.shared_experts.down_proj"], [])
