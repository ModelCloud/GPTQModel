# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AfmoeForCausalLM

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.afmoe import AfMoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.quantization.config import AWQConfig
from tests.test_afmoe_local_port import split_expert_checkpoint, tiny_config


SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/afmoe/checkpoint-schema.json").read_text()
)


@pytest.mark.parametrize("checkpoint", SCHEMA["checkpoints"], ids=lambda entry: entry["repo"])
def test_full_native_meta_inventory_matches_pinned_checkpoint_schema(checkpoint):
    from transformers import AfmoeConfig

    config = AfmoeConfig(**checkpoint["config"])
    with torch.device("meta"):
        model = AfmoeForCausalLM(config)
    actual = set(split_expert_checkpoint(model))
    expected = set(SCHEMA["global_keys"])
    for index in range(config.num_hidden_layers):
        suffixes = (
            SCHEMA["common_layer_keys"]
            + SCHEMA["dense_mlp_keys" if index < config.num_dense_layers else "moe_mlp_keys"]
        )
        for suffix in suffixes:
            for expert in range(config.num_experts) if "#" in suffix else (0,):
                expected.add(f"model.layers.{index}.{suffix.replace('#', str(expert))}")
    assert len(expected) == checkpoint["key_count"]
    assert actual == expected
    del model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dense_layers", [0, 1, 2])
def test_real_checkpoint_owners_and_moe_consumers(device, dense_layers):
    config = tiny_config(num_dense_layers=dense_layers, num_attention_heads=4, num_key_value_heads=2)
    with torch.device(device):
        model = AfmoeForCausalLM(config)
    source_keys = set(split_expert_checkpoint(model))
    _convert_model_with_defuser(AfMoeQModel, model, cleanup_original=True)
    assert set(model.state_dict()) == source_keys
    wrapper = AfMoeQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    simple = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    for index, layer in enumerate(model.model.layers):
        for key in layer.state_dict():
            # A direct parameter such as mlp.expert_bias belongs to mlp,
            # not to a fictitious child named expert_bias.
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (index, key, owner)
            if spec.quantize:
                assert owner in simple
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
            else:
                assert "!" in spec.flags or "?" in spec.flags
        for norm in ("self_attn.q_norm", "self_attn.k_norm"):
            assert not plan.find(norm).quantize
            assert norm not in simple
        for name, role in (("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v")):
            assert role in plan.find(name).roles
        if index >= dense_layers:
            assert plan.find("mlp").capture_only
            assert not plan.find("mlp.router.gate").quantize
            for expert in range(config.num_experts):
                name = f"mlp.experts.{expert}.gate_proj"
                assert plan.is_expert(name)
                assert looper._extract_moe_group_key(name) == f"mlp.experts.{expert}"
                assert name in simple
        assert plan.find("self_attn.q_norm.undeclared") is None
        assert plan.find("mlp.experts_extra.0.gate_proj") is None
    hooks = wrapper.moe_lifecycle_hooks
    assert hooks.module_plan is plan


@pytest.mark.parametrize("dense_layers", [0, 1, 2])
def test_awq_plain_policy_quantizes_every_linear_without_folding(monkeypatch, dense_layers):
    torch.manual_seed(52)
    config = tiny_config(num_dense_layers=dense_layers)
    model = AfmoeForCausalLM(config).eval()
    _convert_model_with_defuser(AfMoeQModel, model, cleanup_original=True)
    qcfg = AWQConfig(device="cpu", group_size=4, fallback=None)
    wrapper = AfMoeQModel(model=model, quantized=False, quantize_config=qcfg)
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=wrapper,
        model=model,
        calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    processor.apply_clip = False
    quantized_keys = {
        f"model.layers.{index}.{name}.weight"
        for index, layer in enumerate(model.model.layers)
        for name, module in layer.named_modules()
        if isinstance(module, torch.nn.Linear) and wrapper.module_plan().find(name).quantize
    }
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items() if name not in quantized_keys}
    for index, layer in enumerate(model.model.layers):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        state.modules = {
            name: NamedModule(module, name, f"model.layers.{index}.{name}", index)
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear) and wrapper.module_plan().find(name).quantize
        }
        assert len(state.modules) == (8 if index < dense_layers else 8 + 3 * config.num_experts)
        features = {name: torch.randn(5, named.module.in_features) for name, named in state.modules.items()}
        monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
        assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
        modules = dict(state.modules)
        before = {name: named.module.weight.detach().clone() for name, named in modules.items()}
        processor._quantize_layer(index, state)
        assert state.quantized
        for name, named in modules.items():
            assert "q_scales" in named.state and "q_zeros" in named.state and "wq" in named.state
            assert not torch.equal(named.module.weight, before[name])
        after = {name: named.module.weight.detach().clone() for name, named in modules.items()}
        processor._quantize_layer(index, state)
        for name, named in modules.items():
            torch.testing.assert_close(named.module.weight, after[name], rtol=0, atol=0)
    for name, tensor in frozen.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[2, 4, 6, 8]])).logits).all()
