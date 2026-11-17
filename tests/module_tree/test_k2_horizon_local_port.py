# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual model/checkpoint ownership, consumers and plain AWQ for K2-Horizon."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.k2_horizon import K2HorizonGPTQ, K2HorizonQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.k2_horizon import K2HorizonConfig, K2HorizonForCausalLM
from gptqmodel.quantization.config import AWQConfig
from tests.test_k2_horizon_local_port import variant_config


LAYOUTS = json.loads((Path(__file__).parents[2] / "gptqmodel/models/ports/k2_horizon/checkpoint_layouts.json").read_text())


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda row: row["repo"])
def test_full_meta_keyset_matches_pinned_upstream_checkpoint(layout):
    config = K2HorizonConfig(**layout["config"])
    with torch.device("meta"):
        model = K2HorizonForCausalLM(config)
    keys = sorted(model.state_dict())
    assert len(keys) == layout["checkpoint_key_count"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == layout["checkpoint_keys_sha256"]
    assert all(t.device.type == "meta" for t in model.state_dict().values())
    del model


def _wrapper(model):
    return K2HorizonQModel(model=model, quantized=False, quantize_config=None)


@pytest.mark.parametrize("variant", ["dense", "mova", "moe", "qknorm"])
def test_native_weight_owners_tags_aliases_and_looper_consumers(variant):
    assert MODEL_MAP["k2_horizon"] is K2HorizonQModel is K2HorizonGPTQ
    config = variant_config(variant)
    with torch.device("meta"):
        model = K2HorizonForCausalLM(config)
    _convert_model_with_defuser(K2HorizonQModel, model, cleanup_original=False)
    wrapper = _wrapper(model)
    plan = wrapper.module_plan()
    assert K2HorizonQModel.extract_layers_node() == ["model.layers"]
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    weight_only = WeightOnlyLooper.__new__(WeightOnlyLooper)
    weight_only.gptq_model = wrapper
    actual_quantized = set()
    for index, layer in enumerate(model.model.layers):
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (variant, owner)
            assert plan.find(owner + ".undeclared") is None
            assert plan.find(owner + "_similar") is None
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                actual_quantized.add(owner)
            else:
                assert "!" in spec.flags or "?" in spec.flags
        for owner in declared:
            try:
                native = layer.get_submodule(owner)
            except AttributeError:
                continue  # The union tree deliberately includes dense and sparse siblings.
            assert isinstance(native, torch.nn.Linear)
            assert plan.find(owner).quantize
        if index and variant in {"moe", "mova", "qknorm"}:
            assert plan.is_expert("mlp.experts.2.down_proj")
            assert looper._extract_moe_group_key("mlp.experts.2.down_proj") == "mlp.experts.2"
            assert weight_only._extract_moe_group_key("mlp.experts.2.down_proj") == "mlp.experts.2"
    assert {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} <= actual_quantized
    if variant == "mova":
        assert "self_attn.v_experts.3" in actual_quantized
        assert "self_attn.v_experts.4" not in declared
        assert "mlp.experts.3.gate_proj" not in declared
        hooks = wrapper.moe_lifecycle_hooks
        layer = model.model.layers[1]
        assert hooks.get_moe_block_for_subset(layer, K2HorizonQModel, {"self_attn.v_experts.3": object()}) is layer.self_attn
        assert hooks.get_moe_block_for_subset(layer, K2HorizonQModel, {"mlp.experts.2.down_proj": object()}) is layer.mlp
    # Fusing this local model is not qualified by structural plan coverage.
    assert K2HorizonQModel.fusable is None


@pytest.mark.parametrize("variant", ["dense", "mova", "moe", "qknorm"])
def test_plain_awq_quantizes_every_supported_linear_without_mutating_norms_or_routers(monkeypatch, variant):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(281)
            model = K2HorizonForCausalLM(variant_config(variant)).eval()
        wrapper = _wrapper(model)
        processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=4, fallback=None),
                                 calibration=CalibrationConfig(), gptq_model=wrapper,
                                 model=model, calculate_w_wq_diff=False)
        processor.draw_progress = lambda *args, **kwargs: None
        processor.log_new_row = lambda *args, **kwargs: None
        processor.apply_clip = False
        plan = wrapper.module_plan()
        changed = set()
        for index, layer in enumerate(model.model.layers):
            original = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
            selected = {name: module for name, module in layer.named_modules()
                        if isinstance(module, torch.nn.Linear) and plan.find(name).quantize}
            assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
            state = processor._get_layer_state(index)
            state.layer_module = layer
            state.modules = {name: NamedModule(module, name, f"model.layers.{index}.{name}", index)
                             for name, module in selected.items()}
            staged = dict(state.modules)
            features = {name: torch.ones(2, module.in_features) for name, module in selected.items()}
            monkeypatch.setattr(processor, "_layer_input_features", lambda state, f=features: f)
            processor._quantize_layer(index, state)
            assert state.quantized
            for name, module in selected.items():
                entry = staged[name]
                assert "q_scales" in entry.state and "q_zeros" in entry.state
                assert not torch.equal(module.weight, original[name + ".weight"])
                changed.add(name)
            for key, tensor in layer.state_dict().items():
                owner = key.rsplit(".", 1)[0]
                if not plan.find(owner).quantize:
                    torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        assert "self_attn.v_proj" in changed
        if variant == "mova":
            assert "self_attn.v_experts.3" in changed
            assert "mlp.experts.2.down_proj" in changed
        with torch.no_grad():
            assert torch.isfinite(model(torch.tensor([[1, 4, 8]]), use_cache=False).logits).all()
    finally:
        torch.set_num_threads(previous_threads)
