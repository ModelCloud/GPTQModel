# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native EXAONE decoder ownership, aliases and conservative AWQ policy."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports import exaone4
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.model import get_layers_with_prefixes
from tests.test_exaone4_local_port import tiny_model


LINEARS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
NORMS = {"self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm", "post_feedforward_layernorm"}
CHECKPOINTS = json.loads(Path(exaone4.__file__).with_name("checkpoint_contract.json").read_text())["models"]


@pytest.mark.parametrize(
    "model_type,device,kv_heads", [("exaone4", "meta", 2), ("exaone4_5_text", "cpu", 4), ("exaone4_5", "meta", 4)]
)
def test_exact_native_state_dict_plan_ownership_and_roles(model_type, device, kv_heads):
    definition = MODEL_MAP[model_type]
    with torch.device(device):
        model = tiny_model(model_type, num_key_value_heads=kv_heads)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert len(layers) == 2
    prefix = "model.language_model" if model_type == "exaone4_5" else "model"
    assert names == [f"{prefix}.layers.{index}" for index in range(2)]
    blocks = definition.simple_layer_modules(model_config=model.config, quantize_config=SimpleNamespace(dynamic=None))
    assert {name for block in blocks for name in block} == LINEARS
    plan = ModulePlan.from_tree(definition.module_tree)
    for layer in layers:
        assert set(layer.state_dict()) == {f"{owner}.weight" for owner in LINEARS | NORMS}
        for owner in LINEARS | NORMS:
            spec = plan.find(owner)
            assert spec is not None
            assert spec.quantize == (owner in LINEARS)
            if owner in NORMS:
                assert "!" in spec.flags
            assert plan.find(owner + ".undeclared") is None
            assert plan.find("other." + owner) is None
        for role in ("q", "k", "v", "o"):
            assert plan.role_paths_in_parent("self_attn", role) == (f"self_attn.{role}_proj",)
    # Every language weight, including the head and embedding, has the native checkpoint name.
    language_keys = {f"{prefix}.layers.{index}.{owner}.weight" for index in range(2) for owner in LINEARS | NORMS}
    language_keys |= {f"{prefix}.embed_tokens.weight", f"{prefix}.norm.weight", "lm_head.weight"}
    if model_type == "exaone4_5":
        assert {key for key in model.state_dict() if not key.startswith("model.visual.")} == language_keys
        assert all(not key.startswith("model.visual") for key in names)
    else:
        assert set(model.state_dict()) == language_keys

    # This oracle was reduced from the published checkpoint header, not from the model or tree.
    observed_keys = {re.sub(r"(?<=\.)\d+(?=\.)", "#", key) for key in model.state_dict()}
    checkpoint = CHECKPOINTS["exaone4_5" if model_type == "exaone4_5" else "exaone4"]
    checkpoint_keys = set(checkpoint["normalized_weight_keys"])
    omitted = checkpoint_keys - observed_keys
    if model_type == "exaone4_5":
        assert omitted and all(key.startswith("mtp.") for key in omitted)
        assert definition.out_of_model_tensors == {"prefixes": ["mtp"]}
    else:
        assert not omitted
    assert not observed_keys - checkpoint_keys


@pytest.mark.parametrize("model_type,fallback", [("exaone4", None), ("exaone4_5", {})])
def test_plain_awq_quantizes_every_native_linear_without_folding(tmp_path, monkeypatch, model_type, fallback):
    model = tiny_model(model_type)
    definition = MODEL_MAP[model_type]
    layer = model.get_submodule("model.language_model.layers.0" if model_type == "exaone4_5" else "model.layers.0")
    wrapper = SimpleNamespace(
        model=model, qlinear_kernel=None, rotary_embedding=None, lm_head="lm_head",
        awq_get_modules_for_scaling=lambda *args: definition.awq_get_modules_for_scaling(None, *args),
    )
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback)
    processor = AWQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(), gptq_model=wrapper,
                             model=model, calculate_w_wq_diff=False)
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    state = processor._get_layer_state(0)
    state.layer_module = layer
    state.modules = {name: NamedModule(layer.get_submodule(name), name, f"model.layers.0.{name}", 0)
                     for name in sorted(LINEARS)}
    features = {name: torch.randn(5, item.module.in_features) for name, item in state.modules.items()}
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
    before = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    processor._quantize_layer(0, state)
    assert state.quantized
    for name, item in state.modules.items():
        assert {"q_scales", "q_zeros", "wq"} <= item.state.keys()
        assert torch.isfinite(item.state["wq"]).all()
        assert not torch.equal(before[name + ".weight"], item.module.weight)
    for name in NORMS:
        torch.testing.assert_close(layer.get_submodule(name).weight, before[name + ".weight"], atol=0, rtol=0)
