# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Checkpoint ownership and conservative AWQ on both real CodeGen layouts."""

import json
from pathlib import Path

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.codegen import CodeGenQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.codegen.configuration_codegen import CodeGenConfig
from gptqmodel.models.ports.codegen.modeling_codegen import CodeGenForCausalLM
from gptqmodel.quantization.config import AWQConfig


CONFIGS = json.loads((Path(__file__).parents[1] / "fixtures/ports/codegen/upstream_configs.json").read_text())
LINEARS = {"attn.qkv_proj", "attn.out_proj", "mlp.fc_in", "mlp.fc_out"}
LAYER_KEYS = {"ln_1.weight", "ln_1.bias", "attn.qkv_proj.weight", "attn.out_proj.weight",
              "mlp.fc_in.weight", "mlp.fc_in.bias", "mlp.fc_out.weight", "mlp.fc_out.bias"}


@pytest.mark.parametrize("name", CONFIGS)
def test_full_official_geometry_meta_ownership(name):
    # Native geometry, not a tree-generated fixture; no trained tensors needed.
    config = CodeGenConfig(**CONFIGS[name]["config"], vocab_size=51200)
    with torch.device("meta"):
        model = CodeGenQModel.loader.from_config(config)
    _convert_model_with_defuser(CodeGenQModel, model, cleanup_original=False)
    wrapper = CodeGenQModel(model=model, quantized=False, quantize_config=None)
    assert MODEL_MAP["codegen"] is CodeGenQModel
    assert wrapper.extract_layers_node() == ["transformer.h"]
    declared = {path for subset in wrapper.simple_layer_modules(config, None) for path in subset}
    assert declared == LINEARS
    expected_keys = {"transformer.wte.weight", "transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight", "lm_head.bias"}
    plan = wrapper.module_plan()
    for index, layer in enumerate(model.transformer.h):
        assert set(layer.state_dict()) == LAYER_KEYS
        expected_keys.update(f"transformer.h.{index}.{key}" for key in LAYER_KEYS)
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (name, owner)
            if spec.quantize:
                assert owner in declared
            else:
                assert "!" in spec.flags
        for boundary in ("ln_1", "attn.qkv_proj", "attn.out_proj", "mlp.fc_in"):
            assert "nosrc" in plan.find(boundary).flags
    assert set(model.state_dict()) == expected_keys
    assert plan.find("attn.qkv_proj_fake") is None


@pytest.mark.parametrize("partitions", [4, 8])
def test_awq_plain_quantization_reaches_every_supported_linear(monkeypatch, partitions):
    torch.manual_seed(81)
    config = CodeGenConfig(n_embd=64, n_head=8, n_layer=1, n_inner=96, rotary_dim=4,
                           n_positions=32, n_ctx=32, vocab_size=41, qkv_layout=f"qvk_{partitions}",
                           bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = CodeGenForCausalLM(config).eval()
    wrapper = CodeGenQModel(model=model, quantized=False, quantize_config=None)
    layer = model.transformer.h[0]
    features = {}
    handles = []
    def capture(name):
        def hook(module, args):
            features[name] = args[0].detach().clone()
        return hook
    try:
        for name in LINEARS:
            handles.append(layer.get_submodule(name).register_forward_pre_hook(capture(name)))
        with torch.inference_mode():
            before_logits = model(torch.tensor([[1, 7, 9, 3, 6]]), use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    assert set(features) == LINEARS
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    processor = AWQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(),
                             gptq_model=wrapper, model=model, calculate_w_wq_diff=False)
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    state = processor._get_layer_state(0)
    state.layer_module = layer
    state.modules = {name: NamedModule(layer.get_submodule(name), name, f"transformer.h.0.{name}", 0) for name in LINEARS}
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    original = {key: tensor.clone() for key, tensor in layer.state_dict().items()}
    # Execute actual pseudo-quantization and state publication, not a call spy.
    processor._quantize_layer(0, state)
    assert state.quantized
    for name, named in state.modules.items():
        assert named.state["q_scales"].numel() and named.state["q_zeros"].numel()
        assert not torch.equal(named.module.weight, original[f"{name}.weight"])
    torch.testing.assert_close(layer.ln_1.weight, original["ln_1.weight"], rtol=0, atol=0)
    torch.testing.assert_close(layer.ln_1.bias, original["ln_1.bias"], rtol=0, atol=0)
    with torch.inference_mode():
        after_logits = model(torch.tensor([[1, 7, 9, 3, 6]]), use_cache=False).logits
    assert after_logits.isfinite().all()
    assert not torch.equal(after_logits, before_logits)
