# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact checkpoint ownership and plain-AWQ consumer tests on real InternLM2."""

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.internlm2 import InternLM2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.internlm2 import InternLM2Config, InternLM2ForCausalLM
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AWQConfig


LINEARS = {"attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"}
NORMS = {"attention_norm", "ffn_norm"}


def build(heads, bias, device="cpu"):
    c = InternLM2Config(
        vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=heads, bias=bias,
    )
    with torch.device(device):
        model = InternLM2ForCausalLM(c).eval()
    _convert_model_with_defuser(InternLM2QModel, model, cleanup_original=False)
    return model, InternLM2QModel(model=model, quantized=False, quantize_config=QuantizeConfig())


@pytest.mark.parametrize("heads,bias,device", [(1, True, "meta"), (2, False, "cpu"), (4, True, "cpu")])
def test_checkpoint_weights_exactly_owned_by_live_plan(heads, bias, device):
    model, wrapper = build(heads, bias, device)
    assert MODEL_MAP["internlm2"] is InternLM2QModel
    assert wrapper.lm_head == "output"
    expected = {"model.tok_embeddings.weight", "model.norm.weight", "output.weight"}
    plan = wrapper.module_plan()
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        keys = {name + ".weight" for name in LINEARS | NORMS}
        if bias:
            keys |= {"attention.wqkv.bias", "attention.wo.bias"}
        assert set(layer.state_dict()) == keys
        expected |= {prefix + key for key in keys}
        for key in keys:
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None and layer.get_submodule(owner) is not None
            assert spec.quantize == (owner in LINEARS)
            if owner in NORMS:
                assert "!" in spec.flags
        assert layer.attention.wqkv.weight.shape == ((4 + 2 * heads) * 8, 32)
    assert set(model.state_dict()) == expected
    for method in ("gptq", "awq"):
        qcfg = QuantizeConfig() if method == "gptq" else AWQConfig()
        blocks = wrapper.simple_layer_modules(model.config, qcfg)
        assert blocks == [["attention.wqkv"], ["attention.wo"], ["feed_forward.w1", "feed_forward.w3"], ["feed_forward.w2"]]
        assert {x for b in blocks for x in b} == LINEARS
    for invalid in ("attention.wqkv_copy", "attention.wqkv.undeclared", "not_attention.wo"):
        assert plan.find(invalid) is None


@pytest.mark.parametrize("heads,bias", [(1, True), (4, False)])
def test_plain_awq_quantizes_every_native_linear_without_norm_fold(monkeypatch, heads, bias):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(57)
        model, wrapper = build(heads, bias)
        layer = model.model.layers[0]
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        before = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        processor = AWQProcessor(
            tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=8, fallback=None),
            calibration=CalibrationConfig(), gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
        )
        processor.draw_progress = lambda *args, **kwargs: None
        processor.log_new_row = lambda *args, **kwargs: None
        state = processor._get_layer_state(0)
        state.layer_module = layer
        state.modules = {
            name: NamedModule(layer.get_submodule(name), name, f"model.layers.0.{name}", 0) for name in sorted(LINEARS)
        }
        processor._feature_task_names = set()
        processor._awq_feature_kwargs = {}
        processor.apply_clip = False
        features = {name: torch.randn(8, named.module.in_features) for name, named in state.modules.items()}
        monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
        processor._quantize_layer(0, state)
        assert state.quantized
        for name, named in state.modules.items():
            assert "q_scales" in named.state and "q_zeros" in named.state and "wq" in named.state
            assert (named.module.weight != before[name + ".weight"]).any()
            assert torch.isfinite(named.module.weight).all()
        for name in NORMS:
            torch.testing.assert_close(layer.get_submodule(name).weight, before[name + ".weight"], atol=0, rtol=0)
        assert torch.isfinite(model(torch.tensor([[1, 5, 7]]), use_cache=False).logits).all()
    finally:
        torch.set_num_threads(previous)
