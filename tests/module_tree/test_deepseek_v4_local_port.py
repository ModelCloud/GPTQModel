# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

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
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import AWQConfig
from tests.test_deepseek_v4_local_port import ATTENTION_TYPES, MLP_TYPES, native_model, source_checkpoint, tiny_config


COMMON_KEYS = {
    "attn_hc.fn", "attn_hc.base", "attn_hc.scale",
    "ffn_hc.fn", "ffn_hc.base", "ffn_hc.scale",
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.sinks", "self_attn.q_a_norm.weight", "self_attn.kv_norm.weight",
    "mlp.gate.weight",
    *(f"self_attn.{name}.weight" for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj")),
    *(f"mlp.shared_experts.{name}_proj.weight" for name in ("gate", "up", "down")),
}
COMPRESSOR_KEYS = {"position_bias", "kv_proj.weight", "gate_proj.weight", "kv_norm.weight"}
INDEXER_KEYS = {*COMPRESSOR_KEYS, "q_b_proj.weight", "scorer.weights_proj.weight"}


def test_full_native_meta_inventory_matches_pinned_flash_checkpoint():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    port = Path(__file__).resolve().parents[2] / "gptqmodel/models/ports/deepseek_v4"
    schema = json.loads((port / "checkpoint-schema.json").read_text())
    assert hashlib.sha256((port / "upstream_config.json").read_bytes()).hexdigest() == schema["config_sha256"]
    config = DeepseekV4Config(**json.loads((port / "upstream_config.json").read_text()))
    expected = set(schema["global_keys"])
    indexed_layer_key_count = 0
    for index, name in schema["layers"].items():
        for suffix in schema["layer_schemas"][name]:
            for expert in range(config.n_routed_experts) if "#" in suffix else (0,):
                key = f"layers.{index}.{suffix.replace('#', str(expert))}"
                indexed_layer_key_count += 1
                if not key.endswith(".scale"):
                    expected.add(key)
    assert len(expected) == schema["native_target_key_count"] == 34223
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(config)
    assert set(source_checkpoint(model)) == expected
    assert schema["index_key_count"] == 69187
    assert schema["mtp_key_count"] == 1575
    assert indexed_layer_key_count + len(schema["global_keys"]) + schema["mtp_key_count"] == schema["index_key_count"]
    del model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("attention", ATTENTION_TYPES)
@pytest.mark.parametrize("mlp", MLP_TYPES)
def test_exact_native_and_defused_checkpoint_owners(device, attention, mlp):
    with torch.device(device):
        model = DeepSeekV4QModel.loader.from_config(tiny_config(attention, mlp))
    expected = COMMON_KEYS | {"mlp.gate." + ("tid2eid" if mlp == "hash_moe" else "e_score_correction_bias")}
    if attention != "sliding_attention":
        expected |= {"self_attn.compressor." + name for name in COMPRESSOR_KEYS}
    if attention == "compressed_sparse_attention":
        expected |= {"self_attn.compressor.indexer." + name for name in INDEXER_KEYS}
    fused = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    assert set(model.model.layers[0].state_dict()) == expected | fused
    assert set(model.state_dict()) - {"model.layers.0." + name for name in expected | fused} == {
        "model.embed_tokens.weight", "model.norm.weight", "model.hc_head.hc_fn",
        "model.hc_head.hc_base", "model.hc_head.hc_scale", "lm_head.weight",
    }
    assert _convert_model_with_defuser(DeepSeekV4QModel, model, cleanup_original=True)
    experts = {f"mlp.experts.{expert}.{proj}_proj.weight" for expert in range(3) for proj in ("gate", "up", "down")}
    assert set(model.model.layers[0].state_dict()) == expected | experts
    wrapper = DeepSeekV4QModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    simple = {name for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block}
    quantized = {
        "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj", "self_attn.o_b_proj",
        *(f"mlp.shared_experts.{proj}_proj" for proj in ("gate", "up", "down")),
        *(key.removesuffix(".weight") for key in experts),
    }
    assert simple == quantized
    layer = model.model.layers[0]
    for key in layer.state_dict():
        owner = key.rsplit(".", 1)[0]
        spec = plan.find(owner)
        assert spec is not None, key
        assert spec.quantize == (owner in quantized), key
        if not spec.quantize:
            assert "!" in spec.flags or "?" in spec.flags
    for name, role in (("q_a_proj", "q"), ("q_b_proj", "q"), ("kv_proj", "k"), ("kv_proj", "v"), ("o_b_proj", "o")):
        assert role in plan.find("self_attn." + name).roles
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    for expert in range(3):
        for proj in ("gate", "up", "down"):
            name = f"mlp.experts.{expert}.{proj}_proj"
            assert plan.is_expert(name)
            assert looper._extract_moe_group_key(name) == f"mlp.experts.{expert}"
    assert wrapper.moe_lifecycle_hooks.module_plan is plan
    assert plan.find("mlp.experts_extra.0.gate_proj") is None
    assert plan.find("self_attn.q_a_proj.undeclared") is None


@pytest.mark.parametrize("attention", ATTENTION_TYPES)
@pytest.mark.parametrize("mlp", MLP_TYPES)
def test_awq_plain_policy_quantizes_and_packs_all_supported_linears(monkeypatch, attention, mlp):
    model = native_model(attention, mlp)
    _convert_model_with_defuser(DeepSeekV4QModel, model, cleanup_original=True)
    qcfg = AWQConfig(device="cpu", group_size=-1, fallback=None)
    wrapper = DeepSeekV4QModel(model=model, quantized=False, quantize_config=qcfg)
    processor = AWQProcessor(
        tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(),
        gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    processor.apply_clip = False
    layer = model.model.layers[0]
    state = processor._get_layer_state(0)
    state.layer_module = layer
    names = {name for block in wrapper.simple_layer_modules(model.config, qcfg) for name in block}
    state.modules = {
        name: NamedModule(layer.get_submodule(name), name, f"model.layers.0.{name}", 0)
        for name in names
    }
    assert len(state.modules) == 16
    features = {name: torch.randn(7, named.module.in_features) for name, named in state.modules.items()}
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    frozen = {key: value.clone() for key, value in layer.state_dict().items() if key.removesuffix(".weight") not in names}
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
    modules = dict(state.modules)
    before = {name: named.module.weight.detach().clone() for name, named in modules.items()}
    processor._quantize_layer(0, state)
    assert state.quantized
    for name, named in modules.items():
        assert {"q_scales", "q_zeros", "wq"} <= named.state.keys()
        assert not torch.equal(named.module.weight, before[name]), name
        linear = named.module
        packed = AwqTorchLinear(
            bits=4, group_size=-1, sym=True, desc_act=False,
            in_features=linear.in_features, out_features=linear.out_features,
            bias=False, register_buffers=True,
        )
        packed.pack(linear, named.state["q_scales"], named.state["q_zeros"])
        assert packed.qweight.dtype == torch.int32
        recovered = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales, 4, -1)
        torch.testing.assert_close(recovered.T, linear.weight, rtol=1e-5, atol=1e-7)
    after = {name: named.module.weight.detach().clone() for name, named in modules.items()}
    processor._quantize_layer(0, state)
    for name, named in modules.items():
        torch.testing.assert_close(named.module.weight, after[name], rtol=0, atol=0)
    for key, value in frozen.items():
        torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[2, 4, 6, 8]])).logits).all()
