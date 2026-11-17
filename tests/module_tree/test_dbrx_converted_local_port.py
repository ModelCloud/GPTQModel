# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-owned converted DBRX plans and conservative AWQ consumer safety."""


import hashlib

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.dbrx_converted import DbrxConvertedQModel
from gptqmodel.models.ports.dbrx_converted.configuration_dbrx import DbrxConfig
from gptqmodel.models.ports.dbrx_converted.modeling_dbrx import DbrxForCausalLM
from gptqmodel.quantization.config import AWQConfig, QuantizeConfig


def expected_layer_keys(experts):
    # Independent inventory observed in both pinned converted checkpoint indexes.
    keys = {"norm_attn_norm.norm_1.weight", "norm_attn_norm.norm_2.weight", "ffn.router.layer.weight"}
    keys.update(f"norm_attn_norm.attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "out_proj"))
    keys.update(f"ffn.experts.mlp.{expert}.{name}.weight" for expert in range(experts) for name in ("w1", "v1", "w2"))
    return keys


@pytest.mark.parametrize("variant", ["base", "instruct"])
def test_documented_checkpoint_layout_has_exact_native_weight_ownership(variant):
    config = DbrxConfig(
        d_model=6144, n_heads=48, n_layers=40, max_seq_len=32768, vocab_size=100352,
        attn_config={"clip_qkv": 8, "kv_n_heads": 8, "rope_theta": 500000},
        ffn_config={"ffn_hidden_size": 10752, "moe_num_experts": 16, "moe_top_k": 4,
                        "moe_jitter_eps": 0.01 if variant == "base" else 0, "moe_loss_weight": 0.05},
    )
    with torch.device("meta"):
        model = DbrxForCausalLM(config)
    keys = {f"transformer.blocks.{index}.{key}" for index in range(40) for key in expected_layer_keys(16)}
    keys.update({"lm_head.weight", "transformer.norm_f.weight", "transformer.wte.weight"})
    assert len(keys) == 2203  # Both authoritative safetensors indexes have exactly these keys.
    assert set(model.state_dict()) == keys
    assert hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest() == (
        "6cbda8bf37a5d87aa14c21e72ec5f5a2889e11f9a7267828b7d848ef53036b51"
    )
    assert model.transformer.blocks[0].norm_attn_norm.attn.q_proj.weight.shape == (6144, 6144)
    assert model.transformer.blocks[0].norm_attn_norm.attn.k_proj.weight.shape == (1024, 6144)
    assert model.transformer.blocks[0].ffn.experts.mlp[15].w2.weight.shape == (6144, 10752)


@pytest.mark.parametrize("experts,top_k", [(1, 1), (3, 2), (4, 4)])
@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_real_plan_covers_every_state_dict_owner_and_looper_family(experts, top_k, method):
    config = DbrxConfig(
        d_model=32, n_heads=4, n_layers=2, vocab_size=41,
        attn_config={"kv_n_heads": 2},
        ffn_config={"ffn_hidden_size": 48, "moe_num_experts": experts, "moe_top_k": top_k},
    )
    with torch.device("meta"):
        model = DbrxForCausalLM(config)
    qcfg = AWQConfig(device="cpu") if method == "awq" else QuantizeConfig(device="cpu")
    wrapped = DbrxConvertedQModel(model, quantized=False, quantize_config=qcfg)
    assert MODEL_MAP["dbrx_converted"] is DbrxConvertedQModel
    assert DbrxConvertedQModel.get_num_experts(config) == experts
    plan = wrapped.module_plan()
    quantized = {name for group in wrapped.simple_layer_modules(config, qcfg) for name in group}
    expected = expected_layer_keys(experts)
    assert len(quantized) == 4 + 3 * experts
    module_looper = ModuleLooper.__new__(ModuleLooper)
    weight_looper = WeightOnlyLooper.__new__(WeightOnlyLooper)
    module_looper.gptq_model = weight_looper.gptq_model = wrapped
    for layer in model.transformer.blocks:
        assert set(layer.state_dict()) == expected
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, owner
            if owner in quantized:
                assert spec.quantize
            else:
                assert not spec.quantize and "!" in spec.flags
            assert plan.find(owner + ".undeclared") is None
            if owner.startswith("ffn.experts.mlp."):
                family = owner.rsplit(".", 1)[0]
                assert plan.is_expert(owner)
                assert module_looper._extract_moe_group_key(owner) == family
                assert weight_looper._extract_moe_group_key(owner) == family
    assert [spec.template for spec in plan.moe_roots()] == ["ffn"]
    assert wrapped.fusable is None
    assert plan.find("ffn.experts.mlp_extra.0.w1") is None
    assert not plan.is_expert("ffn.router.layer")


def test_awq_policy_executes_real_plain_quantization_without_touching_router_or_norms(monkeypatch):
    config = DbrxConfig(
        d_model=32, n_heads=4, n_layers=1, vocab_size=41,
        attn_config={"kv_n_heads": 2, "clip_qkv": 0.15},
        ffn_config={"ffn_hidden_size": 48, "moe_num_experts": 3, "moe_top_k": 2},
    )
    model = DbrxForCausalLM(config).eval()
    qcfg = AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=False)
    wrapped = DbrxConvertedQModel(model, quantized=False, quantize_config=qcfg)
    block = model.transformer.blocks[0]
    assert wrapped.awq_get_modules_for_scaling(block, {}, {}) == []
    assert not wrapped.awq_input_feature_aggregation("ffn").get("capture_root", False)
    processor = AWQProcessor(
        tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(), gptq_model=wrapped, model=model,
        calculate_w_wq_diff=False,
    )
    monkeypatch.setattr(processor, "draw_progress", lambda *a, **k: None)
    monkeypatch.setattr(processor, "log_new_row", lambda *a, **k: None)
    processor.apply_clip = False
    quantized = {name for group in wrapped.simple_layer_modules(config, qcfg) for name in group}
    state = processor._get_layer_state(0)
    state.layer_module = block
    named = {name: NamedModule(block.get_submodule(name), name, "transformer.blocks.0." + name, 0) for name in quantized}
    state.modules = dict(named)
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: {})
    # Search/fold must never be entered; quantization below is not mocked.
    monkeypatch.setattr(processor, "_search_best_scale", lambda *a, **k: pytest.fail("unqualified AWQ search"))
    original = {name: tensor.clone() for name, tensor in block.state_dict().items()}
    expected = {}
    for name in quantized:
        weight = original[name + ".weight"]
        expected[name] = processor.pseudo_quantize_tensor(weight.clone(), bits=4, group_size=16)[0]
    processor._quantize_layer(0, state)
    assert state.quantized and not state.modules
    assert set(expected) == quantized and len(expected) == 13
    for name, module in named.items():
        torch.testing.assert_close(module.weight, expected[name], rtol=0, atol=0)
        assert not torch.equal(module.weight, original[name + ".weight"])
        assert "q_scales" in module.state and "q_zeros" in module.state
    for key in original:
        if key.rsplit(".", 1)[0] not in quantized:
            torch.testing.assert_close(block.state_dict()[key], original[key], rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 3, 7, 11]])).logits).all()
