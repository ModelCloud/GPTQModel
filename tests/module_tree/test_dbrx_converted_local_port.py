# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-owned converted DBRX plans and AWQ scale-fold safety."""


import hashlib

import pytest
import torch

from gptqmodel.looper.module_looper import ModuleLooper
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


@pytest.mark.parametrize("kv_heads,clip_qkv", [(2, 0.15), (4, 0.15), (4, None)])
def test_awq_scale_folds_preserve_attention_and_routed_expert_output(kv_heads, clip_qkv):
    from awq_neutrality import assert_awq_folds_neutral

    from gptqmodel.models.loader import _convert_model_with_defuser
    from tests.test_dbrx_converted_local_port import make_model

    # top_k == experts routes every token through every expert, so every
    # expert's v1 -> w2 group is captured and checked.
    model = make_model(attn_config={"kv_n_heads": kv_heads, "clip_qkv": clip_qkv, "rope_theta": 500000},
                       ffn_config={"ffn_hidden_size": 48, "moe_num_experts": 3, "moe_top_k": 3})
    _convert_model_with_defuser(DbrxConvertedQModel, model, cleanup_original=False)
    tree = DbrxConvertedQModel._resolve_effective_module_tree(model, None)
    # The norm_2 -> experts group reads the shared pre-router input.
    assert DbrxConvertedQModel.awq_input_feature_aggregation("ffn")["capture_root"] is True
    ids = torch.tensor([[1, 3, 7, 11, 19, 23]])

    def forward():
        return model(ids, use_cache=False).logits

    experts = [f"ffn.experts.mlp.{i}" for i in range(3)]
    expected = [
        ("norm_attn_norm.norm_1", [f"norm_attn_norm.attn.{name}_proj" for name in "qkv"], []),
        ("norm_attn_norm.attn.v_proj", ["norm_attn_norm.attn.out_proj"], []),
        ("norm_attn_norm.norm_2", [f"{e}.{w}" for e in experts for w in ("w1", "v1")], ["ffn.router.layer"]),
    ] + [(f"{e}.v1", [f"{e}.w2"], []) for e in experts]
    if clip_qkv is not None or kv_heads != 4:
        # `clip_qkv` clamps v_proj's output, so its row scale cannot be undone
        # by out_proj (the fold changed the output by up to 20%); GQA also
        # narrows v below out_proj's input.
        expected.pop(1)
    for block in model.transformer.blocks:
        checked = assert_awq_folds_neutral(DbrxConvertedQModel, model, block, forward, tree=tree)
        # The router reads norm_2 and is compensated; the activated `w1`
        # branch is never a fold source for `w2`.
        assert checked == expected
