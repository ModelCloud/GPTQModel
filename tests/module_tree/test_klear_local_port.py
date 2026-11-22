# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Klear checkpoint-owned state, dense/MoE variants and real plan consumers."""

import re
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.definitions.klear import KlearQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.klear.configuration_klear import KlearConfig
from gptqmodel.models.ports.klear.modeling_klear import KlearMoeForCausalLM
from tests.test_klear_local_port import tiny_config


# Independently transcribed from both pinned public safetensors indices.
# 25,059 keys = 3 top-level + 32 * (15 + 3 * 256) decoder keys.
CHECKPOINT_PATTERNS = {
    "lm_head.weight", "model.embed_tokens.weight", "model.norm.weight",
    "model.layers.#.input_layernorm.weight", "model.layers.#.post_attention_layernorm.weight",
    "model.layers.#.mlp.coefficient.bias", "model.layers.#.mlp.coefficient.weight",
    "model.layers.#.mlp.expert_bias", "model.layers.#.mlp.gate.weight",
    *(f"model.layers.#.mlp.experts.#.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")),
    *(f"model.layers.#.mlp.shared_experts.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")),
    *(f"model.layers.#.self_attn.{projection}.weight" for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")),
}

EXPECTED_PLAN = {
    "layers": ["model.layers"],
    "blocks": [
        ["input_layernorm:!"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm:!", "self_attn.k_norm:!"],
        ["self_attn.o_proj"], ["post_attention_layernorm:!"], ["mlp:?"], ["mlp.gate:!"],
        ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
        ["mlp.experts.{expert_index}.down_proj"], ["mlp.coefficient:!"],
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"], ["mlp.shared_experts.down_proj"],
        ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"],
    ],
}


def _wrapper(model):
    return KlearQModel(model=model, quantized=False, quantize_config=None)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("layout", ["moe", "dense", "mixed", "periodic", "no-shared"])
@pytest.mark.parametrize("kv,bias", [(1, False), (2, True), (4, False)])
def test_real_state_owners_and_consumers(device, layout, kv, bias):
    variants = {"moe": {}, "dense": {"num_experts": 0}, "mixed": {"mlp_only_layers": [0]},
                "periodic": {"decoder_sparse_step": 2}, "no-shared": {"n_shared_experts": 0}}
    config = tiny_config(num_key_value_heads=kv, attention_bias=bias, **variants[layout])
    with torch.device(device):
        model = KlearQModel.loader.from_config(config)
    assert type(model) is KlearMoeForCausalLM
    _convert_model_with_defuser(KlearQModel, model, cleanup_original=False)
    wrapper = _wrapper(model)
    plan = wrapper.module_plan()
    declared = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    looper, weight_only = ModuleLooper.__new__(ModuleLooper), WeightOnlyLooper.__new__(WeightOnlyLooper)
    looper.gptq_model = weight_only.gptq_model = wrapper
    for layer in model.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
                quantized.add(owner)
            else:
                assert {"!", "?"}.intersection(spec.flags), key
        sparse = hasattr(layer.mlp, "experts")
        expected = 4 + (3 * config.num_experts + 3 if sparse else 3)
        assert len(quantized) == expected
        if sparse:
            hooks = KlearQModel.moe_lifecycle_hooks
            assert hooks.get_moe_block(layer, KlearQModel) is layer.mlp
            assert hooks.get_experts_module(layer.mlp, KlearQModel) is layer.mlp.experts
            assert hooks.get_shared_experts_module(layer.mlp, KlearQModel) is layer.mlp.shared_experts
            assert plan.find("mlp").capture_only  # Owns the persistent expert_bias tensor.
            for i in range(config.num_experts):
                path = f"mlp.experts.{i}.gate_proj"
                assert looper._extract_moe_group_key(path) == f"mlp.experts.{i}"
                assert weight_only._extract_moe_group_key(path) == f"mlp.experts.{i}"
    for path in ["mlp.coefficient_extra", "self_attn.q_norm.child", "mlp.experts.1.undeclared"]:
        assert plan.find(path) is None
    assert KlearQModel.fusable is None


@pytest.mark.parametrize("layout", ["moe", "dense", "mixed", "periodic", "no-shared"])
@pytest.mark.parametrize("kv,bias", [(1, False), (2, True), (4, False)])
def test_awq_folds_are_neutral_on_dense_and_moe_layers(layout, kv, bias):
    variants = {"moe": {}, "dense": {"num_experts": 0}, "mixed": {"mlp_only_layers": [0]},
                "periodic": {"decoder_sparse_step": 2}, "no-shared": {"n_shared_experts": 0}}
    torch.manual_seed(3)
    config = tiny_config(num_key_value_heads=kv, attention_bias=bias, **variants[layout])
    model = KlearQModel.loader.from_config(config).eval()
    _convert_model_with_defuser(KlearQModel, model, cleanup_original=False)
    ids = torch.tensor([[1, 7, 8, 9, 5, 6, 11, 3]])

    def forward():
        return model(ids).logits

    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(KlearQModel, model, layer, forward)
        groups = {(source, tuple(targets)): consumers for source, targets, consumers in checked}
        assert groups[("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))] == []
        if hasattr(layer.mlp, "experts"):
            experts = range(config.num_experts)
            targets = [f"mlp.experts.{i}.{p}" for i in experts for p in ("gate_proj", "up_proj")]
            targets += ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"]
            # The fp32 sigmoid router and the shared/routed mixer both read the
            # normed input, so both absorb the inverse scale. expert_bias is
            # added after the sigmoid and is scale-free.
            assert groups[("post_attention_layernorm", tuple(targets))] == ["mlp.gate", "mlp.coefficient"]
            for index in experts:
                assert (f"mlp.experts.{index}.up_proj", (f"mlp.experts.{index}.down_proj",)) in groups
            shared = config.n_shared_experts > 0
            assert (("mlp.shared_experts.up_proj", ("mlp.shared_experts.down_proj",)) in groups) == shared
        else:
            assert groups[("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"))] == []
            assert ("mlp.up_proj", ("mlp.down_proj",)) in groups


def test_checkpoint_key_patterns_and_exact_plan():
    model = KlearMoeForCausalLM(tiny_config())
    patterns = {re.sub(r"(layers|experts)\.\d+\.", r"\1.#.", name) for name in model.state_dict()}
    assert patterns == CHECKPOINT_PATTERNS
    assert len(model.state_dict()) == 3 + 2 * (15 + 3 * 3)
    assert KlearQModel.extract_layers_node() == EXPECTED_PLAN["layers"]
    assert KlearQModel.build_layer_modules(KlearQModel.module_tree, include_capture_only=True) == EXPECTED_PLAN["blocks"]
    plan = ModulePlan.from_tree(KlearQModel.module_tree)
    for path, role in [("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"),
                       ("self_attn.v_proj", "v"), ("self_attn.o_proj", "o"),
                       ("mlp.experts.0.gate_proj", "gate"), ("mlp.shared_experts.up_proj", "up")]:
        assert role in plan.find(path).roles


def test_pinned_base_and_instruct_checkpoint_full_meta_inventory():
    config = KlearConfig(
        vocab_size=151936, hidden_size=2048, intermediate_size=8064,
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=4,
        moe_intermediate_size=896, num_experts=256, num_experts_per_tok=8,
        n_shared_experts=1, max_position_embeddings=65536, rope_theta=500000.0,
        rms_norm_eps=1e-5, attn_implementation="eager",
    )
    with torch.device("meta"):
        model = KlearQModel.loader.from_config(config)
    checkpoint_names = set()
    for pattern in CHECKPOINT_PATTERNS:
        if "layers.#." not in pattern:
            checkpoint_names.add(pattern)
            continue
        for layer in range(32):
            name = pattern.replace("layers.#.", f"layers.{layer}.")
            if "experts.#." in name:
                checkpoint_names.update(name.replace("experts.#.", f"experts.{expert}.") for expert in range(256))
            else:
                checkpoint_names.add(name)
    actual = model.state_dict()
    assert len(checkpoint_names) == 25059
    assert set(actual) == checkpoint_names
    assert all(tensor.is_meta for tensor in actual.values())
