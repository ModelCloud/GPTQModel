# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Module plans versus actual tiny and 253B-meta checkpoints, with pinned key digests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import rotation_supported
from gptqmodel.models.definitions.decilm import DeciLMQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from tests.test_decilm_local_port import config_kwargs, reference_forward, tiny


QC = SimpleNamespace(dynamic=None)
CONTRACT_PATH = Path(__file__).parents[1] / "fixtures/ports/decilm/checkpoint_contracts.json"


def assert_weight_ownership(model):
    _convert_model_with_defuser(DeciLMQModel, model, cleanup_original=False)
    wrapper = DeciLMQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    declared = {entry for block in wrapper.simple_layer_modules(model.config, QC) for entry in block}
    owners = set()
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            owners.add(owner)
            spec = plan.find(owner)
            assert spec is not None, key
            module = layer.get_submodule(owner)
            if isinstance(module, torch.nn.Linear):
                assert spec.quantize and owner in declared
            else:
                assert not spec.quantize and ("!" in spec.flags or "?" in spec.flags)
    for path in ("self_attn.q_proj", "mlp.up_proj"):
        assert plan.find(path + ".undeclared") is None
        assert plan.find("prefix_" + path) is None
    for path, role in [("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v"),
                       ("self_attn.o_proj", "o"), ("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"),
                       ("mlp.down_proj", "down")]:
        assert plan.roles(path) == (role,)
    assert wrapper.pre_lm_head_norm_module == "model.norm"
    assert not DeciLMQModel.shared_input_verified(model.config)
    assert not rotation_supported(wrapper)
    return owners


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
def test_alias_real_weight_owners_and_plan_roles(alias):
    assert MODEL_MAP[alias] is DeciLMQModel
    model = tiny(alias, attention_bias=True, mlp_bias=True)
    owners = assert_weight_ownership(model)
    assert len(owners) == (9 if alias == "deci" else 11)


@pytest.mark.parametrize("attention", ["full", "linear", "no_op"])
@pytest.mark.parametrize("ffn", ["full", "linear", "no_op"])
def test_all_nas_subblock_layouts_are_checkpoint_owned(attention, ffn):
    def part(kind, full):
        return full if kind == "full" else {"replace_with_linear" if kind == "linear" else "no_op": True}
    block = {"attention": part(attention, {"n_heads_in_group": 2}), "ffn": part(ffn, {"ffn_mult": 2.0})}
    model = tiny("nemotron-nas", block_configs=[block], num_hidden_layers=1)
    assert_weight_ownership(model)
    with torch.inference_mode():
        ids = torch.tensor([[1, 3, 7]])
        actual = model(ids).logits
        torch.testing.assert_close(actual, reference_forward(model, ids), rtol=2e-5, atol=1e-6)
        # Even entirely pruned blocks must retain generation progress.
        continuation = model(ids[:, 2:], past_key_values=model(ids[:, :2]).past_key_values).logits
        torch.testing.assert_close(continuation, actual[:, 2:], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
def test_pinned_real_checkpoint_manifest_matches_exact_meta_key_set(alias):
    from gptqmodel.models.ports.decilm.configuration_decilm import DeciLMConfig

    contract = json.loads(CONTRACT_PATH.read_text())[alias]
    payload = contract["config"]
    if alias == "nemotron-nas":
        # Lossless encoding of the pinned 162-layer upstream config: all other
        # attention/FFN flags are false/None in this released checkpoint.
        payload["block_configs"] = [
            {"attention": {"no_op": group == 0, "n_heads_in_group": group or None},
             "ffn": {"no_op": mult == 0, "ffn_mult": mult or None}}
            for group, mult in contract["blocks_attention_group_ffn_mult"]
        ]
    config = DeciLMConfig(**payload)
    if alias == "nemotron-nas":
        assert config.num_key_value_heads is None and config.intermediate_size is None
    with torch.device("meta"):
        model = DeciLMQModel.loader.from_config(config)
    keys = sorted(model.state_dict())
    assert len(keys) == contract["key_count"]
    assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == contract["sorted_keys_sha256"]
    assert_weight_ownership(model)


def test_registry_proposal_contains_linear_variants_and_norm_exclusions():
    """Pin the exact parent-owned JSON entry while this unit avoids shared files."""
    expected = [
        ["input_layernorm:!"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.linear_attn"],
        ["self_attn.o_proj"],
        ["post_attention_layernorm:!"],
        ["mlp.gate_proj", "mlp.up_proj", "mlp.linear_mlp"],
        ["mlp.down_proj"],
    ]
    assert DeciLMQModel.extract_layers_node() == ["model.layers"]
    assert DeciLMQModel.build_layer_modules(DeciLMQModel.module_tree, include_capture_only=True) == expected
    assert config_kwargs("deci")["model_type"] != config_kwargs("nemotron-nas")["model_type"]


def _nas_layouts():
    def part(kind, full):
        return full if kind == "full" else {"replace_with_linear" if kind == "linear" else "no_op": True}

    blocks = [
        {"attention": part(attention, {"n_heads_in_group": 1}), "ffn": part(ffn, {"ffn_mult": 2.0})}
        for attention in ("full", "linear", "no_op")
        for ffn in ("full", "linear", "no_op")
    ]
    # Grouped-query attention: `v_proj` is narrower than `o_proj`'s input.
    blocks.append({"attention": {"n_heads_in_group": 2}, "ffn": {"ffn_mult": 2.0}})
    return blocks


@pytest.mark.parametrize("alias", ["deci", "nemotron-nas"])
def test_awq_folds_are_neutral_across_every_nas_layout(alias):
    from awq_neutrality import assert_awq_folds_neutral

    if alias == "deci":
        # Per-layer KV heads 2/1/4: only the last layer is full MHA.
        model = tiny(alias, attention_bias=True, mlp_bias=True)
    else:
        model = tiny(alias, block_configs=_nas_layouts(), num_hidden_layers=10)
    ids = torch.randint(2, 30, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    qkv = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(DeciLMQModel, model, layer, forward) if any(
            name in {"self_attn", "mlp"} for name, _child in layer.named_children()
        ) else []
        groups = {source: targets for source, targets, _consumers in checked}
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        expected = {}
        if hasattr(attn, "q_proj"):
            expected["input_layernorm"] = qkv
            if attn.v_proj.out_features == attn.o_proj.in_features:
                expected["self_attn.v_proj"] = ["self_attn.o_proj"]
        elif attn is not None:
            expected["input_layernorm"] = ["self_attn.linear_attn"]
        if hasattr(mlp, "up_proj"):
            expected["post_attention_layernorm"] = ["mlp.gate_proj", "mlp.up_proj"]
            expected["mlp.up_proj"] = ["mlp.down_proj"]
        elif mlp is not None:
            expected["post_attention_layernorm"] = ["mlp.linear_mlp"]
        # `o_proj` never joins the input-norm group, even though the absent
        # `linear_attn` alternative ends the Q/K/V block.
        assert groups == expected
        assert len(checked) == len(groups)
