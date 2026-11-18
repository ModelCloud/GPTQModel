# SPDX-License-Identifier: Apache-2.0
"""Native aliases and FoT memory layers retain every actual checkpoint owner."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.local import LocalModelLoader, local_model_loader
from gptqmodel.models.ports.llama import dispatch
from tests.test_llama_local_port import NATIVE_TYPES, local_model


LINEARS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
CASES = [
    (family, heads, bias)
    for family in NATIVE_TYPES + ["stablelm_epoch", "yi"]
    for heads in (1, 2, 4)
    for bias in (False, True)
] + [("longllama", 4, False)]


def test_alias_loader_is_owned_by_this_definition_not_its_subclasses():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    assert isinstance(LlamaQModel.loader, LocalModelLoader)

    class DerivedDefinition(LlamaQModel):
        pass

    assert DerivedDefinition.loader is AutoModelForCausalLM
    # SenseNova-Loop inherits Qwen3.5's native image-text task factory.
    inherited_loaders = {"sensenova_loop": AutoModelForImageTextToText}
    for family, definition in MODEL_MAP.items():
        if (
            issubclass(definition, LlamaQModel)
            and definition is not LlamaQModel
            and "loader" not in definition.__dict__
        ):
            assert definition.loader is inherited_loaders.get(family, AutoModelForCausalLM), family
            assert local_model_loader(family) is None


@pytest.mark.parametrize("checkpoint", ["longllama", "stablelm_epoch", "yi_coder"])
def test_full_checkpoint_metadata_matches_every_meta_tensor(checkpoint):
    records = json.loads((Path(__file__).parents[1] / "fixtures/ports/llama/checkpoint_inventory.json").read_text())
    record = records[checkpoint]
    config = dispatch._config_class(record["config"]["model_type"]).from_dict(record["config"])
    with torch.device("meta"):
        model = LlamaQModel.loader.from_config(config)
    shapes = {key: list(value.shape) for key, value in model.state_dict().items()}

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert len(shapes) == record["key_count"]
    assert digest(sorted(shapes)) == record["keys_sha256"]
    assert digest(shapes) == record["shapes_sha256"]
    assert all(value.device.type == "meta" for value in model.state_dict().values())


def expected_keys(family, heads, bias):
    keys = {name + ".weight" for name in LINEARS}
    if family != "olmo":
        keys.add("input_layernorm.weight")
        if family not in {"cohere", "cohere2"}:
            keys.add("post_attention_layernorm.weight")
    if family in {"stablelm", "stablelm_epoch"}:
        keys |= {name + ".bias" for name in ("input_layernorm", "post_attention_layernorm")}
    if family == "longllama":
        keys.add("self_attn.rotary_emb.inv_freq")
    if bias:
        if family not in {"mistral", "longllama"}:
            keys |= {f"self_attn.{name}_proj.bias" for name in ("q", "k", "v")}
        if family in {"cohere", "cohere2", "gemma", "granite", "llama", "olmo", "smollm3", "yi"}:
            keys.add("self_attn.o_proj.bias")
        if family in {"granite", "helium", "llama", "seed_oss", "smollm3", "yi"}:
            keys |= {f"mlp.{name}_proj.bias" for name in ("gate", "up", "down")}
        if family == "cohere":
            keys |= {f"self_attn.{name}_norm.weight" for name in ("q", "k")}
        if family == "stablelm":
            keys |= {
                f"self_attn.{name}_layernorm.norms.{index}.weight"
                for name, count in (("q", 4), ("k", heads))
                for index in range(count)
            }
    return keys


@pytest.mark.parametrize("device", ["meta", "cpu"])
@pytest.mark.parametrize("family,heads,bias", CASES)
def test_real_alias_state_owners_and_plan_roles(family, heads, bias, device):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.device(device):
            model = local_model(
                family,
                num_hidden_layers=1,
                num_key_value_heads=heads,
                head_dim=8,
                attention_bias=bias,
                mlp_bias=bias,
                use_qkv_bias=bias,
                use_qk_norm=bias,
                qk_layernorm=bias,
            )
        assert MODEL_MAP[family] is LlamaQModel
        _convert_model_with_defuser(LlamaQModel, model, cleanup_original=False)
        tree = LlamaQModel._resolve_effective_module_tree(model, None)
        model.gptqmodel_effective_module_tree = tree
        assert LlamaQModel._resolve_effective_module_tree(model, None) == tree
        wrapper = LlamaQModel(model=model, quantized=False, quantize_config=None)
        plan = wrapper.module_plan()
        layer = model.model.layers[0]
        assert set(layer.state_dict()) == expected_keys(family, heads, bias)
        declared = {
            name
            for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None))
            for name in block
        }
        assert declared == LINEARS
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (family, key)
            assert spec.quantize == (owner in LINEARS)
            if owner not in LINEARS:
                assert set(spec.flags) & {"!", "?"}
        for name in LINEARS:
            projection = layer.get_submodule(name)
            assert isinstance(projection, nn.Linear)
            assert plan.find(name + ".undeclared") is None
            role = name.rsplit(".", 1)[-1].removesuffix("_proj")
            assert role in plan.find(name).flags
            if role in {"q", "k", "v", "gate", "up"}:
                assert "in=x" in plan.find(name).flags
        assert layer.self_attn.k_proj.weight.shape == (heads * 8, 32)
        assert layer.self_attn.v_proj.weight.shape == (heads * 8, 32)
        if family in {"longllama", "stablelm_epoch", "yi"}:
            assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        if device == "cpu":
            with torch.inference_mode():
                assert torch.isfinite(model(torch.tensor([[3, 7, 5, 9]]), use_cache=False).logits).all()
    finally:
        torch.set_num_threads(previous)
