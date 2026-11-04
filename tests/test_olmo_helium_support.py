# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""OLMo and Helium reuse the Llama module tree; pin that assumption to the source."""

from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.llama import LlamaQModel


CASES = [("olmo", "Olmo"), ("helium", "Helium")]


@pytest.mark.parametrize("model_type,_prefix", CASES)
def test_registry_selects_llama_definition(model_type, _prefix, monkeypatch):
    fake_config = SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.MODEL_MAP[model_type] is LlamaQModel
    assert auto.check_and_get_model_definition(f"/tmp/{model_type}", trust_remote_code=True) is LlamaQModel


@pytest.mark.parametrize("model_type,prefix", CASES)
def test_real_decoder_layer_matches_llama_module_tree(model_type, prefix):
    """Build the real config/decoder layer and assert the quantized shape layout.

    The Llama definition assumes ``layers.#.self_attn.{q,k,v,o}_proj``,
    ``layers.#.mlp.{gate,up,down}_proj``, ``layers.#.{input,post_attention}_layernorm``,
    and a top-level ``model.norm``. If an upstream transformers release renames
    or restructures any of these, quantization would silently target the wrong
    modules, so fail here instead.
    """
    import importlib

    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type.replace(chr(45), chr(95))}"
    )
    cfg_cls = getattr(module, f"{prefix}Config")
    layer_cls = getattr(module, f"{prefix}DecoderLayer")

    cfg = cfg_cls(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, num_hidden_layers=2, vocab_size=256)
    layer = layer_cls(cfg, layer_idx=0)

    assert LlamaQModel.pre_lm_head_norm_module == "model.norm"
    assert LlamaQModel.module_tree[0] == "model"
    assert LlamaQModel.module_tree[1] == "layers"
    assert LlamaQModel.module_tree[2] == "#"

    tree = LlamaQModel.module_tree[3]
    assert set(tree) == {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"}

    names = {name for name, _ in layer.named_modules()}
    for required in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        assert required in names, f"{model_type} decoder layer lost `{required}`"


@pytest.mark.parametrize("model_type,prefix", CASES)
def test_definition_emits_llama_layer_modules(model_type, prefix):
    quantize_config = SimpleNamespace(dynamic=None)
    simple = LlamaQModel.simple_layer_modules(SimpleNamespace(), quantize_config)
    full = LlamaQModel.full_layer_modules(SimpleNamespace())

    assert simple, f"{model_type} produced no quantizable layer modules"
    flat = [name for block in simple for name in block]
    for required in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        assert required in flat, f"{model_type} would not quantize `{required}`"
    assert full, f"{model_type} produced no full layer modules"
