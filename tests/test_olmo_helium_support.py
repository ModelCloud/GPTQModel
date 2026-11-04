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
    import inspect

    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type.replace(chr(45), chr(95))}"
    )
    cfg_cls = getattr(module, f"{prefix}Config")
    layer_cls = getattr(module, f"{prefix}DecoderLayer")

    hidden_size, num_heads, num_kv_heads = 64, 4, 2
    kwargs = {
        "hidden_size": hidden_size,
        "intermediate_size": 128,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": 2,
        "vocab_size": 256,
    }
    # Helium declares `head_dim` as an independent field defaulting to 128, so it
    # does not derive from `hidden_size / num_attention_heads` like Llama does.
    # Leaving the default makes the decoder layer internally inconsistent
    # (`q_proj` would project to num_heads*128 while `o_proj` still expects
    # hidden_size), so pin it to the real model's invariant.
    head_dim = hidden_size // num_heads
    if "head_dim" in inspect.signature(cfg_cls).parameters:
        kwargs["head_dim"] = head_dim
    cfg = cfg_cls(**kwargs)
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

    # Names alone are not enough: the Llama tree also assumes these projections
    # line up, so a shape-only upstream change must fail here too.
    attn = layer.self_attn
    assert attn.q_proj.in_features == hidden_size, model_type
    assert attn.q_proj.out_features == num_heads * head_dim, model_type
    assert attn.k_proj.out_features == num_kv_heads * head_dim, model_type
    assert attn.v_proj.out_features == num_kv_heads * head_dim, model_type
    assert attn.o_proj.in_features == attn.q_proj.out_features, (
        f"{model_type}: o_proj consumes {attn.o_proj.in_features} but q_proj emits "
        f"{attn.q_proj.out_features}"
    )
    assert attn.o_proj.out_features == hidden_size, model_type
    mlp = layer.mlp
    assert mlp.gate_proj.in_features == hidden_size, model_type
    assert mlp.up_proj.in_features == hidden_size, model_type
    assert mlp.down_proj.in_features == mlp.up_proj.out_features, model_type
    assert mlp.down_proj.out_features == hidden_size, model_type


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


@pytest.mark.parametrize("model_type,prefix", CASES)
def test_toplevel_norm_shape_contract(model_type, prefix):
    """Pin the assumption norm tuning makes about each model's top-level norm.

    `LlamaQModel` tags every decoder norm with `:norm`, and `norm_module_names`
    always appends `pre_lm_head_norm_module` (= `model.norm`). Norm tuning then
    requires a trainable floating-point `weight` on each selected module, so a
    norm without affine parameters turns an opt-in `requantize(norm=...)` call
    into a hard `ValueError` instead of a no-op.

    OLMo is exactly that case: its `OlmoLayerNorm` is parameter-free. It is
    registered here so the regression is visible rather than discovered at
    runtime; fixing it means declaring a non-affine-norm exception for `olmo`.
    """
    import importlib
    import inspect

    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type.replace(chr(45), chr(95))}"
    )
    cfg_cls = getattr(module, f"{prefix}Config")
    model_cls = getattr(module, f"{prefix}ForCausalLM")

    hidden_size, num_heads = 64, 4
    kwargs = {
        "hidden_size": hidden_size,
        "intermediate_size": 128,
        "num_attention_heads": num_heads,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 256,
    }
    if "head_dim" in inspect.signature(cfg_cls).parameters:
        kwargs["head_dim"] = hidden_size // num_heads
    model = model_cls(cfg_cls(**kwargs))

    norm = model.get_submodule(LlamaQModel.pre_lm_head_norm_module)
    params = dict(norm.named_parameters())

    if model_type == "olmo":
        # Documented, currently-unfixed regression: OLMo's norm has no gain, so
        # norm tuning cannot select it. Keep the expectation explicit so the
        # failure message points at the real cause when someone wires up tuning.
        assert params == {}, (
            "olmo's top-level norm unexpectedly gained parameters; the "
            "`OlmoLayerNorm` non-affine exception may no longer be needed"
        )
        with pytest.raises(ValueError, match="not a floating point trainable gain"):
            _resolve_norm_params(model, LlamaQModel.pre_lm_head_norm_module)
    else:
        assert "weight" in params, f"{model_type}: top-level norm has no trainable gain"


def _resolve_norm_params(model, name):
    from gptqmodel.quantization.config import NormType
    from gptqmodel.quantization.norm_tuning import resolve_norm_params

    return resolve_norm_params(model, [name], types=(NormType.LAYERNORM, NormType.RMSNORM))
