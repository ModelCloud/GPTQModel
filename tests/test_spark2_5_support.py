# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Spark definition regressions use executable package-owned decoder instances."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.spark2_5 import Spark2_5QModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.spark2_5.modeling_spark import Spark2_5ForCausalLM
from tests.test_spark2_5_local_port import tiny_model


def test_spark2_5_model_type_selects_definition(tmp_path, monkeypatch):
    tiny_model().config.save_pretrained(tmp_path)

    def checkpoint_python_forbidden(*args, **kwargs):
        raise AssertionError("Package-owned Spark must never import checkpoint Python")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", checkpoint_python_forbidden)
    for trust in (False, True):
        assert auto.check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is Spark2_5QModel
    assert isinstance(Spark2_5QModel.loader, LocalModelLoader)


def test_spark2_5_module_tree_matches_decoder_boundaries():
    config = SimpleNamespace(model_type="spark2_5")
    layer_modules = Spark2_5QModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert Spark2_5QModel.require_trust_remote_code is False
    assert Spark2_5QModel.shared_input_verified(config) is True
    assert Spark2_5QModel.extract_layers_node() == ["model.layers"]
    assert Spark2_5QModel.pre_lm_head_norm_module == "model.norm"
    assert Spark2_5QModel.rotary_embedding is None
    assert layer_modules == [
        ["self_attn.q_k_v_proj"],
        ["self_attn.out_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]

    full_modules = {module for block in Spark2_5QModel.full_layer_modules(model_config=config) for module in block}
    assert {
        "input_layernorm:!",
        "self_attn.g_proj:!",
        "post_attention_layernorm:!",
    } <= full_modules
    assert "self_attn.g_proj" not in {module for block in layer_modules for module in block}

    plan = Spark2_5QModel.shared_input_plan(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    assert plan.group_for("mlp.gate_proj").modules == (
        "mlp.gate_proj",
        "mlp.up_proj",
    )


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_spark2_5_current_native_masks_and_hybrid_cache_repeated_load(tmp_path, implementation):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8]])
    for _ in range(2):
        loaded = Spark2_5QModel.loader.from_pretrained(tmp_path, attn_implementation=implementation).eval()
        assert type(loaded) is Spark2_5ForCausalLM
        with torch.inference_mode():
            expected = loaded(tokens, use_cache=False).logits
            prefix = loaded(tokens[:, :2])
            continuation = loaded(tokens[:, 2:], past_key_values=prefix.past_key_values)
        torch.testing.assert_close(continuation.logits, expected[:, 2:], atol=1e-6, rtol=1e-5)
        assert continuation.past_key_values.get_seq_length() == tokens.shape[1]


def test_spark2_5_dense_gate_uses_real_weight_without_remote_hook():
    model = tiny_model().bfloat16()
    weight = model.model.layers[0].mlp.gate_proj.weight
    assert Spark2_5QModel.after_model_load(Spark2_5QModel, model, False) is model
    assert model.model.layers[0].mlp.gate_proj.weight is weight
    assert weight.shape == (48, 32)
    with torch.inference_mode():
        assert model(torch.tensor([[3, 4, 5]])).logits.isfinite().all()


def test_auto_nf12_hooked_linear_exposes_source_weight_contract():
    """The legacy packed adapter retains its bounded source-weight metadata.

    The local Spark port uses norm-owned dtype and no longer needs this adapter
    contract; keep the generic NF12 regression for older source-derived callers.
    """

    from gptqmodel.nn_modules.hooked_linear import HookedLinear
    from gptqmodel.utils.auto_dtype_cast import AutoNF12HookedLinear

    source = HookedLinear(64, 16)
    source.weight = torch.randn(16, 64, dtype=torch.bfloat16)
    source.bias = None

    layer = AutoNF12HookedLinear.from_linear(source)

    assert isinstance(layer.weight, torch.Tensor)
    assert layer.weight.shape == source.weight.shape
    assert layer.weight.dtype is source.weight.dtype
    assert layer.weight.device == source.weight.device
    # A one-element storage with zero strides, not a materialized matrix.
    assert layer.weight.untyped_storage().nbytes() == layer.weight.element_size()
    assert "weight" not in layer.state_dict()
