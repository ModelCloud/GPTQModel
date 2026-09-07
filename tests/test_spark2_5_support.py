# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.spark2_5 import Spark2_5QModel


def test_spark2_5_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="spark2_5")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert (
        auto.check_and_get_model_definition(
            "/monster/data/model/Spark-X2.5-4B", trust_remote_code=True
        )
        is Spark2_5QModel
    )


def test_spark2_5_module_tree_matches_decoder_boundaries():
    config = SimpleNamespace(model_type="spark2_5")
    layer_modules = Spark2_5QModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert Spark2_5QModel.require_trust_remote_code is True
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

    full_modules = {
        module
        for block in Spark2_5QModel.full_layer_modules(model_config=config)
        for module in block
    }
    assert {
        "input_layernorm:!",
        "self_attn.g_proj:!",
        "post_attention_layernorm:!",
    } <= full_modules
    assert "self_attn.g_proj" not in {
        module for block in layer_modules for module in block
    }

    plan = Spark2_5QModel.shared_input_plan(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    assert plan.group_for("mlp.gate_proj").modules == (
        "mlp.gate_proj",
        "mlp.up_proj",
    )


def test_spark2_5_before_model_load_adapts_both_legacy_mask_helpers(monkeypatch):
    remote_module = types.ModuleType("fake_spark2_5_mask_remote")

    def fake_causal_mask(**kwargs):
        return kwargs

    class FakeSparkModel:
        pass

    FakeSparkModel.__module__ = remote_module.__name__
    remote_module.create_causal_mask = fake_causal_mask
    remote_module.create_sliding_window_causal_mask = fake_causal_mask
    monkeypatch.setitem(sys.modules, remote_module.__name__, remote_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_local_path: FakeSparkModel,
    )

    Spark2_5QModel.before_model_load(
        Spark2_5QModel, "/monster/data/model/Spark-X2.5-4B", False
    )

    for mask_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        mask = getattr(remote_module, mask_name)
        assert mask._gptqmodel_spark_mask_compat is True
        assert mask(input_embeds="hidden", cache_position="positions") == {
            "inputs_embeds": "hidden",
        }


def test_spark2_5_before_model_load_is_idempotent(monkeypatch):
    remote_module = types.ModuleType("fake_spark2_5_idempotent_remote")

    def fake_causal_mask(**kwargs):
        return kwargs

    class FakeSparkModel:
        pass

    FakeSparkModel.__module__ = remote_module.__name__
    remote_module.create_causal_mask = fake_causal_mask
    remote_module.create_sliding_window_causal_mask = fake_causal_mask
    monkeypatch.setitem(sys.modules, remote_module.__name__, remote_module)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda class_ref, model_local_path: FakeSparkModel,
    )

    Spark2_5QModel.before_model_load(Spark2_5QModel, "/tmp/spark2_5", False)
    first_helpers = tuple(
        getattr(remote_module, name)
        for name in ("create_causal_mask", "create_sliding_window_causal_mask")
    )
    Spark2_5QModel.before_model_load(Spark2_5QModel, "/tmp/spark2_5", True)
    second_helpers = tuple(
        getattr(remote_module, name)
        for name in ("create_causal_mask", "create_sliding_window_causal_mask")
    )

    assert second_helpers == first_helpers


def test_spark2_5_after_model_load_supplies_quantized_gate_weight_dtype():
    class FakeQuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compute_dtype = torch.bfloat16
            self.register_buffer("scales", torch.ones(1, dtype=torch.bfloat16))

    gate_proj = FakeQuantLinear()
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(gate_proj=gate_proj))],
        ),
    )

    assert not hasattr(gate_proj, "weight")
    returned = Spark2_5QModel.after_model_load(Spark2_5QModel, model, True)

    assert returned is model
    assert gate_proj.weight.numel() == 0
    assert gate_proj.weight.dtype is torch.bfloat16
    assert "weight" not in gate_proj.state_dict()


def test_spark2_5_after_model_load_leaves_dense_model_unchanged():
    dense = torch.nn.Linear(4, 4, bias=False)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace(gate_proj=dense))],
        ),
    )
    original_weight = dense.weight

    returned = Spark2_5QModel.after_model_load(Spark2_5QModel, model, False)

    assert returned is model
    assert dense.weight is original_weight
