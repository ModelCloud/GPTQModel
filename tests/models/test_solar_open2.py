# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from model_test import ModelTest
from torch import nn
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

from gptqmodel import BACKEND
from gptqmodel.models import auto
from gptqmodel.models.definitions.solar_open2 import SolarOpen2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.quantization.config import VramStrategy
from gptqmodel.utils.hf import build_shell_model
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = Path("/monster/data/model/Solar-Open2-250B")


def _tiny_model():
    transformers = pytest.importorskip("transformers")
    SolarOpen2Config = getattr(transformers, "SolarOpen2Config", None)
    SolarOpen2ForCausalLM = getattr(transformers, "SolarOpen2ForCausalLM", None)
    if SolarOpen2Config is None or SolarOpen2ForCausalLM is None:
        pytest.skip("Installed Transformers does not provide Solar Open 2")

    config = SolarOpen2Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        moe_intermediate_size=8,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        gqa_layers=[0],
        linear_attn_config={
            "short_conv_kernel_size": 2,
            "head_dim": 4,
            "num_heads": 4,
            "num_kv_heads": None,
        },
    )
    config._experts_implementation = "eager"
    return SolarOpen2ForCausalLM(config).eval()


def test_solar_open2_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="solar_open2")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/solar-open2") is SolarOpen2QModel


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Solar Open 2 checkpoint is unavailable")
def test_solar_open2_local_checkpoint_selects_definition():
    assert auto.check_and_get_model_definition(str(MODEL_PATH), trust_remote_code=False) is SolarOpen2QModel


def test_solar_open2_module_tree_covers_hybrid_attention_and_moe_paths():
    config = SimpleNamespace(n_routed_experts=3)
    quantize_config = SimpleNamespace(dynamic=None)
    layer_modules = SolarOpen2QModel.simple_layer_modules(config, quantize_config)
    flat_modules = {name for block in layer_modules for name in block}
    capture_modules = {
        name
        for block in SolarOpen2QModel.full_layer_modules(
            config,
            include_capture_only=True,
        )
        for name in block
    }

    assert SolarOpen2QModel.layer_modules_strict is False
    assert SolarOpen2QModel.dynamic_expert_index == "n_routed_experts"
    assert SolarOpen2QModel.extract_layers_node() == ["model.layers"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.g_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "self_attn.b_proj" not in flat_modules
    assert "self_attn.f_a_proj" not in flat_modules
    assert "self_attn.f_b_proj" not in flat_modules
    assert "self_attn.g_a_proj" not in flat_modules
    assert "self_attn.g_b_proj" not in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules
    assert "self_attn.q_norm:!" in capture_modules
    assert "self_attn.k_norm:!" in capture_modules
    assert "self_attn.o_norm:!" in capture_modules
    assert "self_attn.b_proj:!" in capture_modules
    assert "self_attn.f_a_proj:!" in capture_modules
    assert "self_attn.f_b_proj:!" in capture_modules
    assert "self_attn.g_a_proj:!" in capture_modules
    assert "self_attn.g_b_proj:!" in capture_modules
    assert "mlp.gate:!" in capture_modules


def test_solar_open2_replay_rebuilds_full_and_linear_attention_masks():
    model = _tiny_model()
    wrapper = SolarOpen2QModel.__new__(SolarOpen2QModel)
    hidden_states = torch.zeros(1, 6, model.config.hidden_size)
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    full_layer = model.model.layers[0]
    linear_layer = model.model.layers[1]
    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": padding_mask,
        "past_key_values": None,
        "position_ids": None,
    }
    captured_full_mask = create_causal_mask(
        **mask_kwargs,
        layer_idx=full_layer.self_attn.layer_idx,
    )
    wrapper.__dict__["_solar_open2_capture_padding_mask"] = padding_mask
    captured_kwargs = wrapper.capture_first_layer_input_kwargs(
        args=(hidden_states,),
        kwargs={"attention_mask": captured_full_mask},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )
    base_inputs = {
        "attention_mask": captured_full_mask,
        "past_key_values": None,
        **captured_kwargs,
    }

    full_inputs = wrapper.prepare_layer_replay_kwargs(
        full_layer,
        [hidden_states],
        dict(base_inputs),
        torch.device("cpu"),
    )
    linear_inputs = wrapper.prepare_layer_replay_kwargs(
        linear_layer,
        [hidden_states],
        dict(base_inputs),
        torch.device("cpu"),
    )
    expected_linear_mask = create_recurrent_attention_mask(**mask_kwargs)

    assert torch.equal(full_inputs["attention_mask"], captured_full_mask)
    assert torch.equal(linear_inputs["attention_mask"], expected_linear_mask)
    assert linear_inputs["attention_mask"].shape == padding_mask.shape


def test_solar_open2_defuser_expands_routed_experts_without_changing_forward():
    from defuser.model_registry import MODEL_CONFIG

    assert "solar_open2" in MODEL_CONFIG

    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 7, 8, 2]])
    packed_experts = model.model.layers[0].mlp.experts

    assert hasattr(packed_experts, "gate_up_proj")
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits

    assert _convert_model_with_defuser(SolarOpen2QModel, model, cleanup_original=False) is True

    experts = model.model.layers[0].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert isinstance(experts[0].gate_proj, nn.Linear)
    assert isinstance(experts[0].up_proj, nn.Linear)
    assert isinstance(experts[0].down_proj, nn.Linear)
    with torch.inference_mode():
        actual = model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_solar_open2_lazy_turtle_materializes_defused_expert_weights(tmp_path):
    source = _tiny_model()
    packed_experts = source.model.layers[0].mlp.experts
    expected_gate, expected_up = packed_experts.gate_up_proj[0].detach().chunk(2, dim=0)
    expected_down = packed_experts.down_proj[0].detach()
    source.save_pretrained(tmp_path, safe_serialization=True)

    shell = build_shell_model(
        SolarOpen2QModel.loader,
        config=copy.deepcopy(source.config),
        trust_remote_code=False,
        device_map={"": "cpu"},
        _fast_init=True,
    )
    assert _convert_model_with_defuser(SolarOpen2QModel, shell, cleanup_original=False) is True
    turtle = LazyTurtle(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={},
        module_tree=SolarOpen2QModel.module_tree,
        hf_conversion_map_reversed=SolarOpen2QModel.resolve_hf_conversion_map_reversed(shell),
        target_model=shell,
    )

    expert = shell.model.layers[0].mlp.experts[0]
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=expert,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(expert.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert.up_proj.weight, expected_up)
    torch.testing.assert_close(expert.down_proj.weight, expected_down)


class TestSolarOpen2(ModelTest):
    NATIVE_MODEL_ID = str(MODEL_PATH)
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.AUTO
    EVAL_BATCH_SIZE = 1
    EVAL_SINGLE_GPU = False
    MODEL_COMPAT_FAST_LAYER_COUNT = 2
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    EVAL_TASKS_SLOW = {
        "mmlu_pro": {
            "chat_template": True,
            # Native score published with the local Solar Open 2 checkpoint.
            "em,choice_label": {"value": 0.862, "floor_pct": 0.10},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_solar_open2(self):
        self.quantize_and_evaluate()


__all__ = ["TestSolarOpen2"]
