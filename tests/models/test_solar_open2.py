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
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

from gptqmodel import WeightOnlyConfig
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
    EVAL_BATCH_SIZE = 16
    EVAL_SINGLE_GPU = False
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4889, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4642, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)


    def test_solar_open2(self):
        self.quantize_and_evaluate()

__all__ = ["TestSolarOpen2"]
