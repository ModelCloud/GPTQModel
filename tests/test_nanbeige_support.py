# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel
from gptqmodel.models.ports.nanbeige.configuration_nanbeige import NanbeigeConfig
from gptqmodel.models.ports.nanbeige.modeling_nanbeige import NanbeigeForCausalLM


def tiny_config():
    return NanbeigeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_loops=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.mark.parametrize("trust", [False, True])
def test_nanbeige_model_type_selects_owned_definition_and_constructor(tmp_path, trust):
    config = tiny_config()
    config.auto_map = {"AutoConfig": "absent.RemoteConfig", "AutoModelForCausalLM": "absent.RemoteModel"}
    config.save_pretrained(tmp_path)
    definition = auto.check_and_get_model_definition(str(tmp_path), trust_remote_code=trust)
    assert definition is NanbeigeQModel
    selected = definition.loader.load_config(str(tmp_path), trust_remote_code=trust)
    constructed = definition.loader.from_config(selected, trust_remote_code=trust, code_revision="absent")
    assert type(selected) is NanbeigeConfig
    assert type(constructed) is NanbeigeForCausalLM
    assert len(constructed.model.layers) == 2
    assert definition.require_trust_remote_code is False


def test_nanbeige_owned_physical_projection_groups_remain_stable():
    assert NanbeigeQModel.simple_layer_modules(tiny_config(), SimpleNamespace(dynamic=None)) == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]


def test_nanbeige_rotary_frequency_is_constructed_and_rematerialized_without_remote_patch():
    with torch.device("meta"):
        model = NanbeigeForCausalLM(tiny_config())
    model.to_empty(device="cpu").half()
    expected = 1.0 / (model.config.rope_theta ** (torch.arange(0, 8, 2).float() / 8))
    for layer in model.model.layers:
        rotary = layer.self_attn.rotary_emb
        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq, expected, atol=0, rtol=0)


def test_nanbeige_real_generation_bridge_trims_positions_and_matches_cache_continuation():
    model = NanbeigeForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 3, 5, 7]])
    with torch.inference_mode():
        cache = model(ids[:, :3], use_cache=True).past_key_values
        prepared = model.prepare_inputs_for_generation(
            ids,
            past_key_values=cache,
            attention_mask=torch.ones_like(ids),
            cache_position=torch.arange(4),
            position_ids=torch.arange(4).unsqueeze(0),
        )
        assert prepared["input_ids"].tolist() == [[7]]
        assert prepared["position_ids"].tolist() == [[3]]
        assert prepared["cache_position"].tolist() == [3]
        actual = model(**prepared).logits
        torch.testing.assert_close(actual, model(ids, use_cache=False).logits[:, -1:], atol=1e-7, rtol=1e-5)
        assert len(cache.layers) == 4
