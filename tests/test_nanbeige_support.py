# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel


def test_nanbeige_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="nanbeige")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert (
        auto.check_and_get_model_definition("nanbeige-fixture", trust_remote_code=True)
        is NanbeigeQModel
    )


def test_nanbeige_uses_llama_compatible_physical_layer_tree():
    layer_modules = NanbeigeQModel.simple_layer_modules(
        model_config=SimpleNamespace(model_type="nanbeige"),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    assert NanbeigeQModel.require_trust_remote_code is True


def test_nanbeige_restores_missing_rotary_frequency_buffer():
    rotary = torch.nn.Module()
    rotary.dim = 8
    rotary.base = 10000.0
    attention = SimpleNamespace(
        rotary_emb=rotary, q_proj=torch.nn.Linear(8, 8, bias=False)
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    )

    NanbeigeQModel._restore_rotary_inv_freq(model)

    torch.testing.assert_close(rotary.inv_freq, torch.tensor([1.0, 0.1, 0.01, 0.001]))
    assert "inv_freq" in dict(rotary.named_buffers())


def test_nanbeige_generation_inputs_trim_cache_positions_to_current_tokens():
    class FakeModel:
        def prepare_inputs_for_generation(self):
            return {
                "input_ids": torch.tensor([[7]]),
                "position_ids": torch.tensor([[0, 1, 2]]),
                "cache_position": torch.tensor([0, 1, 2]),
            }

    NanbeigeQModel._patch_generation_inputs(FakeModel())
    prepared = FakeModel().prepare_inputs_for_generation()

    assert prepared["position_ids"].tolist() == [[2]]
    assert prepared["cache_position"].tolist() == [2]
