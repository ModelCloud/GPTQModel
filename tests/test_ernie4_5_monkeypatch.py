# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from transformers.models.ernie4_5 import modeling_ernie4_5

from gptqmodel.models.definitions.ernie4_5 import Ernie4_5QModel


class _FakeSelfAttn:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "position_embeddings": position_embeddings,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "kwargs": kwargs,
            }
        )
        return hidden_states, None


class _IdentityModule:
    def __call__(self, hidden_states, *args, **kwargs):
        return hidden_states


class _ResidualAdd:
    def __call__(self, hidden_states, residual):
        return hidden_states + residual


class _FakeLayer:
    __module__ = "transformers.models.ernie4_5.modeling_ernie4_5"

    def __init__(self, with_residual_adds=True):
        self.input_layernorm = _IdentityModule()
        self.self_attn = _FakeSelfAttn()
        self.post_attention_layernorm = _IdentityModule()
        self.mlp = _IdentityModule()
        if with_residual_adds:
            self.residual_add1 = _ResidualAdd()
            self.residual_add2 = _ResidualAdd()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakeRotary:
    def __init__(self):
        self.calls = []
        self.return_value = ("cos", "sin")

    def __call__(self, hidden_states, position_ids=None):
        self.calls.append({"hidden_states": hidden_states, "position_ids": position_ids})
        return self.return_value


class _FakeNorm:
    def __call__(self, hidden_states):
        return hidden_states


class _FakeModel:
    __module__ = "transformers.models.ernie4_5.modeling_ernie4_5"

    def __init__(self, with_residual_adds=True):
        self.config = type("_Cfg", (), {"use_cache": False, "num_hidden_layers": 1})()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.rotary_emb = _FakeRotary()
        self.layers = [_FakeLayer(with_residual_adds=with_residual_adds)]
        self.norm = _FakeNorm()


class _FakeRemoteLayer(_FakeLayer):
    __module__ = "transformers_modules.fake_ernie.modeling_ernie4_5"


class _FakeRemoteModel(_FakeModel):
    __module__ = "transformers_modules.fake_ernie.modeling_ernie4_5"

    def __init__(self, with_residual_adds=True):
        super().__init__(with_residual_adds=with_residual_adds)
        self.layers = [_FakeRemoteLayer(with_residual_adds=with_residual_adds)]


def test_ernie4_5_monkeypatch_forwards_position_embeddings(monkeypatch):
    fake_mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    monkeypatch.setattr(modeling_ernie4_5, "create_causal_mask", lambda **_: fake_mask)

    qmodel = Ernie4_5QModel.__new__(Ernie4_5QModel)
    qmodel.load_quantized_model = False
    qmodel.model = type("_Outer", (), {"model": _FakeModel()})()

    qmodel.monkey_patch()

    result = qmodel.model.model.forward(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        return_dict=True,
    )

    fake_model = qmodel.model.model
    fake_layer = fake_model.layers[0]
    rotary_call = fake_model.rotary_emb.calls[0]
    attn_call = fake_layer.self_attn.calls[0]

    assert result.last_hidden_state.shape == (1, 3, 8)
    assert torch.equal(rotary_call["position_ids"], torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert attn_call["position_embeddings"] == fake_model.rotary_emb.return_value
    assert attn_call["attention_mask"] is fake_mask


def test_ernie4_5_monkeypatch_defaults_to_structured_output(monkeypatch):
    fake_mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    monkeypatch.setattr(modeling_ernie4_5, "create_causal_mask", lambda **_: fake_mask)

    qmodel = Ernie4_5QModel.__new__(Ernie4_5QModel)
    qmodel.load_quantized_model = False
    qmodel.model = type("_Outer", (), {"model": _FakeModel()})()

    qmodel.monkey_patch()

    result = qmodel.model.model.forward(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )

    assert result.last_hidden_state.shape == (1, 3, 8)


def test_ernie4_5_monkeypatch_falls_back_without_residual_add_modules(monkeypatch):
    fake_mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    monkeypatch.setattr(modeling_ernie4_5, "create_causal_mask", lambda **_: fake_mask)

    qmodel = Ernie4_5QModel.__new__(Ernie4_5QModel)
    qmodel.load_quantized_model = False
    qmodel.model = type("_Outer", (), {"model": _FakeModel(with_residual_adds=False)})()

    qmodel.monkey_patch()

    result = qmodel.model.model.forward(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        return_dict=True,
    )

    fake_model = qmodel.model.model
    attn_call = fake_model.layers[0].self_attn.calls[0]

    assert result.last_hidden_state.shape == (1, 3, 8)
    assert attn_call["position_embeddings"] == fake_model.rotary_emb.return_value


def test_ernie4_5_monkeypatch_skips_remote_model_classes():
    original_model_forward = _FakeRemoteModel.forward if hasattr(_FakeRemoteModel, "forward") else None
    original_layer_forward = _FakeRemoteLayer.forward if hasattr(_FakeRemoteLayer, "forward") else None

    qmodel = Ernie4_5QModel.__new__(Ernie4_5QModel)
    qmodel.load_quantized_model = False
    qmodel.model = type("_Outer", (), {"model": _FakeRemoteModel()})()

    qmodel.monkey_patch()

    current_model_forward = _FakeRemoteModel.forward if hasattr(_FakeRemoteModel, "forward") else None
    current_layer_forward = _FakeRemoteLayer.forward if hasattr(_FakeRemoteLayer, "forward") else None

    assert current_model_forward is original_model_forward
    assert current_layer_forward is original_layer_forward
