# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.definitions.granitemoehybrid import GraniteMoeHybridQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def _torch_linear():
    return TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    )


class _QuantMambaBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = _torch_linear()
        self.out_proj = _torch_linear()
        self.path = None
        self.eager_call = None

    def _record_eager(self, input_states, cache_params, attention_mask):
        self.path = "eager"
        self.eager_call = {
            "input_states": input_states,
            "cache_params": cache_params,
            "attention_mask": attention_mask,
        }
        return "eager"


class _EagerOnlyMamba(_QuantMambaBase):
    """Matches Transformers releases that only expose the eager `forward`."""

    def forward(self, input_states, cache_params=None, attention_mask=None):
        return self._record_eager(input_states, cache_params, attention_mask)


class _LegacyTorchForwardMamba(_QuantMambaBase):
    """Matches older Transformers releases that expose `torch_forward`."""

    def torch_forward(self, input_states, cache_params=None, attention_mask=None):
        return self._record_eager(input_states, cache_params, attention_mask)

    def forward(self, *args, **kwargs):
        self.path = "fast"
        return "fast"


def _qmodel_with_mamba(mamba):
    qmodel = GraniteMoeHybridQModel.__new__(GraniteMoeHybridQModel)
    qmodel.model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mamba=mamba)]))
    return qmodel


def _assert_quantized_forward_uses_eager(mamba):
    _qmodel_with_mamba(mamba).monkey_patch()

    hidden_states = torch.zeros(1, 2, 64)
    cache_params = object()
    attention_mask = torch.ones(1, 2)

    result = mamba(
        hidden_states,
        cache_params=cache_params,
        cache_position=torch.arange(2),
        attention_mask=attention_mask,
    )

    assert result == "eager"
    assert mamba.path == "eager"
    assert mamba.eager_call == {
        "input_states": hidden_states,
        "cache_params": cache_params,
        "attention_mask": attention_mask,
    }


def test_granitemoehybrid_quantized_mamba_uses_eager_forward_without_torch_forward():
    _assert_quantized_forward_uses_eager(_EagerOnlyMamba())


def test_granitemoehybrid_quantized_mamba_uses_torch_forward_when_present():
    _assert_quantized_forward_uses_eager(_LegacyTorchForwardMamba())


def test_real_granitemoehybrid_layer_installs_monkey_patch():
    pytest.importorskip("transformers")
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridMambaLayer,
    )

    original_forward = GraniteMoeHybridMambaLayer.forward
    layer = object.__new__(GraniteMoeHybridMambaLayer)
    qmodel = _qmodel_with_mamba(layer)

    try:
        qmodel.monkey_patch()
        assert GraniteMoeHybridMambaLayer.forward is not original_forward
    finally:
        GraniteMoeHybridMambaLayer.forward = original_forward
