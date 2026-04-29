# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from gptqmodel.models.definitions.granitemoehybrid import GraniteMoeHybridQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


class _DummyQuantMamba(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = TorchLinear(
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
        self.out_proj = TorchLinear(
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
        self.path = None
        self.torch_call = None

    def torch_forward(self, input_states, cache_params=None, attention_mask=None):
        self.path = "torch"
        self.torch_call = {
            "input_states": input_states,
            "cache_params": cache_params,
            "attention_mask": attention_mask,
        }
        return "torch"

    def forward(self, *args, **kwargs):
        self.path = "fast"
        return "fast"


def test_granitemoehybrid_quantized_mamba_uses_torch_path():
    qmodel = GraniteMoeHybridQModel.__new__(GraniteMoeHybridQModel)
    qmodel.model = type(
        "_Outer",
        (),
        {
            "model": type(
                "_Inner",
                (),
                {"layers": [type("_Layer", (), {"mamba": _DummyQuantMamba()})()]},
            )()
        },
    )()

    qmodel.monkey_patch()
    mamba = qmodel.model.model.layers[0].mamba

    hidden_states = torch.zeros(1, 2, 64)
    cache_params = object()
    attention_mask = torch.ones(1, 2)

    result = mamba(
        hidden_states,
        cache_params=cache_params,
        cache_position=torch.arange(2),
        attention_mask=attention_mask,
    )

    assert result == "torch"
    assert mamba.path == "torch"
    assert mamba.torch_call == {
        "input_states": hidden_states,
        "cache_params": cache_params,
        "attention_mask": attention_mask,
    }
