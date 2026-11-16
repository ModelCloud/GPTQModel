# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models.definitions.falcon_mamba import FalconMambaQModel


def _tiny_falcon_mamba_config():
    from transformers.models.falcon_mamba.configuration_falcon_mamba import FalconMambaConfig

    return FalconMambaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        state_size=16,
        conv_kernel=4,
        time_step_rank=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_falcon_mamba_monkey_patch_tolerates_missing_legacy_symbols():
    """Transformers 5.22 dropped falcon_mamba_inner_fn and the explicit path methods."""

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return "upstream"

    mixer_class = type("FalconMambaMixer", (), {"forward": forward})
    mixer = mixer_class()
    layer = SimpleNamespace(mixer=mixer)
    qmodel = FalconMambaQModel.__new__(FalconMambaQModel)
    qmodel.model = SimpleNamespace(backbone=SimpleNamespace(layers=[layer]))

    qmodel.monkey_patch()

    assert mixer.forward(object()) == "upstream"


def test_falcon_mamba_real_model_monkey_patch_keeps_upstream_forward():
    from transformers.models.falcon_mamba.modeling_falcon_mamba import (
        FalconMambaForCausalLM,
        FalconMambaMixer,
    )

    upstream_forward = FalconMambaMixer.forward
    model = FalconMambaForCausalLM(_tiny_falcon_mamba_config())
    qmodel = FalconMambaQModel.__new__(FalconMambaQModel)
    object.__setattr__(qmodel, "model", model)

    qmodel.monkey_patch()

    assert FalconMambaMixer.forward is upstream_forward


def _build_mixer_with_meta_buffers():
    def __init__(self):
        nn.Module.__init__(self)
        self.intermediate_size = 8
        self.ssm_state_size = 4
        self.D = nn.Parameter(torch.ones(8))
        self.register_buffer("b_c_rms", torch.empty(4, device="meta"), persistent=False)
        self.register_buffer("dt_rms", torch.empty(8, device="meta"), persistent=False)
        for name, size in (("dt_layernorm", 8), ("b_layernorm", 4), ("c_layernorm", 4)):
            norm = nn.Module()
            norm.register_buffer("weight", torch.empty(size, device="meta"), persistent=False)
            self.add_module(name, norm)

    mixer_class = type("FalconMambaMixer", (nn.Module,), {"__init__": __init__})
    return mixer_class()


def test_falcon_mamba_restores_non_persistent_layernorm_weights():
    """5.22 stores {dt,b,c}_layernorm.weight as non-persistent buffers."""

    mixer = _build_mixer_with_meta_buffers()
    model = SimpleNamespace(backbone=SimpleNamespace(layers=[SimpleNamespace(mixer=mixer)]))

    FalconMambaQModel.__new__(FalconMambaQModel)._restore_falcon_mamba_rms_buffers(model)

    assert mixer.dt_layernorm.weight.device.type == "cpu"
    assert mixer.b_layernorm.weight.device.type == "cpu"
    assert mixer.c_layernorm.weight.device.type == "cpu"
    assert torch.equal(mixer.dt_layernorm.weight, torch.ones(8))
    assert torch.equal(mixer.b_layernorm.weight, torch.ones(4))
    assert torch.equal(mixer.c_layernorm.weight, torch.ones(4))
    assert torch.equal(mixer.b_c_rms, torch.ones(4))
    assert torch.equal(mixer.dt_rms, torch.ones(8))
