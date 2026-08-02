# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel import GPTQModel, QuantizeConfig


MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture
def original_inits():
    return (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
    )


def assert_inits_restored(original_inits):
    assert torch.nn.init.kaiming_uniform_ is original_inits[0]
    assert torch.nn.init.uniform_ is original_inits[1]
    assert torch.nn.init.normal_ is original_inits[2]


def test_nn_init_restored_after_from_pretrained(original_inits):
    model = GPTQModel.load(
        MODEL_ID,
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        device="cpu",
    )
    assert model is not None

    assert_inits_restored(original_inits)

    linear = torch.nn.Linear(256, 256)
    assert torch.isfinite(linear.weight).all()
    assert linear.weight.std().item() > 0.001

    tensor = torch.zeros(64, 64)
    torch.nn.init.kaiming_uniform_(tensor, a=5**0.5)
    assert tensor.abs().sum().item() > 0


def test_nn_init_restored_after_from_pretrained_error(original_inits, tmp_path):
    # config-only dir: load enters the monkeypatched region then fails on missing weights
    config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 1,
        "n_positions": 32,
        "vocab_size": 64,
    }
    import json

    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(Exception):
        GPTQModel.load(
            str(tmp_path),
            quantize_config=QuantizeConfig(bits=4, group_size=128),
            device="cpu",
        )

    assert_inits_restored(original_inits)
