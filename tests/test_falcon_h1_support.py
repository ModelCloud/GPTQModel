# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.models.definitions.falcon_h1 import FalconH1QModel


def _tiny_falcon_h1_config():
    from transformers.models.falcon_h1.configuration_falcon_h1 import FalconH1Config

    return FalconH1Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        mamba_d_ssm=64,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_falcon_h1_restores_non_persistent_mup_vector_after_load():
    """mup_vector is persistent=False, so LazyTurtle leaves it on META."""

    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1ForCausalLM,
        compute_mup_vector,
    )

    config = _tiny_falcon_h1_config()
    expected = compute_mup_vector(config)
    model = FalconH1ForCausalLM(config)

    mixers = [layer.mamba for layer in model.model.layers]
    assert mixers
    for mixer in mixers:
        mixer.register_buffer("mup_vector", torch.empty_like(expected, device="meta"), persistent=False)

    qmodel = FalconH1QModel.__new__(FalconH1QModel)
    restored = qmodel.after_model_load(model, load_quantized_model=False)

    assert restored is model
    for mixer in mixers:
        assert mixer.mup_vector.device.type == "cpu"
        assert torch.equal(mixer.mup_vector, expected)


def test_falcon_h1_restore_keeps_existing_materialized_mup_vector():
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1ForCausalLM,
        compute_mup_vector,
    )

    config = _tiny_falcon_h1_config()
    model = FalconH1ForCausalLM(config)
    mixer = model.model.layers[0].mamba
    sentinel = torch.full_like(compute_mup_vector(config), 3.0)
    mixer.register_buffer("mup_vector", sentinel, persistent=False)

    FalconH1QModel.__new__(FalconH1QModel)._restore_falcon_h1_mup_buffers(model)

    assert torch.equal(mixer.mup_vector, sentinel)
