# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gptqmodel.models.definitions.nemotron_h import NemotronHQModel


def _tiny_nemotron_h_config():
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_head=16,
        mamba_n_heads=4,
        layers_block_type=["mamba", "attention", "mlp", "moe"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_nemotron_h_monkey_patch_tolerates_missing_legacy_paths():
    """5.22's NemotronHMamba2Mixer no longer exposes torch_forward/cuda_kernels_forward."""

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
        return "upstream"

    mixer_class = type("NemotronHMamba2Mixer", (), {"forward": forward})
    mixer = mixer_class()
    layer = SimpleNamespace(mixer=mixer)
    qmodel = NemotronHQModel.__new__(NemotronHQModel)
    qmodel.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    qmodel.load_quantized_model = True

    qmodel.monkey_patch()

    assert mixer.forward(object()) == "upstream"


def test_nemotron_h_real_model_monkey_patch_keeps_upstream_forward():
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHForCausalLM,
        NemotronHMamba2Mixer,
    )

    upstream_forward = NemotronHMamba2Mixer.forward
    model = NemotronHForCausalLM(_tiny_nemotron_h_config())
    qmodel = NemotronHQModel.__new__(NemotronHQModel)
    object.__setattr__(qmodel, "model", model)
    qmodel.load_quantized_model = True

    qmodel.monkey_patch()

    assert NemotronHMamba2Mixer.forward is upstream_forward
