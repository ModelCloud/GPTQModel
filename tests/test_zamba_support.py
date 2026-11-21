# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.zamba import ZambaQModel
from gptqmodel.models.ports.native.zamba.configuration_zamba import ZambaConfig
from gptqmodel.models.ports.native.zamba.modeling_zamba import ZambaForCausalLM


def _tiny_zamba_config() -> ZambaConfig:
    return ZambaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_mamba_heads=4,
        attention_head_dim=32,
        attention_hidden_size=128,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=4,
        max_position_embeddings=128,
        use_mamba_kernels=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


def test_zamba_model_type_selects_definition(tmp_path):
    _tiny_zamba_config().save_pretrained(tmp_path)

    assert auto.check_and_get_model_definition(str(tmp_path)) is ZambaQModel


def test_zamba_module_tree_expands_mamba_and_hybrid_paths():
    layer_modules = ZambaQModel.simple_layer_modules(
        model_config=_tiny_zamba_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}
    full_layer_modules = ZambaQModel.full_layer_modules(model_config=_tiny_zamba_config())
    full_flat_modules = {name for block in full_layer_modules for name in block}

    assert ZambaQModel.layer_modules_strict is False
    assert ZambaQModel.pre_lm_head_norm_module == "model.final_layernorm"
    assert ZambaQModel.extract_layers_node() == ["model.layers"]
    assert "input_layernorm:!" in full_flat_modules
    assert "mamba.conv1d:!" in full_flat_modules
    assert "mamba_decoder.mamba.conv1d:!" in full_flat_modules
    assert "shared_transf:!" in full_flat_modules
    assert "mamba.in_proj" in flat_modules
    assert "mamba.out_proj" in flat_modules
    assert "linear" in flat_modules
    assert "mamba_decoder.mamba.in_proj" in flat_modules
    assert "mamba_decoder.mamba.out_proj" in flat_modules


def test_zamba_base_modules_include_embeddings_and_final_norm():
    model = ZambaForCausalLM(_tiny_zamba_config())
    base_modules = set(ZambaQModel.get_base_modules(model))

    assert "model.embed_tokens" in base_modules
    assert "model.final_layernorm" in base_modules
    assert "model.layers" not in base_modules


def test_zamba_capture_and_replay_kwargs_preserve_original_hidden_states_and_layer_index():
    model_def = ZambaQModel.__new__(ZambaQModel)
    hidden_states = torch.randn(1, 3, 8)
    original_hidden_states = torch.randn(1, 3, 8)
    attention_mask = torch.ones(1, 3)
    causal_mask = torch.zeros(1, 1, 3, 3)

    captured_kwargs = model_def.capture_first_layer_input_kwargs(
        args=(hidden_states, original_hidden_states, 0, attention_mask, causal_mask),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={},
    )

    layer = SimpleNamespace(mamba_decoder=SimpleNamespace(layer_idx=5))
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs=dict(captured_kwargs),
        target_device=torch.device("cpu"),
    )

    assert torch.equal(replay_kwargs["original_hidden_states"], original_hidden_states)
    assert torch.equal(replay_kwargs["attention_mask"], attention_mask)
    assert torch.equal(replay_kwargs["causal_mask"], causal_mask)
    assert replay_kwargs["layer_idx"] == 5
