# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.zamba2 import Zamba2QModel
from gptqmodel.models.ports.native.zamba2.configuration_zamba2 import Zamba2Config
from gptqmodel.models.ports.native.zamba2.modeling_zamba2 import Zamba2ForCausalLM


def _tiny_zamba2_config() -> Zamba2Config:
    return Zamba2Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        n_mamba_heads=4,
        chunk_size=16,
        max_position_embeddings=128,
        layers_block_type=["mamba", "hybrid"],
        num_mem_blocks=1,
        use_shared_attention_adapter=False,
        use_mem_rope=False,
        use_mamba_kernels=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


def test_zamba2_model_type_selects_definition(tmp_path):
    _tiny_zamba2_config().save_pretrained(tmp_path)

    assert auto.check_and_get_model_definition(str(tmp_path)) is Zamba2QModel


def test_zamba2_module_tree_expands_mamba_and_hybrid_paths():
    layer_modules = Zamba2QModel.simple_layer_modules(
        model_config=_tiny_zamba2_config(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}
    full_layer_modules = Zamba2QModel.full_layer_modules(model_config=_tiny_zamba2_config())
    full_flat_modules = {name for block in full_layer_modules for name in block}

    assert Zamba2QModel.layer_modules_strict is False
    assert Zamba2QModel.pre_lm_head_norm_module == "model.final_layernorm"
    assert Zamba2QModel.extract_layers_node() == ["model.layers"]
    assert "input_layernorm:!" in full_flat_modules
    assert "mamba.conv1d:!" in full_flat_modules
    assert "mamba_decoder.mamba.conv1d:!" in full_flat_modules
    assert "shared_transformer:!" in full_flat_modules
    assert "mamba.in_proj" in flat_modules
    assert "mamba.out_proj" in flat_modules
    assert "linear" in flat_modules
    assert "mamba_decoder.mamba.in_proj" in flat_modules
    assert "mamba_decoder.mamba.out_proj" in flat_modules


def test_zamba2_base_modules_include_embeddings_and_final_norm():
    model = Zamba2ForCausalLM(_tiny_zamba2_config())
    base_modules = set(Zamba2QModel.get_base_modules(model))

    assert "model.embed_tokens" in base_modules
    assert "model.final_layernorm" in base_modules
    assert "model.layers" not in base_modules


def test_zamba2_capture_and_replay_kwargs_preserve_original_hidden_states_and_layer_index():
    model_def = Zamba2QModel.__new__(Zamba2QModel)
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

    layer = SimpleNamespace(mamba_decoder=SimpleNamespace(layer_idx=7))
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs=dict(captured_kwargs),
        target_device=torch.device("cpu"),
    )

    assert torch.equal(replay_kwargs["original_hidden_states"], original_hidden_states)
    assert torch.equal(replay_kwargs["attention_mask"], attention_mask)
    assert torch.equal(replay_kwargs["causal_mask"], causal_mask)
    assert replay_kwargs["layer_idx"] == 7
