# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from transformers import AutoModelForCausalLM, LlamaConfig

from gptqmodel.utils.hf import build_shell_model


def _tiny_llama():
    return LlamaConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_build_shell_model_ignores_device_map_and_fast_init_kwargs():
    """build_shell_model is called from from_quantized with HF loader kwargs such
    as device_map and _fast_init that are not accepted by AutoModel.from_config.
    It must strip those kwargs and build the model on meta without raising."""
    config = _tiny_llama()

    shell = build_shell_model(
        AutoModelForCausalLM,
        config,
        trust_remote_code=False,
        device_map={"": "cpu"},
        _fast_init=False,
    )

    assert isinstance(shell, type(AutoModelForCausalLM.from_config(config, trust_remote_code=False)))
    # All parameters and buffers should be meta placeholders.
    first_param = next(shell.parameters())
    assert first_param.device.type == "meta"
    assert first_param.is_meta
    assert shell.model.rotary_emb.inv_freq.is_meta
