# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MiMo-V2-Flash (`mimo_v2_flash`) module-tree contract."""

from types import SimpleNamespace

import pytest

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.mimo_v2_flash import MimoV2FlashQModel
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


def _tiny_mimo_v2_flash():
    transformers = pytest.importorskip("transformers")
    config = transformers.MiMoV2FlashConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.MiMoV2FlashForCausalLM(config).eval()


def test_mimo_v2_flash_is_registered():
    assert MODEL_MAP["mimo_v2_flash"] is MimoV2FlashQModel


def test_mimo_v2_flash_tree_resolves_dense_and_routed_experts():
    pytest.importorskip("defuser")
    from defuser.modeling.update_module import update_module

    model = _tiny_mimo_v2_flash()
    update_module(model, cleanup_original=False)

    layers, layer_names = get_layers_with_prefixes(
        model, MimoV2FlashQModel.extract_layers_node()
    )
    assert layer_names == ["model.layers.0", "model.layers.1"]

    blocks = MimoV2FlashQModel.simple_layer_modules(
        model_config=model.config, quantize_config=SimpleNamespace(dynamic=None)
    )
    resolved = set()
    for layer in layers:
        resolved.update(find_modules(layer))

    missing = []
    for block in blocks:
        for entry in block:
            if ":!" in entry or ":?" in entry:
                continue
            name = entry.split(":")[0]
            if "{expert_index}" in name:
                for expert in range(model.config.n_routed_experts):
                    concrete = name.replace("{expert_index}", str(expert))
                    if concrete not in resolved:
                        missing.append(concrete)
            elif name not in resolved:
                missing.append(name)
    assert not missing, f"unresolved MiMo-V2-Flash specs: {sorted(set(missing))[:10]}"

    assert "mlp.gate_proj" in resolved  # dense layer
    assert "mlp.experts.0.gate_proj" in resolved
    assert "mlp.experts.0.up_proj" in resolved
    assert "mlp.experts.0.down_proj" in resolved
