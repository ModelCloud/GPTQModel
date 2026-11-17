# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`kimi_linear` must dispatch between the K3 runner and native Kimi-Linear."""

from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.kimi_k3 import KimiLinearQModel
from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


def _tiny_native_kimi_linear():
    transformers = pytest.importorskip("transformers")
    config = transformers.KimiLinearConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        linear_head_dim=8,
        linear_num_heads=4,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=1,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["linear_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return transformers.KimiLinearForCausalLM(config).eval()


def test_kimi_linear_registry_still_names_the_k3_definition():
    assert MODEL_MAP["kimi_linear"] is KimiLinearQModel


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(model_type="kimi_linear"), KimiLinearNativeQModel),
        (
            SimpleNamespace(model_type="kimi_linear", use_attn_residuals=True),
            KimiLinearQModel,
        ),
        (
            SimpleNamespace(model_type="kimi_linear", attn_res_block_size=12),
            KimiLinearQModel,
        ),
    ],
)
def test_kimi_linear_dispatch_uses_config_fields(monkeypatch, tmp_path, config, expected):
    import json

    (tmp_path / "config.json").write_text(json.dumps(vars(config)))
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *args, **kwargs: False)
    monkeypatch.setattr(auto, "load_model_config", lambda *args, **kwargs: config)

    assert auto.check_and_get_model_definition(tmp_path) is expected


def test_native_kimi_linear_tree_resolves_every_declared_spec():
    pytest.importorskip("defuser")
    from defuser.modeling.update_module import update_module

    model = _tiny_native_kimi_linear()
    # Native Kimi-Linear has no Defuser registry entry; the loader uses the
    # generic experts pass to split the fused 3-D expert parameters.
    update_module(model, cleanup_original=False)

    layers, layer_names = get_layers_with_prefixes(
        model, KimiLinearNativeQModel.extract_layers_node()
    )
    assert layer_names == ["model.layers.0", "model.layers.1"]

    blocks = KimiLinearNativeQModel.simple_layer_modules(
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
                for expert in range(model.config.num_experts):
                    concrete = name.replace("{expert_index}", str(expert))
                    if concrete not in resolved:
                        missing.append(concrete)
            elif name not in resolved:
                missing.append(name)
    assert not missing, f"unresolved Kimi-Linear specs: {sorted(set(missing))[:10]}"

    # Routed and shared experts must both be quantizable after the generic pass.
    assert "mlp.experts.0.gate_proj" in resolved
    assert "mlp.experts.0.up_proj" in resolved
    assert "mlp.experts.0.down_proj" in resolved
    assert "mlp.shared_experts.down_proj" in resolved
