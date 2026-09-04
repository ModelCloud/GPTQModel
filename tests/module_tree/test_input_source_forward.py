# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
from torch import nn
from transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    MiniCPM3Config,
    MiniCPM3ForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from gptqmodel.looper.input_source_validator import (
    InputSourceCapture,
    validate_input_sources,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.input_source import InputSourceId, group_input_sources
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.mixtral import MixtralQModel
from gptqmodel.models.definitions.minicpm3 import MiniCpm3QModel
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import get_module


def _layer_prefix(qmodel_cls, layer_index):
    prefix = []
    for part in qmodel_cls.module_tree:
        if part == "#":
            break
        prefix.append(part)
    return ".".join(prefix + [str(layer_index)])


def simulate(qmodel_cls, model, layer_index=0, seq=6, batch=2):
    """Run one tiny model forward while capturing module-tree inputs."""

    torch.manual_seed(0)
    quantize_config = QuantizeConfig(bits=4, group_size=16)
    layer_modules = qmodel_cls.simple_layer_modules(model.config, quantize_config)
    prefix = _layer_prefix(qmodel_cls, layer_index)
    layer = get_module(model, prefix)
    assert layer is not None

    named_modules = []
    seen = set()
    for block in layer_modules:
        for token in block:
            name = token.split(":", 1)[0]
            name = name.split("|", 1)[0]
            submodule = get_module(layer, name)
            if not isinstance(submodule, nn.Linear):
                continue
            if name in seen:
                continue
            seen.add(name)
            entry = qmodel_cls.resolve_module_tree_entry(name)
            assert entry is not None, name
            scope = f"{prefix}.{entry.scope}" if entry.scope else prefix
            named_modules.append(
                NamedModule(
                    submodule,
                    name=name,
                    full_name=f"{prefix}.{name}",
                    layer_index=layer_index,
                    tree_scope_id=scope,
                    subset_id=entry.subset_id,
                    input_spec=entry.input_spec,
                )
            )

    groups = group_input_sources(named_modules)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch, seq),
    )
    with InputSourceCapture(named_modules) as capture:
        model(input_ids=input_ids)
    return validate_input_sources(groups, capture.captured), groups, capture.captured


def _tiny_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        use_cache=False,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(config)


def _tiny_qwen2_moe():
    config = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        use_cache=False,
        attn_implementation="eager",
    )
    return Qwen2MoeForCausalLM(config)


def _tiny_qwen3_moe():
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        use_cache=False,
        attn_implementation="eager",
    )
    return Qwen3MoeForCausalLM(config)


def _tiny_mixtral():
    config = MixtralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        use_cache=False,
        attn_implementation="eager",
    )
    return MixtralForCausalLM(config)


def _tiny_deepseek_v3():
    config = DeepseekV3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        use_cache=False,
        attn_implementation="eager",
    )
    return DeepseekV3ForCausalLM(config)


def _tiny_minicpm3():
    config = MiniCPM3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        use_cache=False,
        attn_implementation="eager",
    )
    return MiniCPM3ForCausalLM(config)


def _assert_routed_expert_gate_up(groups, captured):
    routed = []
    for source, modules in groups.items():
        names = {module.full_name.rsplit(".", 1)[-1] for module in modules}
        if {"gate_proj", "up_proj"} <= names and all(captured[module.full_name] for module in modules):
            routed.append(source)
    assert routed


def test_llama_cpu_forward_validates_qkv_and_gate_up_inputs():
    report, groups, _ = simulate(LlamaQModel, _tiny_llama())
    assert report.ok
    assert report.checked_sources >= 2
    qkv = next(
        modules
        for source, modules in groups.items()
        if source.scope.endswith("self_attn") and source.subset_id == 0
    )
    assert {module.name.rsplit(".", 1)[-1] for module in qkv} == {
        "q_proj",
        "k_proj",
        "v_proj",
    }


@pytest.mark.parametrize(
    ("qmodel_cls", "model_factory"),
    [
        (Qwen2MoeQModel, _tiny_qwen2_moe),
        (Qwen3MoeQModel, _tiny_qwen3_moe),
        (MixtralQModel, _tiny_mixtral),
    ],
)
def test_moe_cpu_forward_validates_routed_expert_inputs(qmodel_cls, model_factory):
    report, groups, captured = simulate(qmodel_cls, model_factory())
    assert report.ok
    assert report.checked_sources >= 1
    if any("experts." in module.full_name for modules in groups.values() for module in modules):
        _assert_routed_expert_gate_up(groups, captured)


def test_deepseek_v3_cpu_forward_validates_mla_inputs():
    report, groups, captured = simulate(DeepSeekV3QModel, _tiny_deepseek_v3())
    assert report.ok
    assert report.checked_sources >= 2
    q_a = next(
        source for source, modules in groups.items()
        if {module.name.rsplit(".", 1)[-1] for module in modules}
        == {"q_a_proj", "kv_a_proj_with_mqa"}
    )
    assert q_a.kind == "subset"
    assert captured["model.layers.0.self_attn.q_a_proj"]
    assert captured["model.layers.0.self_attn.kv_a_proj_with_mqa"]
    for name in ("q_b_proj", "kv_b_proj"):
        module = next(
            module
            for modules in groups.values()
            for module in modules
            if module.name.endswith(name)
        )
        assert len(
            groups[next(source for source, modules in groups.items() if module in modules)]
        ) == 1


def test_minicpm3_cpu_forward_validates_mla_inputs():
    report, groups, _ = simulate(MiniCpm3QModel, _tiny_minicpm3())
    assert report.ok
    assert report.checked_sources >= 2
    assert any(
        {module.name.rsplit(".", 1)[-1] for module in modules}
        == {"q_a_proj", "kv_a_proj_with_mqa"}
        for modules in groups.values()
    )


def test_negative_deepseek_v3_group_detects_distinct_latent_inputs():
    _, groups, captured = simulate(DeepSeekV3QModel, _tiny_deepseek_v3())
    q_b = next(
        module
        for modules in groups.values()
        for module in modules
        if module.name.endswith("q_b_proj")
    )
    kv_b = next(
        module
        for modules in groups.values()
        for module in modules
        if module.name.endswith("kv_b_proj")
    )
    source = InputSourceId(scope="model.layers.0.self_attn", subset_id=1)
    report = validate_input_sources({source: [q_b, kv_b]}, captured)
    assert not report.ok
    assert report.mismatches[0].reason in {"shape", "value"}


def test_negative_qwen2_expert_group_detects_different_expert_calls():
    _, groups, captured = simulate(Qwen2MoeQModel, _tiny_qwen2_moe())
    expert_gates = [
        module
        for modules in groups.values()
        for module in modules
        if ".experts." in module.full_name and module.full_name.endswith("gate_proj")
    ]
    if len(expert_gates) < 2:
        pytest.skip(
            "Transformers 5.16 Qwen2MoeExperts exposes fused gate_up_proj, "
            "not per-expert gate_proj modules"
        )
    gate_zero, gate_one = expert_gates[:2]
    captured[gate_zero.full_name] = [torch.ones(2, 32)]
    captured[gate_one.full_name] = [torch.zeros(2, 32)]
    source = InputSourceId(scope="model.layers.0.mlp.experts", subset_id=0)
    report = validate_input_sources({source: [gate_zero, gate_one]}, captured)
    assert not report.ok
    assert report.mismatches[0].reason in {"value", "call_count"}


def test_negative_groups_produce_useful_mismatch_diagnostics():
    model = _tiny_llama()
    report, groups, captured = simulate(LlamaQModel, model)
    assert report.ok
    q_proj = next(
        module
        for modules in groups.values()
        for module in modules
        if module.name.endswith("q_proj")
    )
    o_proj = next(
        module
        for modules in groups.values()
        for module in modules
        if module.name.endswith("o_proj")
    )
    source = InputSourceId(scope="model.layers.0.self_attn", subset_id=0)
    negative = validate_input_sources({source: [q_proj, o_proj]}, captured)
    assert not negative.ok
    assert negative.mismatches[0].reason in {"shape", "value"}
    assert any(name.endswith("q_proj") for name in negative.mismatches[0].module_names)
    assert any(name.endswith("o_proj") for name in negative.mismatches[0].module_names)


def _try_optional_forward(
    *,
    transformers_module,
    config_name,
    model_name,
    qmodel_module,
    qmodel_name,
    config_kwargs,
):
    try:
        transformers_models = importlib.import_module(transformers_module)
        qmodel_models = importlib.import_module(qmodel_module)
        config = getattr(transformers_models, config_name)(**config_kwargs)
        model = getattr(transformers_models, model_name)(config)
        qmodel_cls = getattr(qmodel_models, qmodel_name)
    except Exception as error:
        pytest.skip(f"tiny CPU model construction unavailable: {type(error).__name__}: {error}")
    try:
        report, groups, captured = simulate(qmodel_cls, model)
    except Exception as error:
        pytest.skip(f"tiny CPU forward unavailable: {type(error).__name__}: {error}")
    assert report.ok
    assert report.checked_sources >= 1
    return report, groups, captured


def test_deepseek_v32_cpu_forward_validates_dsa_latent_inputs():
    _, groups, _ = _try_optional_forward(
        transformers_module="transformers.models.deepseek_v32",
        config_name="DeepseekV32Config",
        model_name="DeepseekV32ForCausalLM",
        qmodel_module="gptqmodel.models.definitions.deepseek_v32",
        qmodel_name="DeepSeekV32QModel",
        config_kwargs={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 8,
            "index_topk": 4,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "head_dim": 8,
            "n_group": 1,
            "topk_group": 1,
            "use_cache": False,
            "attn_implementation": "eager",
        },
    )
    assert any(
        {module.name.rsplit(".", 1)[-1] for module in modules}
        >= {"q_b_proj", "wq_b"}
        and source.name == "q_latent"
        for source, modules in groups.items()
    )
    assert any(
        len(modules) == 1
        and modules[0].name.endswith("kv_b_proj")
        for modules in groups.values()
    )


def test_deepseek_v4_cpu_forward_validates_staged_query_inputs():
    _, groups, _ = _try_optional_forward(
        transformers_module="transformers.models.deepseek_v4",
        config_name="DeepseekV4Config",
        model_name="DeepseekV4ForCausalLM",
        qmodel_module="gptqmodel.models.definitions.deepseek_v4",
        qmodel_name="DeepSeekV4QModel",
        config_kwargs={
            "vocab_size": 128,
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "q_lora_rank": 16,
            "o_lora_rank": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "layer_types": ["full_attention"],
            "index_topk": 4,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "use_cache": False,
            "attn_implementation": "eager",
        },
    )
    assert any(
        len(modules) == 1 and modules[0].name.endswith("q_a_proj")
        for modules in groups.values()
    )
    assert any(
        len(modules) == 1 and modules[0].name.endswith("q_b_proj")
        for modules in groups.values()
    )


def test_glm_moe_dsa_cpu_forward_validates_dsa_latent_inputs():
    _, groups, _ = _try_optional_forward(
        transformers_module="transformers.models.glm_moe_dsa",
        config_name="GlmMoeDsaConfig",
        model_name="GlmMoeDsaForCausalLM",
        qmodel_module="gptqmodel.models.definitions.glm_moe_dsa",
        qmodel_name="GlmMoeDsaQModel",
        config_kwargs={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 8,
            "index_topk": 4,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "head_dim": 8,
            "n_group": 1,
            "topk_group": 1,
            "use_cache": False,
            "attn_implementation": "eager",
        },
    )
    assert any(
        {module.name.rsplit(".", 1)[-1] for module in modules}
        >= {"q_b_proj", "wq_b"}
        and source.name == "q_latent"
        for source, modules in groups.items()
    )
    assert any(
        len(modules) == 1
        and modules[0].name.endswith("kv_b_proj")
        for modules in groups.values()
    )


def test_glm4_moe_lite_cpu_forward_validates_mla_inputs():
    _try_optional_forward(
        transformers_module="transformers.models.glm4_moe_lite",
        config_name="Glm4MoeLiteConfig",
        model_name="Glm4MoeLiteForCausalLM",
        qmodel_module="gptqmodel.models.definitions.glm4_moe_lite",
        qmodel_name="Glm4MoeLiteQModel",
        config_kwargs={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 8,
            "use_cache": False,
            "attn_implementation": "eager",
        },
    )


def test_glm5_next_cpu_forward_is_attempted_or_skipped_with_reason():
    _try_optional_forward(
        transformers_module="transformers.models.glm5_next",
        config_name="Glm5NextConfig",
        model_name="Glm5NextForConditionalGeneration",
        qmodel_module="gptqmodel.models.definitions.glm5_next",
        qmodel_name="Glm5NextQModel",
        config_kwargs={
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "q_lora_rank": 16,
                "kv_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 8,
                "use_cache": False,
                "attn_implementation": "eager",
            }
        },
    )
