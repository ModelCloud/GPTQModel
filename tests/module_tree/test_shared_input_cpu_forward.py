# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Verify shared-input plans against real (tiny, CPU) model forwards.

Each case builds a down-scaled HF model from config, runs a forward pass with input hooks on
every quantizable module of every decoder layer, and asserts the plan derived from the model
definition matches the activations actually observed.
"""

from types import SimpleNamespace

import pytest
import torch
import transformers
from defuser import convert_model

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.shared_input import build_shared_input_plan, probe_shared_inputs


QC = SimpleNamespace(dynamic=None)
VOCAB = 64

_DENSE = {
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": VOCAB,
}


def _cfg(name, **overrides):
    kwargs = dict(_DENSE)
    kwargs.update(overrides)
    return getattr(transformers, name)(**kwargs)


def _linear_attn_cfg(name):
    return getattr(transformers, name)(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=2,
        vocab_size=VOCAB,
    )


# name -> (config factory, expected shared (leader, follower) pairs that must be verified in layer 0)
CASES = {
    "llama": (
        lambda: _cfg("LlamaConfig"),
        [("self_attn.q_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj")],
    ),
    "qwen2": (lambda: _cfg("Qwen2Config"), [("self_attn.q_proj", "self_attn.k_proj")]),
    "qwen3": (lambda: _cfg("Qwen3Config"), [("self_attn.q_proj", "self_attn.k_proj")]),
    "mistral": (lambda: _cfg("MistralConfig"), [("mlp.gate_proj", "mlp.up_proj")]),
    "gemma2": (lambda: _cfg("Gemma2Config", head_dim=16), [("self_attn.q_proj", "self_attn.v_proj")]),
    "gemma3_text": (lambda: _cfg("Gemma3TextConfig", head_dim=16), [("mlp.gate_proj", "mlp.up_proj")]),
    "phi3": (
        lambda: _cfg("Phi3Config", pad_token_id=0, eos_token_id=1, bos_token_id=2),
        [],
    ),
    "qwen3_moe": (
        lambda: _cfg(
            "Qwen3MoeConfig", num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16, decoder_sparse_step=1
        ),
        [("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj"), ("mlp.experts.1.gate_proj", "mlp.experts.1.up_proj")],
    ),
    "qwen2_moe": (
        lambda: _cfg(
            "Qwen2MoeConfig",
            num_experts=2,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            decoder_sparse_step=1,
        ),
        [("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"), ("mlp.experts.1.gate_proj", "mlp.experts.1.up_proj")],
    ),
    "mixtral": (
        lambda: _cfg("MixtralConfig", num_local_experts=2, num_experts_per_tok=2),
        [("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj")],
    ),
    "deepseek_v3": (
        lambda: _cfg(
            "DeepseekV3Config",
            n_routed_experts=2,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=16,
            first_k_dense_replace=1,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=16,
            n_group=1,
            topk_group=1,
        ),
        [("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa")],
    ),
    "glm4_moe": (
        lambda: _cfg(
            "Glm4MoeConfig",
            n_routed_experts=2,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=16,
            first_k_dense_replace=1,
            head_dim=16,
            n_group=1,
            topk_group=1,
        ),
        [("self_attn.q_proj", "self_attn.v_proj")],
    ),
    "qwen3_5_moe": (
        lambda: _linear_attn_cfg("Qwen3_5MoeTextConfig"),
        [("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj")],
    ),
    "qwen3_next": (
        lambda: _linear_attn_cfg("Qwen3NextConfig"),
        [("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj")],
    ),
    "gpt_oss": (
        lambda: _cfg(
            "GptOssConfig",
            num_local_experts=2,
            num_experts_per_tok=2,
            head_dim=16,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        ),
        [("mlp.experts.0.gate_proj", "mlp.experts.0.up_proj")],
    ),
    "llama4_text": (
        lambda: transformers.Llama4TextConfig(
            hidden_size=32,
            intermediate_size=48,
            intermediate_size_mlp=48,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=2,
            vocab_size=VOCAB,
            num_local_experts=2,
            num_experts_per_tok=2,
            head_dim=16,
        ),
        [("feed_forward.experts.0.gate_proj", "feed_forward.experts.0.up_proj")],
    ),
}


def _build(name):
    factory, expected_pairs = CASES[name]
    torch.manual_seed(0)
    cfg = factory()
    qmodel_cls = MODEL_MAP[cfg.model_type]
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    convert_model(model, cleanup_original=False)
    layers = model
    for part in qmodel_cls.extract_layers_node()[0].split("."):
        layers = getattr(layers, part)
    return cfg, qmodel_cls, model, layers, expected_pairs


def _probe_layer(model, layer, plan):
    input_ids = torch.randint(0, VOCAB, (2, 6))
    return probe_shared_inputs(layer, plan, lambda: model(input_ids, use_cache=False))


@pytest.mark.parametrize("name", sorted(CASES))
def test_plan_matches_real_forward_inputs(name):
    cfg, qmodel_cls, model, layers, expected_pairs = _build(name)
    template_plan = qmodel_cls.shared_input_plan(cfg, QC)
    assert template_plan.groups

    for layer_index, layer in enumerate(layers):
        live = set(dict(layer.named_modules()))
        plan = template_plan.filter_modules(live)
        report = _probe_layer(model, layer, plan)

        assert report.ok, f"{name} layer {layer_index}:\n{report.describe()}"
        assert report.missing_modules == ()
        assert report.unverified == (), f"{name} layer {layer_index}: {report.unverified}"
        for module in plan.modules:
            assert report.call_counts[module] >= 1, (name, layer_index, module)

        # Every planned module is a real Linear layer in the tiny model.
        for module in plan.modules:
            assert isinstance(layer.get_submodule(module), torch.nn.Linear), (name, module)

        if layer_index == 0:
            for leader, follower in expected_pairs:
                if leader in live and follower in live:
                    assert plan.shares_input(leader, follower), (name, leader, follower)
                    assert plan.group_for(leader).key in report.verified


@pytest.mark.parametrize("name", ["llama", "qwen3_moe", "deepseek_v3"])
def test_plan_covers_every_quantizable_linear(name):
    cfg, qmodel_cls, model, layers, _ = _build(name)
    template_plan = qmodel_cls.shared_input_plan(cfg, QC)
    non_quantized = {"mlp.gate", "self_attn.q_norm", "self_attn.k_norm"}
    for layer in layers:
        live = dict(layer.named_modules())
        plan = template_plan.filter_modules(live)
        linears = {n for n, m in live.items() if isinstance(m, torch.nn.Linear)} - non_quantized
        assert linears == set(plan.modules)


def test_probe_detects_wrong_tag_on_real_model():
    """Removing the MLA `:in=` split would wrongly group q_b_proj with kv_b_proj; the probe must catch it."""
    cfg, qmodel_cls, model, layers, _ = _build("deepseek_v3")
    untagged_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0", "q_a_proj:0", "kv_a_proj_with_mqa:0", "q_b_proj:1", "kv_b_proj:1", "o_proj:2"),
        },
    ]
    layer_modules = [["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"], ["self_attn.q_b_proj", "self_attn.kv_b_proj"]]
    bad_plan = build_shared_input_plan(untagged_tree, layer_modules)
    assert bad_plan.shares_input("self_attn.q_b_proj", "self_attn.kv_b_proj")

    report = _probe_layer(model, layers[0], bad_plan)
    assert not report.ok
    assert {(m.leader, m.module) for m in report.mismatches} == {("self_attn.q_b_proj", "self_attn.kv_b_proj")}
    assert "self_attn:0" in report.verified


def test_probe_reports_undeclared_sharing_on_real_model():
    """Splitting q_proj/k_proj into different tags is flagged because they receive identical inputs."""
    cfg, qmodel_cls, model, layers, _ = _build("llama")
    tree = ["model", "layers", "#", {"self_attn": ("q_proj:0:in=a", "k_proj:0:in=b", "v_proj:0:in=a")}]
    plan = build_shared_input_plan(tree, [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]])
    report = _probe_layer(model, layers[0], plan)
    assert report.verified == ("self_attn:in=a", "self_attn:in=b")
    assert set(report.undeclared) == {
        ("self_attn.q_proj", "self_attn.k_proj"),
        ("self_attn.v_proj", "self_attn.k_proj"),
    }
    assert not report.ok


def test_moe_unrouted_expert_is_reported_unverified():
    """With top-1 routing over 4 experts and few tokens, some experts never run and stay `unverified`."""
    torch.manual_seed(0)
    cfg = _cfg(
        "Qwen3MoeConfig", num_experts=4, num_experts_per_tok=1, moe_intermediate_size=16, decoder_sparse_step=1
    )
    qmodel_cls = MODEL_MAP[cfg.model_type]
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    convert_model(model, cleanup_original=False)
    layer = model.model.layers[0]
    plan = qmodel_cls.shared_input_plan(cfg, QC).filter_modules(dict(layer.named_modules()))
    input_ids = torch.randint(0, VOCAB, (1, 1))
    report = probe_shared_inputs(layer, plan, lambda: model(input_ids, use_cache=False))
    assert report.ok
    routed = [m for m in plan.modules if m.startswith("mlp.experts.") and report.call_counts[m]]
    assert len(routed) == 3  # exactly one expert (gate/up/down) ran for the single token
    assert len(report.unverified) == 3 * 2  # 3 idle experts x (gate/up group, down group)
