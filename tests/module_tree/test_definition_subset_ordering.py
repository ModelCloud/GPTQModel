# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Regression tests for module-tree subset ordering and linear coverage.

Issue #4041: a subset is one quantization/forward-replay block. A consumer
must never share a subset with the producer whose output it consumes, because
the producer's quantized weights are installed after the whole block is
processed. These tests pin the producer/consumer order and the explicit
coverage of every called ``nn.Linear`` for the audited definitions.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import transformers

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)


def _blocks(model_type: str, **config_overrides):
    definition = MODEL_MAP[model_type]
    config = SimpleNamespace(model_type=model_type, **config_overrides)
    if isinstance(definition.dynamic_expert_index, str):
        setattr(config, definition.dynamic_expert_index, 2)
    return definition.simple_layer_modules(model_config=config, quantize_config=QC)


def _subset_indices(blocks) -> dict[str, int]:
    indices: dict[str, int] = {}
    for subset_index, block in enumerate(blocks):
        for raw_name in block:
            name = raw_name.split(":!", 1)[0].split(":?", 1)[0]
            indices.setdefault(name, subset_index)
    return indices


@pytest.mark.parametrize(
    "model_type, expected",
    [
        # `q_b_proj` consumes `q_a_proj`; `o_b_proj` consumes `kv_proj`.
        (
            "deepseek_v4",
            {
                "self_attn.q_a_proj": 0,
                "self_attn.q_b_proj": 1,
                "self_attn.kv_proj": 0,
                "self_attn.o_b_proj": 2,
            },
        ),
        # MLA split mirrors DeepSeek-V3: low-rank producers, then consumers,
        # then the attention output projection.
        (
            "deepseek_v2",
            {
                "self_attn.q_a_proj": 0,
                "self_attn.kv_a_proj_with_mqa": 0,
                "self_attn.q_b_proj": 1,
                "self_attn.kv_b_proj": 1,
                "self_attn.o_proj": 2,
            },
        ),
        # The convolution output projection consumes `in_proj`'s convolution.
        ("lfm2", {"conv.in_proj": 0, "conv.out_proj": 1}),
        # `x_proj` consumes the convolved `in_proj`; `out_proj` consumes the SSM.
        (
            "falcon_mamba",
            {"mixer.in_proj": 0, "mixer.x_proj": 1, "mixer.out_proj": 2},
        ),
        # The gate and QKV read the same input; `out_proj` consumes the gated result.
        (
            "minimax",
            {"self_attn.qkv_proj": 0, "self_attn.output_gate": 0, "self_attn.out_proj": 2},
        ),
    ],
)
def test_consumer_is_quantized_after_its_producer(model_type, expected):
    indices = _subset_indices(_blocks(model_type))

    for name, subset_index in expected.items():
        assert indices.get(name) == subset_index, (model_type, name, indices)


def test_jamba_declares_mamba_x_proj():
    indices = _subset_indices(_blocks("jamba"))

    assert indices["mamba.in_proj"] < indices["mamba.x_proj"] < indices["mamba.out_proj"]


def _tiny_axk2_config():
    from gptqmodel.models.ports.native.axk2.configuration_axk2 import AXK2Config

    return AXK2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        index_head_dim=8,
        index_n_heads=2,
        gated_norm_rank=4,
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=32,
    )


def test_axk2_declares_every_called_linear():
    """Every called ``nn.Linear`` is either quantized or explicitly excluded."""

    definition = MODEL_MAP["axk2"]
    config = _tiny_axk2_config()
    model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers

    plan = ModulePlan.from_tree(definition.module_tree)
    declared = {name.split(":", 1)[0] for block in _blocks("axk2") for name in block}

    for layer in layers:
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            spec = plan.find(name)
            assert spec is not None, f"axk2: undeclared nn.Linear {name}"
            if spec.quantize:
                assert name in declared, f"axk2: quantizable linear missing from subsets: {name}"
            else:
                assert "!" in spec.flags or "?" in spec.flags, (name, spec.flags)

    for excluded in (
        "input_layernorm.mlp.fc1",
        "input_layernorm.mlp.fc2",
        "post_attention_layernorm.mlp.fc1",
        "post_attention_layernorm.mlp.fc2",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
    ):
        spec = plan.find(excluded)
        assert spec is not None, excluded
        assert not spec.quantize, excluded

    # Sanity check the real layers expose the paths the plan declares.
    resolved = set(find_modules(layers[0]))
    assert "self_attn.indexer.wq_b" in resolved
    assert "input_layernorm.mlp.fc1" in resolved


# --- Issue #4372: producer/consumer subset ordering in six definitions. ---


def _blocks_for(definition, model_type: str, **config_overrides):
    config = SimpleNamespace(model_type=model_type, **config_overrides)
    field = definition.dynamic_expert_index
    if isinstance(field, str) and not hasattr(config, field):
        setattr(config, field, 2)
    return definition.simple_layer_modules(model_config=config, quantize_config=QC)


def _assert_after(indices, producer: str, consumer: str) -> None:
    assert indices[producer] < indices[consumer], (
        producer,
        indices[producer],
        consumer,
        indices[consumer],
    )


@pytest.mark.parametrize(
    "model_type, expected",
    [
        (
            "longcat_flash",
            {
                "self_attn.0.q_a_proj": 0,
                "self_attn.0.q_b_proj": 1,
                "self_attn.1.q_b_proj": 6,
                "self_attn.1.o_proj": 7,
                "mlps.0.gate_proj": 3,
                "mlps.0.down_proj": 4,
                "mlps.1.gate_proj": 8,
                "mlps.1.down_proj": 9,
            },
        ),
        (
            "nemotron_h",
            {"mixer.in_proj": 2, "mixer.out_proj": 3},
        ),
        (
            "lfm2_moe",
            {"conv.in_proj": 0, "conv.out_proj": 1},
        ),
        (
            "bailing_moe",
            {"self_attn.kv_a_proj_with_mqa": 1, "self_attn.kv_b_proj": 2, "self_attn.o_proj": 3},
        ),
        (
            "deepseek_vl_v2",
            {
                "self_attn.q_a_proj": 0,
                "self_attn.kv_a_proj_with_mqa": 0,
                "self_attn.q_b_proj": 1,
                "self_attn.kv_b_proj": 1,
                "self_attn.o_proj": 2,
            },
        ),
    ],
)
def test_issue_4372_declarations_follow_forward_order(model_type, expected):
    indices = _subset_indices(_blocks(model_type))
    for name, subset_index in expected.items():
        assert indices.get(name) == subset_index, (model_type, name, indices)


def test_issue_4372_kimi_linear_native_consumer_ordering():
    from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel

    indices = _subset_indices(_blocks_for(KimiLinearNativeQModel, "kimi_linear", num_experts=2))
    for name in (
        "self_attn.g_b_proj",
        "self_attn.forget_gate.f_b_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
    ):
        assert indices[name] == 1, (name, indices)
    assert indices["self_attn.o_proj"] == 2, indices
    for producer in (
        "self_attn.g_b_proj",
        "self_attn.forget_gate.f_b_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
    ):
        _assert_after(indices, producer, "self_attn.o_proj")


def test_issue_4372_nemotron_omni_shares_mixer_ordering():
    from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel

    indices = _subset_indices(_blocks_for(NemotronOmniQModel, "nemotron_h", n_routed_experts=2))
    assert indices["mixer.in_proj"] == 2, indices
    assert indices["mixer.out_proj"] == 3, indices
    _assert_after(indices, "mixer.in_proj", "mixer.out_proj")


# --- Generic data-flow probe: a consumer must be quantized after the Linear
# --- whose output reaches its input. DeepSeek-VL2 is static-only here because
# --- its remote modeling file needs timm; its declaration is pinned above.


def _build_longcat_flow():
    from tests.test_awq_scaling_groups import _tiny_config

    config = _tiny_config("longcat_flash")
    for key, value in {
        "n_routed_experts": 4,
        "zero_expert_num": 0,
        "expert_ffn_hidden_size": 8,
        "num_experts_per_tok": 2,
    }.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(
        config, trust_remote_code=False
    ).eval()
    return model, MODEL_MAP["longcat_flash"], "longcat_flash", {"n_routed_experts": 4}


def _build_kimi_linear_flow():
    from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel
    from tests.test_awq_scaling_groups import _tiny_config

    config = _tiny_config("kimi_linear")
    config.layer_types = ["linear_attention", "full_attention"]
    config.mlp_layer_types = ["dense", "sparse"]
    config.num_experts = 4
    config.num_experts_per_token = 2
    config.num_expert_group = 1
    config.topk_group = 1
    config.num_shared_experts = 1
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.KimiLinearForCausalLM(config).eval()
    return model, KimiLinearNativeQModel, "kimi_linear", {"num_experts": 4}


def _build_nemotron_h_flow():
    from tests.test_awq_scaling_groups import _tiny_config

    config = _tiny_config("nemotron_h")
    config.layers_block_type = ["linear_attention", "full_attention"]
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.NemotronHForCausalLM(config).eval()
    return model, MODEL_MAP["nemotron_h"], "nemotron_h", {"n_routed_experts": 4}


def _build_lfm2_moe_flow():
    from tests.test_awq_scaling_groups import _tiny_config

    config = _tiny_config("lfm2_moe")
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.Lfm2MoeForCausalLM(config).eval()
    return model, MODEL_MAP["lfm2_moe"], "lfm2_moe", {"num_experts": 4}


def _build_bailing_moe_flow():
    config = transformers.BailingHybridConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["full_attention"],
        mlp_layer_types=["dense"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = transformers.BailingHybridForCausalLM(config).eval()
    return (
        model,
        MODEL_MAP["bailing_moe"],
        "bailing_moe",
        {"num_experts": 2, "num_local_experts": 2, "num_shared_experts": 1},
    )


_FLOW_BUILDERS = {
    "longcat_flash": _build_longcat_flow,
    "kimi_linear": _build_kimi_linear_flow,
    "nemotron_h": _build_nemotron_h_flow,
    "lfm2_moe": _build_lfm2_moe_flow,
    "bailing_moe": _build_bailing_moe_flow,
}


_EAGER_MAMBA_KERNELS = {
    "transformers.models.jamba.modeling_jamba": (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "mamba_inner_fn",
        "mamba_selective_scan",
        "mamba_selective_state_update",
    ),
    "transformers.models.nemotron_h.modeling_nemotron_h": (
        "mamba2_chunk_scan",
        "mamba2_split_conv1d_scan_combined",
        "mamba2_selective_state_update",
    ),
}


@contextmanager
def _eager_torch_mamba_kernels():
    """Pin the Torch Mamba reference for one CPU forward.

    ``transformers`` resolves ``mamba_ssm``/``causal_conv1d`` at import time and
    those kernels are CUDA-only. The decorator keeps the undecorated Torch
    reference on ``__wrapped__``; this data-flow probe runs on CPU, so it must
    use that reference rather than hand a CPU tensor to a CUDA kernel.
    """

    import importlib

    saved = []
    try:
        for module_path, names in _EAGER_MAMBA_KERNELS.items():
            module = importlib.import_module(module_path)
            for name in names:
                wrapper = getattr(module, name, None)
                reference = getattr(wrapper, "__wrapped__", None)
                if wrapper is None or reference is None:
                    continue
                saved.append((module, name, wrapper))
                setattr(module, name, reference)
        yield
    finally:
        for module, name, wrapper in reversed(saved):
            setattr(module, name, wrapper)


def _is_routing_dependent(name: str) -> bool:
    return (
        ".experts." in name
        or ".router" in name
        or name.endswith(".gate")
        or name.endswith(".classifier")
    )


def _observed_linear_edges(model, layers, indices):
    """Return ``(producer, consumer)`` Linear pairs seen in a tiny native forward."""

    ids = torch.randint(0, int(model.config.vocab_size), (1, 5))
    edges = set()
    for layer_index, layer in enumerate(layers):
        tracked = {
            name: module
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name in indices
            and not _is_routing_dependent(name)
        }
        if not tracked:
            continue

        captured = {}
        handles = []

        def record(name):
            def hook(module, args, kwargs):
                value = args[0] if args else kwargs.get("hidden_states")
                if isinstance(value, torch.Tensor):
                    captured.setdefault(name, value.detach().clone())

            return hook

        for name, module in tracked.items():
            handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))

        try:
            with _eager_torch_mamba_kernels():
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                baseline = dict(captured)

                for producer_index, (producer, module) in enumerate(tracked.items()):
                    captured.clear()
                    generator = torch.Generator().manual_seed(
                        1000 + 100 * layer_index + producer_index
                    )

                    def perturb(module, args, output, _generator=generator):
                        scale = output.detach().float().std().clamp_min(1e-3)
                        noise = torch.randn(
                            output.shape,
                            generator=_generator,
                            dtype=output.dtype,
                            device=output.device,
                        ) * scale
                        return output + noise

                    handle = module.register_forward_hook(perturb)
                    try:
                        with torch.no_grad():
                            model(input_ids=ids, use_cache=False)
                    finally:
                        handle.remove()

                    for consumer, value in captured.items():
                        if consumer == producer:
                            continue
                        reference = baseline.get(consumer)
                        if reference is None:
                            continue
                        if not torch.allclose(reference, value, atol=1e-6, rtol=1e-4):
                            edges.add((producer, consumer))
        finally:
            for handle in handles:
                handle.remove()
    return edges


@pytest.mark.parametrize("case_id", sorted(_FLOW_BUILDERS))
def test_issue_4372_tiny_model_data_flow(case_id):
    model, definition, model_type, overrides = _FLOW_BUILDERS[case_id]()
    indices = _subset_indices(_blocks_for(definition, model_type, **overrides))
    layers = model.get_submodule(definition.extract_layers_node()[0])
    edges = _observed_linear_edges(model, layers, indices)

    assert edges, f"{case_id}: data-flow probe found no producer/consumer edge"
    violations = [
        (producer, indices[producer], consumer, indices[consumer])
        for producer, consumer in sorted(edges)
        if indices[consumer] <= indices[producer]
    ]
    assert not violations, (
        f"{case_id}: consumer is not quantized after its producer: {violations}"
    )
